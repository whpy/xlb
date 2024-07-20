from jax import config
import numpy as np
import jax.numpy as jnp
import os

from src.boundary_conditions import *
from src.models import BGKSim, KBCSim, AdvectionDiffusionBGK, HeatBGK
from src.lattice import LatticeD2Q9
from src.utils import *

from jax.experimental.multihost_utils import process_allgather

class Nsstepper(BGKSim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_boundary_conditions(self):
         # concatenate the indices of the left, right, and bottom walls
        walls = np.concatenate((self.boundingBoxIndices["left"], self.boundingBoxIndices["right"], self.boundingBoxIndices["bottom"]))
        # apply bounce back boundary condition to the walls
        self.BCs.append(BounceBackHalfway(tuple(walls.T), self.gridInfo, self.precisionPolicy))

        # apply inlet equilibrium boundary condition to the top wall
        moving_wall = self.boundingBoxIndices["top"]
        print("shape of moving_wall: ",moving_wall.shape)

        rho_wall = np.ones((moving_wall.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        vel_wall = np.zeros(moving_wall.shape, dtype=self.precisionPolicy.compute_dtype)
        print("shape of vel_wall: ",vel_wall.shape)
        
        vel_wall[:, 0] = prescribed_vel
        self.BCs.append(EquilibriumBC(tuple(moving_wall.T), self.gridInfo, self.precisionPolicy, rho_wall, vel_wall))

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs["rho"][1:-1, 1:-1])
        u = np.array(kwargs["u"][1:-1, 1:-1, :])
        print(kwargs['u'].shape)
        timestep = kwargs["timestep"]

        save_image(timestep, u)
        fields = {"rho": rho[..., 0], "u_x": u[..., 0], "u_y": u[..., 1]}
        save_fields_vtk(timestep, fields)
        save_BCs_vtk(timestep, self.BCs, self.gridInfo)

class heat(HeatBGK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fluidstepper = kwargs.get("fluidstepper", None)
    
    def set_boundary_conditions(self):
        # concatenate the indices of the right and bottom walls
        # walls = np.concatenate((self.boundingBoxIndices["left"], self.boundingBoxIndices["right"], self.boundingBoxIndices["bottom"]))

        adia_walls = np.concatenate((self.boundingBoxIndices["right"], self.boundingBoxIndices["bottom"], self.boundingBoxIndices["top"]))
        self.BCs.append(HeatConst(tuple(adia_walls.T), self.gridInfo, self.precisionPolicy, 0.0))

        T_hot_wall = self.boundingBoxIndices["left"]
        self.BCs.append(HeatConst(tuple(T_hot_wall.T), self.gridInfo, self.precisionPolicy, 1.0))

    def output_data(self, **kwargs):
        rho = np.array(kwargs["rho"][1:-1, 1:-1])

        u = np.array(kwargs["u"][1:-1, 1:-1, :])
        timestep = kwargs["timestep"]

        fields = {"rho": rho[..., 0]}
        save_fields_vtk(timestep, fields)
        # save_image(timestep, rho)

if __name__ == "__main__":
    precision = "f32/f32"
    lattice = LatticeD2Q9(precision)

    nx = 512
    ny = 512

    alpha = 1.0
    vel_u = 0.1
    vel_v = 0.4
    u = np.zeros((nx,ny,2))
    u[:,:,0] = vel_u
    u[:,:,1] = vel_v
    u = jnp.array(u)

    clength = nx - 1

    checkpoint_rate = 1000
    checkpoint_dir = os.path.abspath("./heat_checkpoints")

    Re = 200.0
    prescribed_vel = 0.5
    visc1 = prescribed_vel * (nx-1) / Re
    omega1 = 1.0 / (3.0 * visc1 + 0.5)
    print(omega1)

    visc = alpha
    omega = 1.0 / (3.0 * visc + 0.5)
    
    
    os.system("rm -rf ./*.vtk && rm -rf ./*.png")

    kwargs = {
        'lattice': lattice,
        'omega': omega1,
        'nx': nx,
        'ny': ny,
        'nz': 0,
        'precision': precision,
        'io_rate': 5,
        'print_info_rate': 1,
        'checkpoint_rate': checkpoint_rate,
        'checkpoint_dir': checkpoint_dir,
        'restore_checkpoint': False
    }
    
    print("id of u: ",id(u))
    nsstepper = Nsstepper(**kwargs)
    kwargs = {
        'lattice': lattice,
        'omega': omega,
        'nx': nx,
        'ny': ny,
        'nz': 0,
        'precision': precision,
        'io_rate': 5,
        'print_info_rate': 1,
        'checkpoint_rate': checkpoint_rate,
        'checkpoint_dir': checkpoint_dir,
        'restore_checkpoint': False,
        "vel": u,
        "fluidstepper":nsstepper
    }
    sim = heat(**kwargs)
    print("omega is: ",sim.omega)
    
    nsf = sim.fluidstepper.assign_fields_sharded()
    heatf = sim.assign_fields_sharded()

    timestep = 0
    for i in range(1000):
        nsf, nsfstar = sim.fluidstepper.step(nsf, timestep, False)
        rho_prev, u_prev = sim.fluidstepper.update_macroscopic(nsf)
        sim.vel = u_prev
        heatf, heatfstar = sim.step(heatf, timestep, False)
        if i%10 == 0:
            print(i)

        if i%100==0:
            rho_prev = downsample_field(rho_prev, sim.fluidstepper.downsamplingFactor)
            u_prev = downsample_field(u_prev, sim.fluidstepper.downsamplingFactor)
            # Gather the data from all processes and convert it to numpy arrays (move to host memory)
            rho_prev = process_allgather(rho_prev)
            u_prev = process_allgather(u_prev)
            rho = np.array(rho_prev[1:-1, 1:-1])
            # print("rho shape",rho.shape)
            u = np.array(u_prev[1:-1, 1:-1, :])
            save_image(timestep, u)
            fields = {"rho": rho[..., 0], "u_x": u[..., 0], "u_y": u[..., 1]}
            save_fields_vtk(timestep, fields)
        timestep = timestep + 1
    print(id(nsf))

    # rho_prev, u_prev = sim.fluidstepper.update_macroscopic(nsf)
    # u_prev = process_allgather(u_prev)
    # print(u_prev[...,0,0])
    # print(u_prev[...,255,0])
    
    # sim.run(1000)
