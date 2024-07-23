"""
This example implements a 2D Lid-Driven Cavity Flow simulation using the lattice Boltzmann method (LBM). 
The Lid-Driven Cavity Flow is a standard test case for numerical schemes applied to fluid dynamics, which involves fluid in a square cavity with a moving lid (top boundary).

In this example you'll be introduced to the following concepts:

1. Lattice: The simulation employs a D2Q9 lattice. It's a 2D lattice model with nine discrete velocity directions, which is typically used for 2D simulations.

2. Boundary Conditions: The code implements two types of boundary conditions:

    BounceBackHalfway: This condition is applied to the stationary walls (left, right, and bottom). It models a no-slip boundary where the velocity of fluid at the wall is zero.
    EquilibriumBC: This condition is used for the moving lid (top boundary). It defines a boundary with a set velocity, simulating the "driving" of the cavity by the lid.

3. Checkpointing: The simulation supports checkpointing. Checkpoints are saved periodically (determined by the 'checkpoint_rate'), allowing the simulation to be stopped and restarted from the last checkpoint. This can be beneficial for long simulations or in case of unexpected interruptions.

4. Visualization: The simulation outputs data in VTK format for visualization. It also provides images of the velocity field and saves the boundary conditions at each time step. The data can be visualized using software like Paraview.

"""
from jax import config
import numpy as np
import jax.numpy as jnp
import os

from src.boundary_conditions import *
from src.models import BGKSim, KBCSim
from src.lattice import LatticeD2Q9
from src.utils import *

from jax.experimental.multihost_utils import process_allgather
# Use 8 CPU devices
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

class Channel(BGKSim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.IBM_applied = True
    
    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def IBM_collision(self, f, timestep):
        f = self.precisionPolicy.cast_to_compute(f)
        rho, u = self.update_macroscopic(f)
        feq = self.equilibrium(rho, u, cast_output=False)
        fneq = f - feq
        fout = (timestep+1)/(timestep+1)*(f - self.omega * fneq)
        if self.IBM_applied:
            fout = self.apply_IBM_force(fout, feq, rho, u, timestep)
        return self.precisionPolicy.cast_to_output(fout)

    @partial(jit, static_argnums=(0,), inline=True)
    def apply_IBM_force(self, f_postcollision, feq, rho, u, timestep):
        delta_u = self.get_IBM_force(timestep)
        # delta_u = u
        feq_force = self.equilibrium(rho, u + delta_u, cast_output=False)
        f_postcollision = f_postcollision + feq_force - feq
        return f_postcollision

    @partial(jit, static_argnums=(0, 3), donate_argnums=(1,))
    def IBM_step(self, f_poststreaming, timestep, return_fpost=False):
        f_postcollision = self.IBM_collision(f_poststreaming,timestep)
        f_postcollision = self.apply_bc(f_postcollision, f_poststreaming, timestep, "PostCollision")
        f_poststreaming = self.streaming(f_postcollision)
        f_poststreaming = self.apply_bc(f_poststreaming, f_postcollision, timestep, "PostStreaming")

        if return_fpost:
            return f_poststreaming, f_postcollision
        else:
            return f_poststreaming, None

    def get_IBM_force(self,timestep):
        # define the external force
        force = np.zeros((self.nx, self.ny, 2))
        force[..., 0] = Re**2 * visc**2 / 500**3
        return self.precisionPolicy.cast_to_output(force)

    def set_boundary_conditions(self):
        walls = np.concatenate((self.boundingBoxIndices["top"], self.boundingBoxIndices["bottom"]))
        self.BCs.append(BounceBack(tuple(walls.T), self.gridInfo, self.precisionPolicy))

        outlet = self.boundingBoxIndices["right"]
        inlet = self.boundingBoxIndices["left"]

        rho_wall = np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        vel_wall = np.zeros(inlet.shape, dtype=self.precisionPolicy.compute_dtype)
        vel_wall[:, 0] = prescribed_vel*0.0
        self.BCs.append(EquilibriumBC(tuple(inlet.T), self.gridInfo, self.precisionPolicy, rho_wall, vel_wall))

        self.BCs.append(ExtrapolationOutflow(tuple(outlet.T), self.gridInfo, self.precisionPolicy))

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

if __name__ == "__main__":
    precision = "f32/f32"
    lattice = LatticeD2Q9(precision)

    nx = 5000
    ny = 500

    Re = 200.0
    prescribed_vel = 0.1
    clength = ny - 1

    checkpoint_rate = 1000
    checkpoint_dir = os.path.abspath("./checkpoints")

    visc = prescribed_vel * clength / Re
    omega = 1.0 / (3.0 * visc + 0.5)
    
    os.system("rm -rf ./*.vtk && rm -rf ./*.png")

    kwargs = {
        'lattice': lattice,
        'omega': omega,
        'nx': nx,
        'ny': ny,
        'nz': 0,
        'precision': precision,
        'io_rate': 100,
        'print_info_rate': 100,
        'checkpoint_rate': checkpoint_rate,
        'checkpoint_dir': checkpoint_dir,
        'restore_checkpoint': False,
    }

    sim = Channel(**kwargs)
    timestep = 0
    nsf = sim.assign_fields_sharded()
    for i in range(200000):
        nsf, nsfstar = sim.IBM_step(nsf, timestep, False)
        # nsf, nsfstar = sim.step(nsf, timestep, False)
        rho_prev, u_prev = sim.update_macroscopic(nsf)

        
        if i%10 == 0:
            print(i)

        if i%1000==0:
            rho_prev, u_prev = sim.update_macroscopic(nsf)
            rho_prev = downsample_field(rho_prev, sim.downsamplingFactor)
            u_prev = downsample_field(u_prev, sim.downsamplingFactor)
            # Gather the data from all processes and convert it to numpy arrays (move to host memory)
            rho_prev = process_allgather(rho_prev)
            u_prev = process_allgather(u_prev)
            # print("prev shape: ",rho_prev.shape)
            rho = np.array(rho_prev[1:-1, 1:-1])
            # print("rho shape",rho.shape)
            u = np.array(u_prev[1:-1, 1:-1, :])
            save_image(timestep, u)
            fields = {"rho": rho[..., 0], "u_x": u[..., 0], "u_y": u[..., 1]}
            save_fields_vtk(timestep, fields)
        timestep = timestep + 1
    # sim.run(500)