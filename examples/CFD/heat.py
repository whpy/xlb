from jax import config
import numpy as np
import jax.numpy as jnp
import os

from src.boundary_conditions import *
from src.models import BGKSim, KBCSim, AdvectionDiffusionBGK
from src.lattice import LatticeD2Q9
from src.utils import *

class heat(AdvectionDiffusionBGK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
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

    nx = 256
    ny = 256

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

    visc = alpha
    omega = 1.0 / (3.0 * visc + 0.5)
    
    
    os.system("rm -rf ./*.vtk && rm -rf ./*.png")

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
        "vel": u
    }
    
    sim = heat(**kwargs)
    print("omega is: ",sim.omega)
    sim.run(1000)
