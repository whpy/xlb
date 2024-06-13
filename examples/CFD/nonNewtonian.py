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

# Use 8 CPU devices
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

class Cavity(BGKSim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Tau = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.output_dtype, init_val=1.0/self.omega)
        self.Dxx = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.output_dtype, init_val=1.0/self.omega)
        self.Dyy = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.output_dtype, init_val=1.0/self.omega)
        self.Dxy = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.output_dtype, init_val=1.0/self.omega)
        
        self.NNPower = kwargs.get("NNPower", 1.0)
        self.NNnu = kwargs.get("NNnu")
        self.preheat = 20
        # print("shape of attribute Tau: ", self.Tau.shape)
        # for i in range(10):
        #     print("shape of attribute Tau: ", self.Tau[0,i,0])
        
    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, f):
        """
        BGK collision step for lattice.

        The collision step is where the main physics of the LBM is applied. In the BGK approximation, 
        the distribution function is relaxed towards the equilibrium distribution function.
        """
        f = self.precisionPolicy.cast_to_compute(f)
        print("shape of distribution function is: {}".format(f.shape))
        print("type of f is: {}".format(type(f)))
        rho, u = self.update_macroscopic(f)
        feq = self.equilibrium(rho, u, cast_output=False)
        # print(feq.shape)
        fneq = f - feq
        # print(fneq.shape)
        # self.Tau[0,0,0] = 1.0/self.omega

        # relaxation time set
        # self.Tau = self.Tau.at[:,:,:].set(1.0/self.omega)
        # Dtmp = self.StressTensor(fneq, rho, self.Tau)
        # print("D shape ", Dtmp.shape)
        # self.Tau = self.GetTau(Dtmp, rho)
        # print("Tau shape",self.Tau.shape)
        # end

        fout = f - 1.0/self.Tau * fneq
        # fout = f - self.omega * fneq
        if self.force is not None:
            fout = self.apply_force(fout, feq, rho, u)
        
        # relaxation time set
        # self.Tau = self.Tau.at[:,:,:].set(1.0/self.omega)
        Dtmp = self.StressTensor(fneq, rho, self.Tau)
        print("D shape ", Dtmp.shape)
        self.Tau = self.GetTau(Dtmp, rho)
        print("Tau shape",self.Tau.shape)
        # end
        return self.precisionPolicy.cast_to_output(fout)

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def StressTensor(self, fneq, rho, Tau):
        Tau = self.precisionPolicy.cast_to_compute(Tau)
        rho = self.precisionPolicy.cast_to_compute(rho)
        fneq = self.precisionPolicy.cast_to_compute(fneq)

        Dxx = fneq[:,:,3] + fneq[:,:,5] + fneq[:,:,7] + fneq[:,:,4] + fneq[:,:,6] + fneq[:,:,8]
        Dyy = fneq[:,:,1] + fneq[:,:,4] + fneq[:,:,7] + fneq[:,:,2] + fneq[:,:,5] + fneq[:,:,8]
        Dxy = fneq[:,:,7] + fneq[:,:,8] - fneq[:,:,4] - fneq[:,:,5]
        Dxx = jnp.expand_dims(Dxx,-1)
        Dyy = jnp.expand_dims(Dyy,-1)
        Dxy = jnp.expand_dims(Dxy,-1)

        print("rho shape ",rho.shape)
        print("Dxx shape ",Dxx.shape)
        # - 1 / (2* tau_{n-1} * rho * cs^2 * c^2 * dt) 
        coeff = -1.5/rho/Tau
        return self.precisionPolicy.cast_to_output(coeff*jnp.sqrt(2.0*(Dxx*Dxx + Dyy*Dyy + 2*Dxy*Dxy + 1e-20)))
    
    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def GetTau(self, DGamma, rho):
        DGamma = self.precisionPolicy.cast_to_compute(DGamma)
        rho = self.precisionPolicy.cast_to_compute(rho)
        # (nu0 * gamma^{n-1})/(cs^2 * c*c *dt) + 0.5 = 3 * ((nu0 * gamma^{n-1})) + 0.5
        return 3*self.NNnu* jnp.pow(DGamma, self.NNPower-1)/rho + 0.5


    def set_boundary_conditions(self):

        # concatenate the indices of the left, right, and bottom walls
        walls = np.concatenate((self.boundingBoxIndices["left"], self.boundingBoxIndices["right"], self.boundingBoxIndices["bottom"]))
        # apply bounce back boundary condition to the walls
        self.BCs.append(BounceBackHalfway(tuple(walls.T), self.gridInfo, self.precisionPolicy))

        # apply inlet equilibrium boundary condition to the top wall
        moving_wall = self.boundingBoxIndices["top"]

        rho_wall = np.ones((moving_wall.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        vel_wall = np.zeros(moving_wall.shape, dtype=self.precisionPolicy.compute_dtype)
        vel_wall[:, 0] = prescribed_vel
        self.BCs.append(EquilibriumBC(tuple(moving_wall.T), self.gridInfo, self.precisionPolicy, rho_wall, vel_wall))

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs["rho"][1:-1, 1:-1])
        u = np.array(kwargs["u"][1:-1, 1:-1, :])
        timestep = kwargs["timestep"]

        # Tau = np.array(self.Tau[:,:,0])
        # np.savetxt("./{}_Tau.csv".format(timestep),Tau, delimiter=",")
        save_image(timestep, u)
        # fields = {"rho": rho[..., 0], "u_x": u[..., 0], "u_y": u[..., 1]}
        # save_fields_vtk(timestep, fields)
        # save_BCs_vtk(timestep, self.BCs, self.gridInfo)

if __name__ == "__main__":
    precision = "f32/f32"
    lattice = LatticeD2Q9(precision)

    nx = 256
    ny = 256

    Re = 400.0
    prescribed_vel = 0.1
    clength = nx - 1

    # definition of non newtonian parameters
    NNPower = 1.0
    NNnu = pow(prescribed_vel,2 - NNPower)*pow(clength,NNPower)/Re

    checkpoint_rate = 1000
    checkpoint_dir = os.path.abspath("./checkpoints")

    visc = prescribed_vel * clength / Re
    omega = 1.0 / (3.0 * visc + 0.5)
    
    
    os.system("rm -rf ./*.vtk && rm -rf ./*.png")

    kwargs = {
        'lattice': lattice,
        'omega': 1.0 / (3.0 * NNnu + 0.5),
        'nx': nx,
        'ny': ny,
        'nz': 0,
        'precision': precision,
        'io_rate': 100,
        'print_info_rate': 10,
        'checkpoint_rate': checkpoint_rate,
        'checkpoint_dir': checkpoint_dir,
        'restore_checkpoint': False,
        'NNPower': NNPower,
        'NNnu' : NNnu
    }
    
    sim = Cavity(**kwargs)
    print("omega is: ",sim.omega)
    sim.run(50000)
    print("preheat: ",sim.preheat)
