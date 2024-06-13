import os
import optax
import jax
from jax import jit
from functools import partial
import numpy as np
import jax.lax as lax
import jax.numpy as jnp
from jax import vmap
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import flax.linen as nn
from flax.training import checkpoints
from src.boundary_conditions import *
from jax.experimental.multihost_utils import process_allgather
from src.models import BGKSim, KBCSim
from src.lattice import LatticeD2Q9
from src.utils import downsample_field
from typing import List

# jax.config.update("jax_debug_nans", True)
@dataclass
class SimulationParameters:
    nx_lr: int = 76 * 2
    ny_lr: int = 20 * 2
    scaling_factor: int = 6
    nx_hr: int = nx_lr * scaling_factor
    ny_hr: int = ny_lr * scaling_factor
    precision: str = "f32/f32"
    prescribed_velocity: float = 0.05
    Re: List[float] = field(default_factory=lambda: [950, 1000, 1100])
    Re_test: float = 1050
    unrolling_steps: int = 100
    steps: int = 200
    epochs: int = 50
    correction_factor: float = 1.e-5
    learning_rate: float = 1e-3
    l1_coef: float = 0.0
    load_from_checkpoint: bool = False
    batch_size: int = 20
    offset: int = 10000

config = SimulationParameters()

poiseuille_profile  = lambda x,x0,d,umax: np.maximum(0.,4.*umax/(d**2)*((x-x0)*d-(x-x0)**2))


# class Corrector(nn.Module):
#     @nn.compact
#     def __call__(self, x):
#         shape = x.shape
#         x = x.reshape(-1)
#         x = self._dense(x, 32)
#         x = self._dense(x, 64)
#         x = self._dense(x, 64)
#         x = self._dense(x, 64)
#         x = self._dense(x, 64)
#         x = self._dense(x, 32)
#         x = nn.Dense(features=np.prod(shape))(x)
#         return x.reshape(shape)

#     def _dense(self, x, features):
#         x = nn.Dense(features=features, kernel_init=nn.initializers.he_normal(), bias_init=nn.initializers.ones_init())(x)
#         return nn.relu(x)

class ResidualBlock(nn.Module):
    filters: int
    kernel_size: int = 3
    
    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Conv(self.filters, kernel_size=(self.kernel_size, self.kernel_size), 
                    kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.ones_init())(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(self.filters, kernel_size=(self.kernel_size, self.kernel_size), 
                    kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.ones_init())(x)
        return nn.relu(x + residual)


class Corrector(nn.Module):
    layers: int = 4
    @nn.compact
    def __call__(self, x):

        # Initial Conv layer
        x = nn.Conv(32, kernel_size=(5, 5))(x)
        x = nn.relu(x)

        # Residual Blocks
        for _ in range(self.layers):
            x = ResidualBlock(32)(x)
        # Output layer
        x = nn.Conv(2, kernel_size=(5, 5))(x)
        
        return x
    
def prepare_simulation_parameters(nx, ny, Re):
    lattice = LatticeD2Q9(config.precision)
    characteristic_length = nx - 2
    diam = ny // 4
    viscosity = config.prescribed_velocity * characteristic_length / Re
    omega = 1.0 / (3.0 * viscosity + 0.5)

    return {
        'diam': diam,
        'prescribed_velocity': config.prescribed_velocity,
        'lattice': lattice,
        'omega': omega,
        'nx': nx,
        'ny': ny,
        'nz': 0,
        'precision': config.precision
    }

class Cylinder(BGKSim):
    def __init__(self, Re, corrector=None, **kwargs):
        self.diam = kwargs.get('diam')
        self.corrector = corrector
        self.Re = Re
        self.prescribed_vel = kwargs.get('prescribed_velocity')
        super().__init__(**kwargs)

    def set_boundary_conditions(self):
        # Define the cylinder surface
        coord = np.array([(i, j) for i in range(self.nx) for j in range(self.ny)])
        xx, yy = coord[:, 0], coord[:, 1]
        cx, cy = self.nx / 4, self.ny / 2
        cylinder = (xx - cx)**2 + (yy-cy)**2 <= (self.diam/2.)**2
        cylinder = coord[cylinder]
        self.BCs.append(BounceBackHalfway(tuple(cylinder.T), self.gridInfo, self.precisionPolicy))
        # wall = np.concatenate([cylinder, self.boundingBoxIndices['top'], self.boundingBoxIndices['bottom']])
        # self.BCs.append(BounceBack(tuple(wall.T), self.gridInfo, self.precisionPolicy))

        outlet = self.boundingBoxIndices['right']
        rho_outlet = np.ones(outlet.shape[0], dtype=self.precisionPolicy.compute_dtype)
        self.BCs.append(ExtrapolationOutflow(tuple(outlet.T), self.gridInfo, self.precisionPolicy))
        # self.BCs.append(Regularized(tuple(outlet.T), self.gridInfo, self.precisionPolicy, 'pressure', rho_outlet))

        inlet = self.boundingBoxIndices['left']
        rho_inlet = np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        vel_inlet = np.zeros(inlet.shape, dtype=self.precisionPolicy.compute_dtype)
        yy_inlet = yy.reshape(self.nx, self.ny)[tuple(inlet.T)]
        vel_inlet[:, 0] = poiseuille_profile(yy_inlet,
                                             yy_inlet.min(),
                                             yy_inlet.max()-yy_inlet.min(), 3.0 / 2.0 * self.prescribed_vel)
        # self.BCs.append(EquilibriumBC(tuple(inlet.T), self.gridInfo, self.precisionPolicy, rho_inlet, vel_inlet))
        self.BCs.append(Regularized(tuple(inlet.T), self.gridInfo, self.precisionPolicy, 'velocity', vel_inlet))

        wall = np.concatenate([self.boundingBoxIndices['top'], self.boundingBoxIndices['bottom']])
        self.BCs.append(BounceBack(tuple(wall.T), self.gridInfo, self.precisionPolicy))

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    @partial(jax.checkpoint, static_argnums=(0,))
    def collision(self, f, params=None):
        f = self.precisionPolicy.cast_to_compute(f)
        rho, u = self.compute_macroscopic(f)
        feq = self.equilibrium(rho, u, cast_output=False)
        fneq = f - feq
        fout = f - self.omega * fneq
        if params is not None:
            # Re_channel = self.Re * jnp.ones_like(u[..., :1])
            # x = jnp.concatenate([u, Re_channel], axis=-1)
            # force = config.correction_factor * self.corrector.apply(params, x)
            force = config.correction_factor * self.corrector.apply(params, u)

            fout = self.apply_force(force, fout, feq, rho, u)

        return self.precisionPolicy.cast_to_output(fout)
    

    @partial(jit, static_argnums=(0, 4))
    def step(self, f_poststreaming, timestep, params=None, return_fpost=False):
        f_postcollision = self.collision(f_poststreaming, params)
        f_postcollision = self.apply_bc(f_postcollision, f_poststreaming, timestep, "PostCollision")
        f_poststreaming = self.streaming(f_postcollision)
        f_poststreaming = self.apply_bc(f_poststreaming, f_postcollision, timestep, "PostStreaming")

        if return_fpost:
            return f_poststreaming, f_postcollision
        else:
            return f_poststreaming, None
        
    @partial(jit, static_argnums=(0,))
    def step_vmapped(self, f_poststreaming, timestep, params=None, return_fpost=False):
        # vmap using the step function
        step_vmap = vmap(self.step, in_axes=(0, None, None, None))
        f_poststreaming, f_postcollision = step_vmap(f_poststreaming, timestep, params, return_fpost)
        
        return f_poststreaming, f_postcollision
    
    @partial(jit, static_argnums=(0,))
    def compute_macroscopic_vmapped(self, f):
        # vmap using the step function
        rho, u = vmap(self.compute_macroscopic, in_axes=(0))(f)
        return rho, u

class Dataset(object):
    def __init__(self, simulation_lr, simulation_hr):
        self.simulation_lr_data = None
        self.simulation_hr_data = None
        self.generate_data(simulation_lr, simulation_hr)

    def generate_data(self, simulation_lr, simulation_hr):
        
        f_lr = simulation_lr.assign_fields_sharded()
        f_hr = simulation_hr.assign_fields_sharded()
        
        lr_data_list = []
        hr_data_list = []

        print("Generating low-resolution data...")
        req_range = config.steps + config.offset + config.batch_size
        for step in range(config.steps + config.offset + config.batch_size):
            f_lr, _ = simulation_hr.step(f_lr, step)

            if step >= config.offset:
                lr_data_list.append(np.array(process_allgather(f_lr)))
        print("Low-resolution data generated!")

        print("Generating high-resolution data...")
        req_range = config.steps + config.batch_size + config.unrolling_steps + config.offset
        for step in range(req_range):
            for i in range(config.scaling_factor):
                f_hr, _ = simulation_hr.step(f_hr, step + i)
            
            if step >= config.offset:
                f_hr_downsampled = downsample_field(f_hr, config.scaling_factor, method='bicubic')
                hr_data_list.append(np.array(process_allgather(f_hr_downsampled)))

        print("High-resolution data generated!")
        self.simulation_lr_data = np.array(lr_data_list)
        self.simulation_hr_data = np.array(hr_data_list)

        u_lr = simulation_lr.compute_macroscopic_vmapped(self.simulation_lr_data[-1, 1:-1, 1:-1, :])[1]     
        u_hr = simulation_hr.compute_macroscopic_vmapped(self.simulation_hr_data[-1, 1:-1, 1:-1, :])[1]
        error = np.mean((u_lr[1:-1, 1:-1, ...] - u_hr[1:-1, 1:-1, ...])**2)
        print("Order of error to beat ~", error)

    def get_lr_data(self, step, batch_size):
        if step >= self.simulation_lr_data.shape[0]:
            raise ValueError("Data not available for the given step")

        end_step = step + batch_size
        if end_step > self.simulation_lr_data.shape[0]:
            raise ValueError("Window size exceeds available data length")

        return self.simulation_lr_data[step:end_step]

    def get_hr_data(self, step, batch_size):
        if step >= self.simulation_hr_data.shape[0]:
            raise ValueError("Data not available for the given step")

        end_step = step + batch_size
        if end_step > self.simulation_hr_data.shape[0]:
            raise ValueError("Window size exceeds available data length")

        return self.simulation_hr_data[step:end_step]

def train_model(optimizer, dataset, params_nn, optimizer_state, simulation_lr, simulation_hr):
    params = params_nn
    for epoch in range(config.epochs):
        epoch_loss = 0
        number_of_batches = config.steps // config.batch_size
        for batch_id in range(number_of_batches):
            params, optimizer_state, loss = update(batch_id, dataset, params, optimizer, optimizer_state, simulation_lr, simulation_hr)
            epoch_loss += loss

            print(f"Epoch {epoch + 1}, Batch: {batch_id} / {number_of_batches}, Loss: {loss}")   
        
        average_loss = epoch_loss / number_of_batches
        print(f"Epoch {epoch + 1}, Average Loss over all steps: {average_loss}")
        
    print(f"Training done for {config.epochs} epochs")

    return params

def update(batch_id, dataset, params, optimizer, optimizer_state, simulation_lr, simulation_hr):
    batch_size_hr =  config.batch_size + config.unrolling_steps
    batch_size_lr =  config.batch_size

    step = batch_id * config.batch_size

    f_hr = dataset.get_hr_data(step, batch_size_hr)
    f_lr_init = dataset.get_hr_data(step, batch_size_lr)
    loss, grad = jax.value_and_grad(loss_fn)(params, simulation_lr, simulation_hr, f_lr_init, f_hr)
    # scaled_grad = jax.tree_map(lambda g: g / 1000, grad)

    updates, optimizer_state = optimizer.update(grad, optimizer_state)
    params = optax.apply_updates(params, updates)
    return params, optimizer_state, loss

def loss_fn(params, simulation_lr, simulation_hr, f_lr_init, f_hr):
    error = 0
    batch_size = f_lr_init.shape[0]
    f_lr_corrected = f_lr_init
    for i in range(config.unrolling_steps):
        f_lr_corrected, _ = simulation_lr.step_vmapped(f_lr_corrected, 0, params)

        u_lr_corrected = simulation_lr.compute_macroscopic_vmapped(f_lr_corrected)[1]
        u_hr = simulation_hr.compute_macroscopic_vmapped(f_hr[i + 1:i + 1 + batch_size, ...])[1]
        l2_x = jnp.mean((u_lr_corrected[..., 0] - u_hr[..., 0])**2)
        l2_y = jnp.mean((u_lr_corrected[..., 1] - u_hr[..., 1])**2)
        # l1_x = jnp.mean(jnp.abs(u_lr_corrected[:, 1:-1, 1:-1, 0] - u_hr[:, 1:-1, 1:-1, 0]))
        # l1_y = jnp.mean(jnp.abs(u_lr_corrected[:, 1:-1, 1:-1, 1] - u_hr[:, 1:-1, 1:-1, 1]))
        error += (l2_x + l2_y) # + l1_x + l1_y
    # l1_penalty = 0
    # for p in jax.tree_util.tree_leaves(params):
    #     l1_penalty += jnp.sum(jnp.abs(p))

    return error / config.unrolling_steps#+ config.l1_coef * l1_penalty


def test_error(params, simulation_lr, simulation_hr):
    print(f"Testing for {config.Re_test}...")
    params_lr = prepare_simulation_parameters(config.nx_lr, config.ny_lr, config.Re_test)
    params_hr = prepare_simulation_parameters(config.nx_hr, config.ny_hr, config.Re_test)
    corrector = Corrector()
    simulation_lr = Cylinder(config.Re_test, corrector=corrector, **params_lr)
    simulation_hr = Cylinder(config.Re_test, **params_hr) 
    dataset = Dataset(simulation_lr, simulation_hr)

    f_lr_corrected = dataset.get_hr_data(0, 1)[0]
    f_lr = dataset.get_hr_data(0, 1)[0]
    print("Initial data setup finished.")

    mean_error_with_corrector = []
    mean_error_without_corrector = []
    print("Starting main simulation loop...")

    for timestep in range(config.steps):
        f_lr_corrected, _ = simulation_lr.step(f_lr_corrected, timestep, params)
        f_lr, _ = simulation_lr.step(f_lr, timestep)

        u_lr_corrected = simulation_lr.compute_macroscopic(f_lr_corrected[1:-1, 1:-1, :])[1]
        u_lr = simulation_lr.compute_macroscopic(f_lr[1:-1, 1:-1, :])[1]
        u_hr = simulation_hr.compute_macroscopic(dataset.get_hr_data(timestep+1, 1)[0][1:-1, 1:-1, :])[1]

        u_lr_corrected_magnitude = np.sqrt(u_lr_corrected[..., 0]**2 + u_lr_corrected[..., 1]**2)
        u_lr_magnitude = np.sqrt(u_lr[..., 0]**2 + u_lr[..., 1]**2)
        u_hr_magnitude = np.sqrt(u_hr[..., 0]**2 + u_hr[..., 1]**2)

        mean_error_with_corrector.append(np.mean(np.abs(u_hr_magnitude - u_lr_corrected_magnitude)))
        mean_error_without_corrector.append(np.mean(np.abs(u_hr_magnitude - u_lr_magnitude)))

    print("Main simulation loop finished.")

    error_with_corrector = np.abs(u_hr_magnitude - u_lr_corrected_magnitude)
    error_without_corrector = np.abs(u_hr_magnitude - u_lr_magnitude)

    max_error = np.max([error_without_corrector.max(), error_with_corrector.max()])
    min_error = np.min([error_without_corrector.min(), error_with_corrector.min()])
    print("Maximum error calculated: ", max_error)
    print("Minimum error calculated: ", min_error)

    max_velocity = np.max([u_lr_corrected_magnitude.max(), u_lr_magnitude.max(), u_hr_magnitude.max()])
    min_velocity = np.min([u_lr_corrected_magnitude.min(), u_lr_magnitude.min(), u_hr_magnitude.min()])
    print("Maximum velocity calculated: ", max_velocity)
    print("Minimum velocity calculated: ", min_velocity)

    print("Printing all error averages...")
    print("Error with corrector: ", mean_error_with_corrector[-1])
    print("Error without corrector: ", mean_error_without_corrector[-1])
    print("Error without corrector / Error with corrector: ", mean_error_without_corrector[-1] / mean_error_with_corrector[-1])

    print("Generating error plots...")
    # Figure for errors
    fig_error = plt.figure(figsize=(10, 4))
    ax1 = fig_error.add_subplot(1, 2, 1)
    im1 = ax1.imshow(error_with_corrector.T, cmap="jet", origin='lower', vmin=min_error, vmax=max_error)
    ax1.set_title("Error low-res with corrector")
    plt.colorbar(im1, ax=ax1)
    
    ax2 = fig_error.add_subplot(1, 2, 2)
    im2 = ax2.imshow(error_without_corrector.T, cmap="jet", origin='lower', vmin=min_error, vmax=max_error)
    ax2.set_title("Error low-res without corrector")
    plt.colorbar(im2, ax=ax2)

    fig_error.savefig("error.png", dpi=600)

    np.save('u_lr_corrected_magnitude.npy', u_lr_corrected_magnitude)
    np.save('u_lr_magnitude.npy', u_lr_magnitude)
    np.save('u_hr_magnitude.npy', u_hr_magnitude)
    np.save('error_with_corrector.npy', error_with_corrector)
    np.save('error_without_corrector.npy', error_without_corrector)

    # Figure for u_magnitude
    fig_u_magnitude = plt.figure(figsize=(16, 4))
    
    ax0 = fig_u_magnitude.add_subplot(1, 3, 1)
    im0 = ax0.imshow(u_hr_magnitude.T, cmap="jet", origin='lower', vmin=min_velocity, vmax=max_velocity)
    ax0.set_title("Reference (high-res)")
    plt.colorbar(im0, ax=ax0)

    ax1 = fig_u_magnitude.add_subplot(1, 3, 2)
    im1 = ax1.imshow(u_lr_corrected_magnitude.T, cmap="jet", origin='lower', vmin=min_velocity, vmax=max_velocity)
    ax1.set_title("Low-res with corrector")
    plt.colorbar(im1, ax=ax1)

    ax2 = fig_u_magnitude.add_subplot(1, 3, 3)
    im2 = ax2.imshow(u_lr_magnitude.T, cmap="jet", origin='lower', vmin=min_velocity, vmax=max_velocity)
    ax2.set_title("Low-res without corrector")
    plt.colorbar(im2, ax=ax2)

    fig_u_magnitude.savefig("u_magnitude.png", dpi=600)

    fig_mean_error = plt.figure(figsize=(8, 8))
    plt.plot(range(config.steps), mean_error_with_corrector, label='With Corrector')
    plt.plot(range(config.steps), mean_error_without_corrector, label='Without Corrector')

    plt.xlabel('Timesteps', fontsize=16)
    plt.ylabel(r'Mean $L_2$ Error', fontsize=16)
    plt.legend(fontsize=14)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    fig_mean_error.savefig("mean_error.png", dpi=600)


def main():
    corrector = Corrector()
    params_nn = corrector.init(jax.random.PRNGKey(0), jnp.zeros((config.nx_lr, config.ny_lr, 2)))
    if config.load_from_checkpoint:
        print("Loading checkpoint...")
        params_nn = checkpoints.restore_checkpoint('./', params_nn)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params_nn))
    print(f"Total number of trainable parameters: {param_count}")

    optimizer = optax.adam(config.learning_rate)
    optimizer_state = optimizer.init(params_nn)

    for Re in config.Re:
        print(f"Training for Re = {Re}...")
        params_lr = prepare_simulation_parameters(config.nx_lr, config.ny_lr, Re)
        params_hr = prepare_simulation_parameters(config.nx_hr, config.ny_hr, Re)
        simulation_lr = Cylinder(Re, corrector=corrector, **params_lr)
        simulation_hr = Cylinder(Re, **params_hr)
        
        dataset = Dataset(simulation_lr, simulation_hr)
        params_nn = train_model(optimizer, dataset, params_nn, optimizer_state, simulation_lr, simulation_hr)
        
    print("Saving checkpoint...")
    absolute_path = os.path.abspath('./')
    checkpoints.save_checkpoint(absolute_path, params_nn, config.epochs, overwrite=True)
    print("Checkpoint saved!")

    return params_nn, simulation_lr, simulation_hr

if __name__ == "__main__":
    os.system("rm -rf ./*.vtk && rm -rf ./*.png")
    params_nn, simulation_lr, simulation_hr  = main()
    test_error(params_nn, simulation_lr, simulation_hr)