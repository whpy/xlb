import os
import optax # optimizer library based on jax
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
from src.models import BGKSim, KBCSim, AdvectionDiffusionBGK
from src.lattice import LatticeD2Q9
from src.utils import downsample_field
from typing import List
from PIL import Image, ImageDraw, ImageFont

# jax.config.update("jax_debug_nans", True)
@dataclass
class SimulationParameters:
    nx: int = 300
    ny: int = 300
    precision: str = "f32/f32"
    steps: int = 200
    epochs: int = 300
    correction_factor: float = 1e-2
    learning_rate: float = 1e-4
    load_from_checkpoint: bool = True
    bump_size = 1e-3
    omega = 1.0

config = SimulationParameters()

class Initializer(nn.Module):
    @nn.compact
    def __call__(self, x):
        shape = x.shape
        x = x.reshape(-1)
        x = self._dense(x, 32)
        x = self._dense(x, 64)
        x = self._dense(x, 32)
        x = nn.Dense(features=np.prod(shape))(x)
        x = x.reshape(shape)
        # x = nn.tanh(x)
        return x


    def _dense(self, x, features):
        x = nn.Dense(features=features, kernel_init=nn.initializers.he_normal(), bias_init=nn.initializers.zeros_init())(x)
        return nn.leaky_relu(x)

def prepare_simulation_parameters(nx, ny, omega):
    lattice = LatticeD2Q9(config.precision)
    return {
        'lattice': lattice,
        'omega': omega,
        'nx': nx,
        'ny': ny,
        'nz': 0,
        'precision': config.precision,
    }


class Block(BGKSim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

def create_XLB_field(nx, ny, bump_size):
    image = Image.new('RGB', (nx, ny), 'white')
    draw = ImageDraw.Draw(image)

    font_size = ny // 3
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    text = "XLB"

    text_width = draw.textlength(text, font=font)
    text_height = font_size
    x = (nx - text_width) // 2
    y = (ny - text_height) // 2

    draw.text((x, y), text, font=font, fill='black')

    gray_image = image.convert('L')
    threshold = 200
    binary_image = gray_image.point(lambda p: p > threshold and 255)

    field = np.array(binary_image) / 255.0 
    field = 1.0 - field  
    field *= bump_size
    field += 1.0

    field = field[:, :, np.newaxis]

    return field


def train_model(params_nn, initializer_nn, optimizer, optimizer_state, simulation):
    params = params_nn
    for epoch in range(config.epochs):
        epoch_loss = 0
        params, optimizer_state, loss = update(params, initializer_nn, optimizer, optimizer_state, simulation)
        epoch_loss += loss
        
        average_loss = epoch_loss / config.epochs
        print(f"Epoch {epoch + 1}, Average Loss over all steps: {average_loss}")
        
    print(f"Training done for {config.epochs} epochs")

    return params


def update(params, initializer_nn, optimizer, optimizer_state, simulation):
    rho_init = jnp.ones((config.nx, config.ny, 1))
    u_init = jnp.zeros((config.nx, config.ny, 2))
    desired_rho = create_XLB_field(config.nx, config.ny, config.bump_size)
    
    def loss_fn(params, rho_init, u_init, desired_rho):
        rho_init += config.correction_factor * initializer_nn.apply(params, rho_init)
        f = simulation.equilibrium(rho_init, u_init)
        for i in range(config.steps):
            f, _ = simulation.step(f, i)

        rho, u = simulation.compute_macroscopic(f)
        error_l2 = jnp.sum((rho - desired_rho)**2)

        return error_l2

    loss, grad = jax.value_and_grad(loss_fn)(params, rho_init, u_init, desired_rho)

    updates, optimizer_state = optimizer.update(grad, optimizer_state)
    params = optax.apply_updates(params, updates)
    return params, optimizer_state, loss

def visualize_result(params_nn, initializer_nn, simulation):
    rho_init = jnp.ones((config.nx, config.ny, 1))
    u_init = jnp.zeros((config.nx, config.ny, 2))
    rho_init += config.correction_factor * initializer_nn.apply(params_nn, rho_init)

    fig = plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(rho_init[:, :, 0], cmap='viridis')
    plt.savefig(f'init_{str(0).zfill(4)}.png', dpi=600, bbox_inches='tight')

    f = simulation.equilibrium(rho_init, u_init)
    rho_init, u_init = simulation.compute_macroscopic(f)
    for i in range(config.steps):
        f, _ = simulation.step(f, i)
        # rho, u = simulation.compute_macroscopic(f)

        # if i % 1 == 0:
        #     fig, ax = plt.subplots(figsize=(10, 10))
        #     plt.axis('off') 
        #     im = ax.imshow(rho[:, :, 0], cmap='viridis')

        #     # divider = make_axes_locatable(ax)
        #     # cax = divider.append_axes('right', size='5%', pad=0.05)
        #     # fig.colorbar(im, cax=cax, orientation='vertical')

        #     plt.savefig(f'simulation_results_{str(i).zfill(4)}.png', dpi=600, bbox_inches='tight')
        #     plt.close(fig)

    rho_final, u_final = simulation.compute_macroscopic(f)
    desired_rho = create_XLB_field(config.nx, config.ny, config.bump_size)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im1 = axes[0].imshow(rho_init[:, :, 0], cmap='viridis')
    axes[0].set_title('Initial Density Field')
    divider1 = make_axes_locatable(axes[0])
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax1, orientation='vertical')

    im2 = axes[1].imshow(desired_rho[:, :, 0], cmap='viridis')
    axes[1].set_title('Ground Truth Density Field')
    divider2 = make_axes_locatable(axes[1])
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax2, orientation='vertical')

    im3 = axes[2].imshow(rho_final[:, :, 0], cmap='viridis')
    axes[2].set_title('Final Density Field')
    divider3 = make_axes_locatable(axes[2])
    cax3 = divider3.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im3, cax=cax3, orientation='vertical')

    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.20, hspace=0.20)

    plt.savefig('simulation_results.png', dpi=600, bbox_inches='tight')
    plt.show()

def main():
    initializer_nn = Initializer()
    params_nn = initializer_nn.init(jax.random.PRNGKey(0), jnp.zeros((config.nx, config.ny, 1)))
    if config.load_from_checkpoint:
        print("Loading checkpoint...")
        absolute_path = os.path.abspath('./')
        params_nn = checkpoints.restore_checkpoint(absolute_path, params_nn)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params_nn))
    print(f"Total number of trainable parameters: {param_count}")

    optimizer = optax.adam(config.learning_rate)
    optimizer_state = optimizer.init(params_nn)

    params_sim = prepare_simulation_parameters(config.nx, config.ny, config.omega)
    simulation = Block(**params_sim)  
      
    params_nn = train_model(params_nn, initializer_nn, optimizer, optimizer_state, simulation)
        
    print("Saving checkpoint...")
    absolute_path = os.path.abspath('./')
    checkpoints.save_checkpoint(absolute_path, params_nn, config.epochs, overwrite=True)
    print("Checkpoint saved!")

    return params_nn, initializer_nn, simulation

if __name__ == "__main__":
    os.system("rm -rf ./*.vtk && rm -rf ./*.png")
    params_nn, initializer_nn, simulation = main()
    visualize_result(params_nn, initializer_nn, simulation)