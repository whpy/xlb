from jax import config
import numpy as np
import jax.numpy as jnp
import os

from src.boundary_conditions import *
from src.models import BGKSim, KBCSim, AdvectionDiffusionBGK
from src.lattice import LatticeD2Q9
from src.utils import *

