from typing import Callable, Union, Tuple
import jax.numpy as jnp
from jax_md.partition import NeighborList
from jax_md.simulate import (
    NPTNoseHooverState,
    NVEState,
    NVTLangevinState,
    NVTNoseHooverState,
)

Array = jnp.ndarray
MDState = Union[NVEState, NVTLangevinState, NVTNoseHooverState, NPTNoseHooverState]
StateAndNeighbors = Tuple[NVTNoseHooverState, NeighborList]

EnergyFn = Callable[..., Array]
