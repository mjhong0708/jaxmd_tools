from functools import partial
from typing import Callable, Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax import jit, lax
from jax_md import simulate, space
from jax_md.partition import NeighborFn, NeighborList
from jax_md.simulate import (
    NPTNoseHooverState,
    NVEState,
    NVTLangevinState,
    NVTNoseHooverState,
)

from jaxmd_tools._src import utils
from jaxmd_tools._src import io

Array = np.ndarray
MDState = Union[NVEState, NVTLangevinState, NVTNoseHooverState, NPTNoseHooverState]
StateAndNeighbors = Tuple[NVTNoseHooverState, NeighborList]


class MolecularDynamics:
    """MD simulation.

    Args:
        positions: initial positions.
        species: atomic numbers of atoms.
        box: simulation box.
        masses: masses of atoms in unit of eV, Å and ps.
        neighbor_fn: neighbor list function.
        energy_fn: energy function.
        ensemble: ensemble type. NVT, NVE, NPT are possible.
        initial_temperature: initial temperature. For NVE, this value determines initial velocity.
        fractional_coordinates: whether the coordinates are fractional.
        traj_writer: trajectory writer.

    """

    def __init__(
        self,
        positions: Array,
        species: Array,
        box: Array,
        masses: Array,
        neighbor_fn: NeighborFn,
        energy_fn: Callable,
        ensemble: str,
        initial_temperature: float,
        fractional_coordinates: bool = False,
        traj_writer: io.TrajectoryWriter = None,
        **kwargs,
    ):
        self.positions = jnp.asarray(positions)
        self.species = jnp.asarray(species)
        self.box = jnp.asarray(box)
        self.masses = jnp.asarray(masses)
        self.neighbor_fn = neighbor_fn
        self.energy_fn = energy_fn
        self.ensemble = ensemble
        # Initial temperature. For NVE, this value determines initial velocity.
        # For other ensembles, it is used in thermostat.
        self.T0 = initial_temperature
        self.kT0 = utils.kT(T=self.T0)

        self.fractional_coordinates = fractional_coordinates
        if traj_writer is not None:
            self.traj_writer = traj_writer
        else:
            self.traj_writer = io.ASETrajWriter("md.traj", fractional_coordinates)
        # self.traj_writer.clear()

        # TODO: change it to periodic_general for NPT case

        self.displacement_fn, self.shift_fn = space.periodic_general(
            box, fractional_coordinates=fractional_coordinates
        )
        self.neighborlist = neighbor_fn.allocate(positions, kwargs.get("extra_capacity", 10))
        self.kwargs = kwargs

    def _get_simulate_fns(self, dt):
        if self.ensemble.upper() == "NVE":
            init_fn, apply_fn = simulate.nve(self.energy_fn, self.shift_fn, dt)

        elif self.ensemble.upper() == "NVT":
            init_fn, apply_fn = simulate.nvt_nose_hoover(
                self.energy_fn, self.shift_fn, dt, self.kT0
            )

        elif self.ensemble.upper() == "NPT":
            if not self.fractional_coordinates:
                raise ValueError("NPT is only supported for fractional coordinates.")
            try:
                pressure = self.kwargs.get("pressure")
            except KeyError:
                raise ValueError("NPT requires pressure.")
            init_fn, apply_fn = simulate.npt_nose_hoover(
                self.energy_fn, self.shift_fn, dt, pressure, self.kT0
            )

        else:
            raise ValueError(f"Unknown ensemble {self.ensemble}.")
        return init_fn, apply_fn

    def run(self, prng_key, n_steps, dt, write_every=10, log_file: str = None):
        """
        Args:
            prng_key: JAX PRNG key.
            n_steps: number of steps.
            dt: time step in ps.
            write_every: write trajectory every `write_every` steps.
            log_file: log file. If None, log is written to only stdout.
        """
        log = partial(utils.log, task="SIMULATION", filename=log_file)
        log(f"Running MD with {self.ensemble} ensemble.")
        log(f"Number of steps: {n_steps}")
        log(f"Time step: {dt} ps")
        log(f"Total simulation time: {n_steps * dt} ps")
        log(f"Initial temperature: {self.T0} K")
        _step_pad = int(np.log10(n_steps)) + 2
        init_fn, apply_fn = self._get_simulate_fns(dt)
        log("Initializing state...")
        if self.ensemble.upper() == "NPT":
            state = init_fn(
                prng_key,
                self.positions,
                self.box,
                mass=self.masses,
                kT=self.kT0,
                neighbor=self.neighborlist,
            )
        else:
            state = init_fn(
                prng_key,
                self.positions,
                mass=self.masses,
                kT=self.kT0,
                neighbor=self.neighborlist,
            )

        @jit
        def step_fn(i: int, state_and_nbrs: StateAndNeighbors) -> StateAndNeighbors:
            del i
            state, neighbors = state_and_nbrs
            neighbors = neighbors.update(state.position)
            state = apply_fn(state, neighbor=neighbors)
            return state, neighbors

        _energy_fn = jit(self.energy_fn)

        log("Start simulation loop.")
        log(f"Trajectory will be written to {self.traj_writer.filename}")
        step = 0
        while step < (n_steps // write_every):
            new_state, self.neighborlist = lax.fori_loop(
                0, write_every, step_fn, (state, self.neighborlist)
            )
            if self.neighborlist.did_buffer_overflow:
                log("Neighborlist overflowed, reallocating.")
                self.neighborlist = self.neighbor_fn.allocate(new_state.position)
            else:
                state = new_state
                step += 1
            PE = _energy_fn(state.position, neighbor=self.neighborlist)

            if self.ensemble.upper() == "NPT":
                curr_box = simulate.npt_box(state)
                snapshot = io.Snapshot(
                    state.position,
                    state.velocity,
                    state.force,
                    PE,
                    self.masses,
                    self.species,
                    curr_box,
                )
                vol = jnp.dot(curr_box[0], jnp.cross(curr_box[1], curr_box[2]))
                log_msg = (
                    f"Step {str(step*write_every).ljust(_step_pad)} "
                    f"T={snapshot.temperature:.3f} K  PE={PE:.3f} eV  KE={snapshot.kinetic_energy:.3f} eV "
                    f"Vol={vol:.3f} Å^3"
                )
            else:
                snapshot = io.Snapshot(
                    state.position,
                    state.velocity,
                    state.force,
                    PE,
                    self.masses,
                    self.species,
                    self.box,
                )
                log_msg = (
                    f"Step {str(step*write_every).ljust(_step_pad)} "
                    f"T={snapshot.temperature:.3f} K  PE={PE:.3f} eV  KE={snapshot.kinetic_energy:.3f} eV"
                )
            log(log_msg)
            self.traj_writer.write(snapshot)
