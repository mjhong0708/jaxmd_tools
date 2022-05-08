import os
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from functools import cached_property
import pickle

import ase
import ase.data
import ase.io
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from jax_md import quantity

from jaxmd_tools._src import utils
from jaxmd_tools._src.types import Array


@dataclass(eq=True, frozen=True)
class Snapshot:
    """Snapshot of MD simulation."""

    positions: Array
    velocity: Array
    force: Array
    potential_energy: float
    mass: Array
    species: Array
    box: Array

    @cached_property
    def temperature(self):
        kT = quantity.temperature(self.velocity, self.mass.reshape(-1, 1))
        T = utils.kT_inv(kT)
        return T

    @cached_property
    def kinetic_energy(self):
        KE = quantity.kinetic_energy(self.velocity, self.mass.reshape(-1, 1))
        return KE

    def write(self, writer, **kwargs):
        writer.write(self, **kwargs)


class TrajectoryWriter(metaclass=ABCMeta):
    def __init__(self, filename, fractional_coordinates=False):
        self.filename = filename
        self.fractional_coordinates = fractional_coordinates

    def clear(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    @abstractmethod
    def write(self, snapshot, **kwargs):
        """Write a snapshot to the trajectory."""


class PickleWriter(TrajectoryWriter):
    def write(self, snapshot):
        with open(self.filename, "ab") as f:
            pickle.dump(snapshot, f)


class ASETrajWriter(TrajectoryWriter):
    def __init__(self, filename, fractional_coordinates=False):
        super().__init__(filename, fractional_coordinates=fractional_coordinates)
        self.traj = ase.io.Trajectory(filename, "w")

    def write(self, snapshot):
        atoms = to_atoms(snapshot, self.fractional_coordinates)
        self.traj.write(atoms)


class ASEWriter(TrajectoryWriter):
    def write(self, snapshot):
        atoms = to_atoms(snapshot, self.fractional_coordinates)
        ase.io.write(self.filename, atoms, format="xyz", append=True)


def to_atoms(snapshot, fractional_coordinates=False):
    _positions = snapshot.positions
    if fractional_coordinates:
        positions, scaled_positions = None, snapshot.positions
    else:
        positions, scaled_positions = _positions, None
    cell = np.array(snapshot.box)
    numbers = snapshot.species

    atoms = ase.Atoms(
        numbers=numbers,
        positions=positions,
        scaled_positions=scaled_positions,
        cell=cell,
        pbc=True,
        velocities=snapshot.velocity * (1e-2),
    )

    calc = SinglePointCalculator(atoms, energy=snapshot.potential_energy, forces=snapshot.force)
    atoms.calc = calc
    return atoms
