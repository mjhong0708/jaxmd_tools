# jaxmd_tools

> **Note**: This repo is on very early stage, and currently developed for personal use. Please do not depend on this project until it gets stable.

Convenience wrapper for [`jax-md`](https://github.com/google/jax-md) by Google. Uses `ASE` as main interface, and all units are based on `eV` and `Å`.


## Quick example

A simple NVT simulation with stillinger-weber potential for Si bulk:
```python
import os

import ase.build
import ase.io
import jax
import jax.numpy as jnp
from jax_md import energy, space

import jaxmd_tools

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
jax.config.update("jax_enable_x64", True)

# Prepare data
atoms = ase.build.bulk("Si", crystalstructure="diamond", a=5.43, cubic=True).repeat((2, 2, 2))

R_init = jnp.asarray(atoms.get_scaled_positions())
species = jnp.asarray(atoms.get_atomic_numbers())
box = jnp.asarray(atoms.cell.array)
masses = jnp.asarray(atoms.get_masses()) * 1e-4

# Define space & energy function
displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=True)
neighbor_fn, energy_fn = energy.stillinger_weber_neighbor_list(displacement_fn, box,fractional_coordinates=True)

# Simulate
dynamics = jaxmd_tools.MolecularDynamics(
    positions=R_init,
    species=species,
    box=box,
    masses=masses,
    neighbor_fn=neighbor_fn,
    energy_fn=energy_fn,
    ensemble="NVT",
    initial_temperature=300,
    fractional_coordinates=True,
    pressure=0.016021766208,
    traj_writer=jaxmd_tools.io.ASETrajWriter("md_nvt.traj", fractional_coordinates=True),
)
#                                                 in ps                                                
dynamics.run(jax.random.PRNGKey(32), n_steps=100, dt=1e-3, write_every=10)
```

```log
[2022-05-09 10:01:57] - SIMULATION	Running MD with NVT ensemble.
[2022-05-09 10:01:57] - SIMULATION	Number of steps: 100
[2022-05-09 10:01:57] - SIMULATION	Time step: 0.001 ps
[2022-05-09 10:01:57] - SIMULATION	Total simulation time: 0.1 ps
[2022-05-09 10:01:57] - SIMULATION	Initial temperature: 300 K
[2022-05-09 10:01:57] - SIMULATION	Initializing state...
[2022-05-09 10:02:04] - SIMULATION	Start simulation loop.
[2022-05-09 10:02:04] - SIMULATION	Trajectory will be written to md_nvt.traj
[2022-05-09 10:02:06] - SIMULATION	Step 10   T=151.111 K  PE=-276.427 eV  KE=1.250 eV
[2022-05-09 10:02:06] - SIMULATION	Step 20   T=80.029 K  PE=-275.831 eV  KE=0.662 eV
[2022-05-09 10:02:06] - SIMULATION	Step 30   T=162.728 K  PE=-276.491 eV  KE=1.346 eV
[2022-05-09 10:02:06] - SIMULATION	Step 40   T=113.889 K  PE=-276.043 eV  KE=0.942 eV
[2022-05-09 10:02:06] - SIMULATION	Step 50   T=100.492 K  PE=-275.898 eV  KE=0.831 eV
[2022-05-09 10:02:06] - SIMULATION	Step 60   T=184.605 K  PE=-276.535 eV  KE=1.527 eV
[2022-05-09 10:02:06] - SIMULATION	Step 70   T=158.513 K  PE=-276.250 eV  KE=1.311 eV
[2022-05-09 10:02:06] - SIMULATION	Step 80   T=158.184 K  PE=-276.197 eV  KE=1.309 eV
[2022-05-09 10:02:06] - SIMULATION	Step 90   T=195.703 K  PE=-276.454 eV  KE=1.619 eV
[2022-05-09 10:02:06] - SIMULATION	Step 100  T=124.505 K  PE=-275.819 eV  KE=1.030 eV

```