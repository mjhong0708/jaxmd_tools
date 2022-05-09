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
atoms = ase.build.bulk("Si", crystalstructure="diamond", a=5.43, cubic=True).repeat((5, 5, 5))

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
    traj_writer=jaxmd_tools.io.ASETrajWriter("md_nvt.traj", fractional_coordinates=True),
)
#                                                    in ps                                                
dynamics.run(jax.random.PRNGKey(32), n_steps=5000, dt=1e-3, write_every=500)
```

```log
[2022-05-09 10:08:04] - SIMULATION	Running MD with NVT ensemble.
[2022-05-09 10:08:04] - SIMULATION	Number of steps: 5000
[2022-05-09 10:08:04] - SIMULATION	Time step: 0.001 ps
[2022-05-09 10:08:04] - SIMULATION	Total simulation time: 5.0 ps
[2022-05-09 10:08:04] - SIMULATION	Initial temperature: 300 K
[2022-05-09 10:08:04] - SIMULATION	Initializing state...
[2022-05-09 10:08:04] - SIMULATION	Start simulation loop.
[2022-05-09 10:08:04] - SIMULATION	Trajectory will be written to md_nvt.traj
[2022-05-09 10:08:06] - SIMULATION	Step 500   T=222.556 K  PE=-4306.503 eV  KE=28.768 eV
[2022-05-09 10:08:07] - SIMULATION	Step 1000  T=341.019 K  PE=-4302.845 eV  KE=44.080 eV
[2022-05-09 10:08:08] - SIMULATION	Step 1500  T=313.636 K  PE=-4300.885 eV  KE=40.541 eV
[2022-05-09 10:08:09] - SIMULATION	Step 2000  T=282.477 K  PE=-4294.374 eV  KE=36.513 eV
[2022-05-09 10:08:10] - SIMULATION	Step 2500  T=307.574 K  PE=-4297.625 eV  KE=39.757 eV
[2022-05-09 10:08:11] - SIMULATION	Step 3000  T=284.591 K  PE=-4296.865 eV  KE=36.786 eV
[2022-05-09 10:08:12] - SIMULATION	Step 3500  T=256.655 K  PE=-4293.984 eV  KE=33.175 eV
[2022-05-09 10:08:13] - SIMULATION	Step 4000  T=295.389 K  PE=-4297.385 eV  KE=38.182 eV
[2022-05-09 10:08:14] - SIMULATION	Step 4500  T=296.050 K  PE=-4296.823 eV  KE=38.267 eV
[2022-05-09 10:08:15] - SIMULATION	Step 5000  T=308.599 K  PE=-4298.198 eV  KE=39.890 eV
```