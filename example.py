from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax
import numpy as onp
from functools import partial

from engine import step_generator, simple_integrator, leap_frog
from potential import lj_potential
import matplotlib.pyplot as plt
from tqdm import tqdm


cell = jnp.eye(3)*30
cutoff = 2
nsteps = 200000
nreport = 10



n_energies = nsteps // nreport

onp.random.seed(1)
positions = jnp.array(onp.random.random((50,3))*10)
velocities = jnp.zeros_like(positions)

generator = jax.jit(step_generator(simple_integrator, 0.0005, lj_potential, cell, cutoff, nreport))

energies = onp.zeros(n_energies+1)
new_kin = 0
energies[0] = lj_potential(positions, cell, cutoff)

for i in tqdm(range(n_energies)):
    positions, velocities, energy = generator(positions, velocities)
    new_kin = (0.5*(velocities**2)).sum()
    energies[i+1] = energy + new_kin
plt.plot(energies, label="Simple")

onp.random.seed(1)
positions = jnp.array(onp.random.random((50,3))*10)
velocities = jnp.zeros_like(positions)

generator = step_generator(leap_frog, 0.0005, lj_potential, cell, cutoff, nreport)

energies = onp.zeros(n_energies+1)
new_kin = 0
energies[0] = lj_potential(positions, cell, cutoff)
for i in tqdm(range(n_energies)):
    old_kin = new_kin
    positions, velocities, energy = generator(positions, velocities)
    new_kin = (0.5*(velocities**2)).sum()
    energies[i+1] = energy + 0.5*(new_kin+old_kin)
plt.plot(energies, label="Leap Frog")
plt.legend()

plt.xlabel("Step")
plt.ylabel("Energy")

plt.show()