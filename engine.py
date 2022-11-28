import jax
import jax.numpy as jnp


def step_generator(integrator, delta, potential_function, cell, cutoff, nsteps): 
    temp  = jax.jit(jax.grad(potential_function))
    force_function = jax.jit(lambda positions, cell, cutoff: -1*temp(positions, cell, cutoff))
    @jax.jit
    def perform_step(positions, velocities):
        for _ in range(nsteps):
            forces = force_function(positions, cell, cutoff)
            positions, velocities, forces = integrator(positions,velocities, forces, delta)
        potential_energy = potential_function(positions, cell, cutoff)
        return positions, velocities, potential_energy
    return perform_step

@jax.jit
def simple_integrator(positions, velocities, forces, delta):
    positions = positions + velocities * delta
    velocities = velocities + delta*forces
    return positions, velocities, forces


@jax.jit
def leap_frog(positions, velocities, forces, delta):
    velocities = velocities + delta*forces
    positions = positions + velocities * delta
    return positions, velocities, forces


        
        
    