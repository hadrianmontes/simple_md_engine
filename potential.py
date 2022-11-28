import jax
import jax.numpy as jnp
import numpy as np

SIGMA = 0.5
EPSILON = 0.25

@jax.jit
def calculate_pbc_delta(reference: jnp.ndarray, points: jnp.ndarray,
                        cell: jnp.ndarray) -> jnp.ndarray:
    "Calculates the matrix of displacements between to arrays of points taking into account pbc"
    direct = jnp.linalg.solve(cell.T, (points-reference).T).T
    direct -= jnp.round(direct)
    delta = direct @ cell
    return delta

full_pbc_delta = jax.jit(jax.vmap(calculate_pbc_delta, in_axes=(0, None, None)))

@jax.jit
def squared_distance_matrix(reference: jnp.ndarray, target: jnp.ndarray,
                            cell: jnp.ndarray) -> jnp.ndarray:
    "Calculates the matrix of squared distances between to arrays of points taking into account pbc"
    deltas = full_pbc_delta(reference, target, cell)
    squared_distances = jnp.sum(deltas * deltas, axis=-1)
    return squared_distances

@jax.jit
def distance_matrix(reference: jnp.ndarray, target: jnp.ndarray,
                            cell: jnp.ndarray) -> jnp.ndarray:
    "Calculates the matrix of distances between to arrays of points taking into account pbc"
    deltas = full_pbc_delta(reference, target, cell)
    distances = jnp.linalg.norm(deltas * deltas, axis=-1)
    return distances


@jax.jit
def lj_potential(positions: jnp.ndarray, cell: jnp.ndarray, cutoff: float) -> float:
    r2 = squared_distance_matrix(positions, positions, cell)
    r2 += jnp.eye(positions.shape[0])* (cutoff + 1)**2
    r2 = jnp.where(r2 > cutoff**2, jnp.inf, r2)
    r6 = (SIGMA**6)/(r2 ** 3)
    energy = 4*EPSILON * (r6**2 - r6)
    return energy.sum()/2

_lj_force = jax.jit(jax.grad(lj_potential))

@jax.jit
def lj_force(positions: jnp.ndarray, cell: jnp.ndarray, cutoff: float):
    return -1*_lj_force(positions, cell, cutoff)

if __name__ == "__main__":
    positions = jnp.array([[0, 0, 0], [0.05, 0, 0]])
    box = jnp.eye(3)*3
    cutoff = 1.5
    print(lj_potential(positions, box, cutoff))
    print(lj_force(positions, box, cutoff))
