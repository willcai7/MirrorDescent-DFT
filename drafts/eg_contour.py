from src import *
import jax.numpy as jnp
import jax

import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'  # Optional: control memory allocation

# Ns = (8000,)
Ns,Ls = (11,11,11), (10,10,10)
Ns,Ls = (1001,),(10,)
N_vec = np.prod(Ns)
D = genDiscretizedLaplacianEigenvalues(Ns, Ls)
v = jnp.complex128(jax.random.normal(jax.random.PRNGKey(0), (N_vec,20)))
c_H = -1/2
beta = 10
# v_H = jax.random.normal(jax.random.PRNGKey(0), (N_vec,20))
v_H = jax.random.normal(jax.random.PRNGKey(0), (N_vec, 1)).flatten()
energy_min = c_H*jnp.max(D) - 1
energy_max = c_H*jnp.min(D) + 1
N = 20  # Define N for gen_contour
shifts, weights = gen_contour(energy_min, energy_max, beta, N) 
weights *= complex_square_root_fermi_dirac(beta*shifts)

# Track peak memory usage
from jax.experimental.compilation_cache import compilation_cache as cc
cc.set_cache_dir("./jax_cache")  # Optional: cache compiled code

# Get initial memory stats
initial_stats = jax.device_get(jax.devices()[0].memory_stats())
initial_used = initial_stats['peak_bytes_in_use']

# Run the computation
fv = contour_matvec(c_H, v_H, D, v, 1e-10, shifts, weights)

t0 = time.time()
fv = contour_matvec(c_H, v_H, D, v, 1e-10, shifts, weights)
t1 = time.time()
print(f"Time taken: {t1-t0:.2f} seconds")

# Get final memory stats
final_stats = jax.device_get(jax.devices()[0].memory_stats())
peak_used = final_stats['peak_bytes_in_use']

# Calculate and print memory usage
memory_used = peak_used - initial_used
# These memory calculations are correct for both Linux and other platforms
# JAX's memory_stats() returns bytes regardless of OS
print(f"Peak GPU memory used: {memory_used / (1024**2):.2f} MB")
print(f"Total peak memory: {peak_used / (1024**2):.2f} MB")




