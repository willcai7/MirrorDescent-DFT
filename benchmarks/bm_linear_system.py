# This file is to benchmark the scaling of the inverse linear system solver
# We will consider two cases for two variables: N_vec, N_samples 
#   1. N_vec = 1000, N_samples = linspace(10, 500, 50)
#   2. N_vec = linspace(10, 1000, 100), N_samples = 100

import os
import sys
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp
from src.models.hamiltonian import *
from src.models.contour import *
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

def benchmark_linear_system(Ns, Ls, N_samples, tol=1e-6, test_size=10):
    # Initialize the Hamiltonian
    c_H = 1

    N_vec = np.prod(Ns)    
    v_H = jax.random.normal(jax.random.PRNGKey(0), (N_vec, 1)).flatten()
    D = genDiscretizedLaplacianEigenvalues(Ns, Ls)

    times = np.zeros(test_size)
    errs = np.zeros(test_size)
    for i in range(test_size):
        v = jax.random.normal(jax.random.PRNGKey(i), (N_vec, N_samples))
        v = jnp.complex128(v)
        shift = jnp.complex128(1j+2)
        
        start = time.time()
        u = shift_inv_system(c_H, v_H, D, v, shift, tol)
        end = time.time()
        times[i] = end - start

        temp_v = v_H[:, None] * u
        u = fftnProduct(shift - c_H *D, u)
        u = u - temp_v
        errs[i] = jnp.mean(jnp.abs(u - v))
    
    times = times[1:]
    errs = errs[1:]

    return (times.mean(), times.std()), (errs.mean(), errs.std())


# Scaling of the inverse linear system solver

# 1D scaling of N_samples
Ns = (101,)
Ls = (1,)
N_sampless = np.linspace(0, 500, 51).astype(int)[1:]
times = np.zeros((len(N_sampless), 2))
errs = np.zeros((len(N_sampless), 2))

for i in tqdm(range(len(N_sampless))):
    N_samples = N_sampless[i]
    times[i, :], errs[i, :] = benchmark_linear_system(Ns, Ls, N_samples)

os.makedirs('figures/benchmarks', exist_ok=True)

plt.figure(figsize=(10, 4), dpi=100)
plt.subplot(1, 2, 1)
plt.errorbar(N_sampless, times[:, 0], yerr=times[:, 1], label='Time')
plt.xlabel('N_samples')
plt.ylabel('Time')

plt.subplot(1, 2, 2)
plt.errorbar(N_sampless, errs[:, 0], yerr=errs[:, 1], label='Error')
plt.xlabel('N_samples')
plt.ylabel('Error')

plt.suptitle('1D scaling of shift system solver w.r.t N_samples')
plt.tight_layout()
plt.savefig('figures/benchmarks/bm_shift_system_solver_1D_scaling_N_samples.png')
plt.show()

# 1D scaling of N_vec
Ns = np.linspace(0, 1000, 101).astype(int)[1:]
times = np.zeros((len(Ns), 2))
errs = np.zeros((len(Ns), 2))

for i in tqdm(range(len(Ns))):
    temp_Ns = (Ns[i],)
    times[i, :], errs[i, :] = benchmark_linear_system(temp_Ns, Ls, N_samples) 

plt.figure(figsize=(10, 4), dpi=100)
plt.subplot(1, 2, 1)
plt.errorbar(Ns, times[:, 0], yerr=times[:, 1], label='Time')
plt.xlabel('N_vec')
plt.ylabel('Time')

plt.subplot(1, 2, 2)
plt.errorbar(Ns, errs[:, 0], yerr=errs[:, 1], label='Error')
plt.xlabel('N_vec')
plt.ylabel('Error')

plt.suptitle('1D scaling of shift system solver w.r.t N_vec')
plt.tight_layout()
plt.savefig('figures/benchmarks/bm_shift_system_solver_1D_scaling_N_vec.png')
plt.show()