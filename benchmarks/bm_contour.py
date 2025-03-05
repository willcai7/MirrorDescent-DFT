# This file is to benchmark the scaling of the contour integration method
# We will consider four cases for: N_poles, N_samples, N_vec, beta
#   1. N_poles = 100, N_samples = 100, N_vec = linspace(0, 1000, 20), beta = 10
#   2. N_poles = 100, N_samples = linspace(0, 300, 30), N_vec = 1000, beta = 10
#   3. N_poles = linspace(0, 300, 30), N_samples = 100, N_vec = 1000, beta = 10
#   4. N_poles = 100, N_samples = 100, N_vec = 1000, beta = linspace(0, 10, 20)

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

def benchmark_contour(Ns, Ls, N_poles, N_samples, beta, test_size=10, tol=1e-6):
    # Initialize the Hamiltonian
    D = genDiscretizedLaplacianEigenvalues(Ns, Ls)
    c_H = -1 
    N_vec = np.prod(Ns)
    v_H = jax.random.normal(jax.random.PRNGKey(0), (N_vec, 1)).flatten() 
    energy_min = jnp.max(D)*c_H + jnp.min(v_H)
    energy_max = jnp.min(D)*c_H + jnp.max(v_H)
    shifts, weights = gen_contour(energy_min, energy_max, beta, N_poles)
    
    times = np.zeros(test_size)

    for i in range(test_size):
        v = jax.random.normal(jax.random.PRNGKey(i), (N_vec, N_samples))
        v = jnp.complex128(v)

        start = time.time()
        u = contour_matvec(c_H, v_H, D, v, tol, shifts, weights)
        end = time.time()
        times[i] = end - start

    times = times[1:]
    
    return times.mean(), times.std()



# 1D scaling of N_poles
print('1D scaling of N_poles')

N_poless = np.linspace(0, 300, 31).astype(int)[1:]
N_samples = 100
Ns = (1000,)
Ls = (1,)
beta = 10

times = np.zeros((len(N_poless), 2))

for i in tqdm(range(len(N_poless))):
    N_poles = N_poless[i]
    times[i, :] = benchmark_contour(Ns, Ls, N_poles, N_samples, beta)


os.makedirs('figures/benchmarks', exist_ok=True)

plt.figure(figsize=(6, 4), dpi=100)
plt.errorbar(N_poless, times[:, 0], yerr=times[:, 1], label='Time')
plt.xlabel('N_poles')
plt.ylabel('Time')
plt.suptitle('1D scaling of contour integration method w.r.t N_poles')
plt.tight_layout()
plt.savefig('figures/benchmarks/bm_contour_1D_scaling_N_poles.png')
plt.show()


# 1D scaling of N_samples
print('1D scaling of N_samples')
N_poles = 100
N_sampless = np.linspace(0, 300, 31).astype(int)[1:]
Ns = (1000,)
beta = 10

times = np.zeros((len(N_sampless), 2))

for i in tqdm(range(len(N_sampless))):
    N_samples = N_sampless[i]
    times[i, :] = benchmark_contour(Ns, Ls, N_poles, N_samples, beta)   

plt.figure(figsize=(6, 4), dpi=100)
plt.errorbar(N_sampless, times[:, 0], yerr=times[:, 1], label='Time')
plt.xlabel('N_samples')
plt.ylabel('Time')
plt.suptitle('1D scaling of contour integration method w.r.t N_samples')
plt.tight_layout()
plt.savefig('figures/benchmarks/bm_contour_1D_scaling_N_samples.png')
plt.show()


# 1D scaling of N_vec
print('1D scaling of N_vec')
N_poles = 100
N_samples = 100
N_vecs = np.linspace(0, 1000, 21).astype(int)[1:]
beta = 10

times = np.zeros((len(N_vecs), 2))

for i in tqdm(range(len(N_vecs))):
    temp_Ns = (N_vecs[i],)
    times[i, :] = benchmark_contour(temp_Ns, Ls, N_poles, N_samples, beta)

plt.figure(figsize=(6, 4), dpi=100)
plt.errorbar(N_vecs, times[:, 0], yerr=times[:, 1], label='Time')
plt.xlabel('N_vec')
plt.ylabel('Time')
plt.suptitle('1D scaling of contour integration method w.r.t N_vec')
plt.tight_layout()
plt.savefig('figures/benchmarks/bm_contour_1D_scaling_N_vec.png')
plt.show()


# 1D scaling of beta
print('1D scaling of beta')
N_poles = 100
N_samples = 100
Ns = (1000,)
Ls = (1,)
betas = np.linspace(0, 100, 21)[1:]

times = np.zeros((len(betas), 2))

for i in tqdm(range(len(betas))):
    beta = betas[i]
    times[i, :] = benchmark_contour(Ns, Ls, N_poles, N_samples, beta)

plt.figure(figsize=(6, 4), dpi=100)
plt.errorbar(betas, times[:, 0], yerr=times[:, 1], label='Time')
plt.xlabel('beta')
plt.ylabel('Time')
plt.suptitle('1D scaling of contour integration method w.r.t beta')
plt.tight_layout()
plt.savefig('figures/benchmarks/bm_contour_1D_scaling_beta.png')
plt.show()