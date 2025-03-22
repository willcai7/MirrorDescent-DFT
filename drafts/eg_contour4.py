
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
import sys
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from collections import defaultdict
import numpy as np
from benchmarks.benchmark_tool import *
import pickle
from tabulate import tabulate
from src import *


Ns = (101,)
Ls = (10,)
N_poles = 80
N_samples = 10
beta = 96.4
test_size = 20
ratio=1
alpha=0.5
N_vec = np.prod(Ns)
D = jax.device_put(genDiscretizedLaplacianEigenvalues(Ns, Ls))
c_H = jax.device_put(-1/2)
# v_H = jax.device_put(jax.random.normal(jax.random.PRNGKey(key), (N_vec, 1)).flatten())
n_particles = np.ceil(np.prod(Ls) * ratio).astype(int)
external_density = gen_rho_ext(n_particles, Ns)
Y = genDiscretizedYukawaInteractionEigenvalues(Ns, Ls, alpha)
v_H = -fftnProduct(Y, external_density)
v_H = jnp.real(v_H)

Mat = c_H*genDiscretizedLaplacian(Ns, Ls) + jnp.diag(v_H) 
w, V = jnp.linalg.eigh(Mat)
w = jnp.real(w)
# w1 = jnp.sqrt(1/(1+jnp.exp(jnp.clip(beta*jnp.real(w),-700,700))))
# w1 = complex_square_root_fermi_dirac(jnp.clip(beta*jnp.real(w),-600,600))
w1 = complex_square_root_fermi_dirac(beta*jnp.real(w))
P = V@jnp.diag(w1)@V.T 
P = (P + P.T)/2 
errors = []
# Compute energy bounds and contour parameters
energy_min = jnp.max(D)*c_H + jnp.min(v_H)
energy_max = jnp.min(D)*c_H + jnp.max(v_H)

print(f"energy_min: {energy_min}, energy_max: {energy_max}")


shifts, weights = gen_contour(energy_min, energy_max, beta, N_poles,function="fermi_dirac", clip=False)

xs = np.linspace(energy_min, energy_max, 1000) 
ys = complex_square_root_fermi_dirac(beta*xs)
ys = jnp.real(ys)
ys_contour = contour_f(xs, shifts, weights)
print("contour error:",jnp.linalg.norm(ys - ys_contour))
print(ys[:10])
print(ys_contour[:10])

plt.figure()
plt.scatter(shifts.real, shifts.imag)
plt.grid(True)
plt.savefig("shifts.png")
plt.close()


shifts = jax.device_put(shifts)
weights = jax.device_put(weights)
v = jax.random.normal(jax.random.PRNGKey(0), (N_vec, 10))
v = jnp.complex128(v)
res = 0 
for shift, weight in zip(shifts, weights):
    # print(f"shift: {shift}, weight: {weight}")         
    res_temp = shift_inv_system(c_H, v_H, D, v, shift, 1e-10)
    # Compute the direct solution using the shifted system
    direct_sol = jnp.linalg.solve(shift*jnp.eye(N_vec) - Mat, v)
    # Compare with iterative solution
    error = jnp.linalg.norm(direct_sol - res_temp) / jnp.linalg.norm(direct_sol)
    print(f"Relative error for shift {shift:.2e}: {error:.2e}")
    res += weight * res_temp
res = jnp.imag(res) 

true_res = P@v 

print(jnp.linalg.norm(res - true_res)/jnp.linalg.norm(true_res))

