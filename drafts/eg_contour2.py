from src import *
import jax.numpy as jnp
import jax

import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'  # Optional: control memory allocation

# Ns = (8000,)
Ns = (101,101,101)
Ls = (100,100,100)
N_vec = np.prod(Ns)
ratio = 1
alpha = 0.5
beta = 10
tol = 1e-3
n_particles = np.ceil(np.prod(Ls) * ratio).astype(int)
# n_particles = 100
external_density = gen_rho_ext(n_particles, Ns)
Y = genDiscretizedYukawaInteractionEigenvalues(Ns, Ls, alpha)
D = genDiscretizedLaplacianEigenvalues(Ns, Ls)
c_H = -1/2
v_H = -fftnProduct(Y, external_density)
time_start = time.time()
v_H = -fftnProduct(Y, external_density)
print(f"Time taken: {time.time() - time_start} seconds")
print(jnp.sum(v_H))
v_H = v_H.real
# v_H = jnp.zeros(N_vec)
energy_min = c_H*jnp.max(D) + np.min(v_H) 
energy_max = c_H*jnp.min(D) + np.max(v_H)
N = 10  # Define N for gen_contour
v = jnp.complex128(jax.random.normal(jax.random.PRNGKey(0), (N_vec,10)))

shifts, weights = gen_contour(energy_min, energy_max, beta, N) 
weights1 = weights * complex_square_root_fermi_dirac(beta*shifts)
weights2 = weights * complex_bin_entropy_fermi_dirac(beta*shifts)
combined_weights = jnp.stack([weights1, weights2], axis=1) # [N_poles, 2]
# fv_square_root = contour_matvec(c_H, v_H, D, v, 1e-5, shifts, weights)
# fv_square_root = contour_matvec(c_H, v_H, D, v, 1e-5, shifts, weights)
print("Time for contour")
time_start = time.time()
fv_combined = contour_matvec(c_H, v_H, D, v, tol, shifts, combined_weights)
print(f"Time taken for contour_matvec: {time.time() - time_start} seconds")
fv_combined = contour_matvec(c_H, v_H, D, v, tol, shifts, combined_weights)

time_start = time.time()
# fv_square_root = contour_matvec(c_H, v_H, D, v, 1e-5, shifts, weights)
fv_combined = contour_matvec(c_H, v_H, D, v, tol, shifts, combined_weights)

time_end = time.time()
print(f"Time taken: {time_end - time_start} seconds")

# plt.figure(figsize=(6,5), dpi=300)
# # v_H = v_H.sort()
# plt.plot(v_H)
# plt.grid(True)
# plt.savefig(f"v_H_{Ls[0]}.png")
# plt.close()


# weights1 = weights * complex_square_root_fermi_dirac(beta*shifts)
# weights2 = weights * complex_bin_entropy_fermi_dirac(beta*shifts)
# combined_weights = jnp.stack([weights1, weights2], axis=1) # [N_poles, 2]

# fv_combined = contour_matvec(c_H, v_H, D, v, 1e-10, shifts, combined_weights)
# fv_square_root = contour_matvec(c_H, v_H, D, v, 1e-10, shifts, weights1)
# fv_bin_entropy = contour_matvec(c_H, v_H, D, v, 1e-10, shifts, weights2)

# print(fv_combined.shape, fv_square_root.shape, fv_bin_entropy.shape)
# print(jnp.linalg.norm(fv_combined[:,:,0] - fv_square_root))
# print(jnp.linalg.norm(fv_combined[:,:,1] - fv_bin_entropy))









