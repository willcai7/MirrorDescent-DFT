import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
import sys
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from collections import defaultdict
import numpy as np
from benchmarks.benchmark_tool import *
import pickle
from tabulate import tabulate

Nss = [(101,),(11,11), (5, 5, 5)] # 1D, 2D, 3D
Lss = [(10,), (10,10), (10, 10, 10)] # 1D, 2D, 3D


betas = [1,2,4,8,16,32,64, 128]
n_poless = np.linspace(1,60,61).astype(int)[1:]
errors = defaultdict(list)


for dim in range(len(Nss)):
    print(f'{dim+1}D scaling of beta')
    Ns = Nss[dim]
    Ls = Lss[dim]
    N_vec = np.prod(Ns)

    c_H = -1
    v_H = jnp.ones(N_vec)
    D = genDiscretizedLaplacianEigenvalues(Ns, Ls)
    K = genDiscretizedLaplacian(Ns, Ls)
    H = c_H * K + jnp.diag(v_H) 
    v = jnp.complex128(jax.random.normal(jax.random.PRNGKey(0), (N_vec, 1)))
    w, V = jnp.linalg.eigh(H) 

    E_m = c_H * (jnp.max(D))- 1
    E_M = c_H * (jnp.min(D))+ 1

    for i in range(len(betas)):
        beta = betas[i]
        x = w*beta  
        w1 = complex_square_root_fermi_dirac(x)
        w1 = jnp.real(w1)
        square_f_beta_H = V @ jnp.diag(w1) @ V.T
        true_result = square_f_beta_H @ v
        true_result = jnp.real(true_result)
        errors[dim].append([])


        for j in tqdm(range(len(n_poless))):

            n_poles = n_poless[j]
            shifts, weights = gen_contour(E_m, E_M, beta, n_poles, function='fermi_dirac')
            result = contour_matvec(c_H, v_H, D, v, 1e-12, shifts, weights)
            error = jnp.mean(jnp.abs(result - true_result))
            errors[dim][i].append(error.item())


# After collecting all errors, print them in a tabulated format
for dim in range(len(Nss)):
    print(f"\n{dim+1}D Error Analysis for Different Beta Values")
    
    for i, beta in enumerate(betas):
        print(f"\nBeta = {beta}")
        headers = ["N_poles", "Error"]
        table = [[n_poless[j], errors[dim][i][j]] for j in range(len(n_poless))]
        print(tabulate(table, headers=headers, floatfmt=".8e"))

# Save the error data
data = {'errors': errors, 'betas': betas, 'n_poless': n_poless}
with open('./error_vs_beta.pkl', 'wb') as f:
    pickle.dump(data, f)


