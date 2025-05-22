import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.contour import * 
from src.models.hamiltonian import * 
from tqdm import tqdm

# Hyperparameters 
Nss = [101, 31, 11]
Lss = [100,30,10]
N_samples = 10
alpha = 0.5 
betas = 2**np.linspace(0,8,9)
poles = np.linspace(0, 60,61).astype(int)[1:]
errors = np.zeros((len(Nss), len(betas), len(poles)))

# Generate data 
for dim in range(3):

    Ns = [Nss[dim] for _ in range(dim+1)] 
    Ls = [Lss[dim] for _ in range(dim+1)]
    # Ns = (101,)
    # Ls = (100,)
    n_particles = np.prod(Ls).astype(int)
    # print(Ns, n_particles)
    external_density = gen_rho_ext(n_particles, Ns)
    Y = genDiscretizedYukawaInteractionEigenvalues(Ns, Ls, alpha)
    D = genDiscretizedLaplacianEigenvalues(Ns, Ls)
    c_H = -1/2
    v_H = -fftnProduct(Y, external_density)

    K = genDiscretizedLaplacian(Ns, Ls) 
    H = c_H*K + jnp.diag(v_H)  

    E_m = c_H * (jnp.max(D)) + jnp.min(v_H) 
    E_M = c_H * (jnp.min(D)) + jnp.max(v_H)
    print("Min and max eigenvalues of H: ", E_m, E_M)

    N_vec = np.prod(Ns)
    v = jnp.complex128(jax.random.normal(jax.random.PRNGKey(0), (N_vec, N_samples)))
    w, V = jnp.linalg.eigh(H) 

    for i in range(len(betas)):

        beta = betas[i]
        x = w*beta  
        # x = jnp.clip(x, -400, 400) 
        w1 = complex_square_root_fermi_dirac(x)
        w1 = jnp.real(w1)
        square_f_beta_H = V @ jnp.diag(w1) @ V.T
        true_result = square_f_beta_H @ v
        true_result = jnp.real(true_result)


        for j in tqdm(range(len(poles))):

            n_poles = poles[j]
            shifts, weights = gen_contour(E_m, E_M, beta, n_poles, function='fermi_dirac', clip=False)
            result = contour_matvec(c_H, v_H, D, v, 1e-12, shifts, weights)
            error = jnp.linalg.norm(result - true_result, ord=1)/jnp.linalg.norm(true_result, ord=1)
            # print(f"Error: {error}")
            errors[dim,i,j] = error.item()    

# Save data 
data_dir = "./figures/fig_contour_error"
data = {
    "errors": errors,
    "betas": betas,
    "poles": poles,
    "Nss": Nss,
    "Lss": Lss,
    "alpha": alpha,
    "N_samples": N_samples
    }
import pickle 
with open(os.path.join(data_dir, "data.pkl"), "wb") as f:
    pickle.dump(data, f)

# Print data 
# Print data in tabular format
from tabulate import tabulate

for dim in range(len(Nss)):
    print(f"\n{dim+1}D Contour Integration Errors:")
    
    # Create a table with beta, N_poles, error format
    headers = ["Beta", "N_poles", "Error"]
    table = []
    
    for i in range(len(betas)):
        for j in range(len(poles)):
            row = [betas[i], poles[j], errors[dim, i, j]]
            table.append(row)
    
    print(tabulate(table, headers=headers, floatfmt=".6e"))
    
    # Save table to txt file
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f'./figures/fig_contour_error/contour_errors_{dim+1}D.txt', 'w') as f:
        f.write(f'{dim+1}D Contour Integration Errors\n')
        f.write(f'Generated on: {timestamp}\n')
        f.write(tabulate(table, headers=headers, floatfmt=".6e"))

