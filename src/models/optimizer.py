import jax.numpy as jnp
from tqdm import tqdm
import numpy as np
import jax

def binary_search_mu(eigvals, N_electrons, beta, tol=1e-3, max_iter=1000):
    """Find chemical potential mu using binary search to match target electron number.
    
    Args:
        eigvals: Array of eigenvalues
        N_electrons: Target number of electrons
        beta: Inverse temperature
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
        
    Returns:
        mu: Chemical potential that gives target electron number
    """
    mu_min = -100/beta + eigvals.min()
    mu_max = 100/beta + eigvals.max()
    
    for _ in range(max_iter):
        mu_mid = (mu_min + mu_max)/2
        n_electrons = jnp.sum(1/(1 + jnp.exp(beta*(eigvals-mu_mid))))
        
        if jnp.abs(n_electrons - N_electrons) < tol:
            return mu_mid
            
        if n_electrons > N_electrons:
            mu_max = mu_mid
        else:
            mu_min = mu_mid
            
    raise RuntimeError(f"Binary search did not converge after {max_iter} iterations")

def run_scf(ham, N_electrons, max_iter=100, tol=1e-8):
    """
    Run self-consistent field iterations for the given Hamiltonian.
    
    Args:
        ham: Hamiltonian object
        N_electrons: Number of electrons in the system
        max_iter: Maximum number of SCF iterations (default: 100)
        tol: Convergence tolerance (default: 1e-8)
    
    Returns:
        dict: Results containing final X_scf, convergence error, and other relevant quantities
    """
    beta = ham.beta
    N_vec = np.prod(ham.Ns)
    C = ham.C
    X_scf = jnp.eye(N_vec)/2
    prev_X_scf = X_scf

    pbar = tqdm(range(max_iter), 
                desc="SCF", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')

    for i in pbar:
        prev_X_scf = X_scf
        new_ham = C + jnp.diag(ham.potential_yukawa(jnp.diag(X_scf)))
        new_ham = (new_ham + new_ham.T)/2

        eigvals, eigvecs = jnp.linalg.eigh(new_ham)
        eigvals = jnp.clip(eigvals, -400/beta, 400/beta)   
        mu = binary_search_mu(eigvals, N_electrons, beta)
        eigvals = 1/(1+ jnp.exp(beta*(eigvals-mu)))
        X_scf = eigvecs @ jnp.diag(eigvals) @ eigvecs.T
        
        error = jnp.linalg.norm(jnp.diag(X_scf) - jnp.diag(prev_X_scf))
        pbar.set_postfix({"error": f"{error:.3e}"})
        
        if error < tol:
            print(f"\nConverged at iteration {i} with error {error:.3e}")
            break
    
    # Calculate final quantities
    density_scf = jnp.diag(X_scf).real
    res = ham.objective(P=X_scf, mu=mu) 
    res["mu"] = mu
    res["density"] = density_scf
    res["density_matrix"] = X_scf
    res["iterations"] = i+1
    res["error"] = error

    return res

class GoldAlgo:
    def __init__(self, ham, rho_scf, mu, n_samples): 
        self.key = 0
        self.ham = ham 
        self.n_samples = n_samples
        H = ham.C + jnp.diag(ham.potential_yukawa(rho_scf)) 
        eigvals, eigvecs = jnp.linalg.eigh(H)
        eigvals = jnp.clip(eigvals, -400/ham.beta, 400/ham.beta)
        eigvals = jnp.sqrt(1/(1+jnp.exp(ham.beta*(eigvals-mu))))
        self.half_P = eigvecs @ jnp.diag(eigvals) @ eigvecs.T
        self.average_rho = jnp.zeros_like(rho_scf)
        self.total_samples = 0

    def step(self):
        v = jax.random.normal(jax.random.PRNGKey(self.key), (self.ham.N_vec, self.n_samples))
        fv = self.half_P @ v
        self.average_rho += jnp.diag(fv.dot(fv.T))
        self.total_samples += self.n_samples
        self.average_rho_temp = self.average_rho / self.total_samples
        self.key += 1
        return self.average_rho_temp.real