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

def benchmark_linear_system(Ns, Ls, N_samples, tol=1e-6, test_size=30):

    # Initialize the Hamiltonian
    c_H = -1
    N_vec = np.prod(Ns)    
    
    # Move constants to device once
    v_H = jax.device_put(jax.random.normal(jax.random.PRNGKey(0), (N_vec, 1)).flatten())
    D = jax.device_put(genDiscretizedLaplacianEigenvalues(Ns, Ls))
    shift = jax.device_put(jnp.complex128(1j+2))

    # Compile function once before timing
    @jax.jit
    def run_shift_inv(v):
        return shift_inv_system(c_H, v_H, D, v, shift, tol)
    
    # Warmup run
    v_warmup = jnp.complex128(jax.random.normal(jax.random.PRNGKey(99), (N_vec, N_samples)))
    _ = run_shift_inv(v_warmup)
    
    times = []
    errs = []
    for i in range(test_size):
        # Generate and move data to device before timing
        v = jax.device_put(jnp.complex128(jax.random.normal(jax.random.PRNGKey(i), (N_vec, N_samples))))
        
        # Ensure GPU is synchronized before timing
        jax.block_until_ready(v)
        
        start = time.perf_counter()  # More precise than time.time()
        u = run_shift_inv(v)
        # Ensure computation is complete before stopping timer
        jax.block_until_ready(u)
        end = time.perf_counter()
        times.append(end - start)

        # Error computation
        temp_v = v_H[:, None] * u
        u = fftnProduct(shift - c_H *D, u)
        u = u - temp_v
        errs.append(float(jnp.mean(jnp.abs(u - v))))
    
    # Remove warmup period
    times = np.array(times[5:])  # Skip first 5 iterations
    errs = np.array(errs[5:])

    return (times.mean(), times.std()), (errs.mean(), errs.std())

def genNpoles(betas,tol=1e-6):
    M = np.log10(tol)
    betas = np.array(betas)
    return np.maximum(10, (-M+0.5)*(0.5 +0.3*np.log2(betas))+2).astype(int)

def benchmark_contour(Ns, Ls, N_poles, N_samples, beta, test_size=30, verbose=False, tol=1e-6, key=0,gen_N_poles=False, check_error=False,ratio=1,alpha=0.5):
    """
    Benchmark contour_matvec with improved timing stability.
    Returns: (mean_time, std_time)
    """
    # Initialize constants and move to device once
    N_vec = np.prod(Ns)
    D = jax.device_put(genDiscretizedLaplacianEigenvalues(Ns, Ls))
    c_H = jax.device_put(-1/2)
    # v_H = jax.device_put(jax.random.normal(jax.random.PRNGKey(key), (N_vec, 1)).flatten())
    n_particles = np.ceil(np.prod(Ls) * ratio).astype(int)
    external_density = gen_rho_ext(n_particles, Ns)
    Y = genDiscretizedYukawaInteractionEigenvalues(Ns, Ls, alpha)
    v_H = -fftnProduct(Y, external_density)
    v_H = jnp.real(v_H)

    # Compute energy bounds and contour parameters
    energy_min = jnp.max(D)*c_H + jnp.min(v_H)
    energy_max = jnp.min(D)*c_H + jnp.max(v_H)
    if verbose:
        print(f"energy_min: {energy_min}, energy_max: {energy_max}")
    if gen_N_poles:
        # print(max(abs(energy_min), abs(energy_max)))
        N_poles = genNpoles(beta * max(abs(energy_min), abs(energy_max)), tol=tol)
        # print(f"N_poles: {N_poles}")

    shifts, weights = gen_contour(energy_min, energy_max, beta, N_poles,function="fermi_dirac", clip=False)
    shifts = jax.device_put(shifts)
    weights = jax.device_put(weights)

    # Compile function once before timing
    @jax.jit
    def run_contour(v):
        return contour_matvec(c_H, v_H, D, v, tol, shifts, weights)
    
    # Warmup run
    v_warmup = jnp.complex128(jax.random.normal(jax.random.PRNGKey(99), (N_vec, N_samples)))
    _ = run_contour(v_warmup)
    jax.block_until_ready(_)

    if check_error:
        Mat = c_H*genDiscretizedLaplacian(Ns, Ls) + jnp.diag(v_H) 
        w, V = jnp.linalg.eigh(Mat)
        w = jnp.real(w)
        # w1 = jnp.sqrt(1/(1+jnp.exp(jnp.clip(beta*jnp.real(w),-700,700))))
        w1 = complex_square_root_fermi_dirac(jnp.clip(beta*jnp.real(w),-600,600))
        P = V@jnp.diag(w1)@V.T 
        P = (P + P.T)/2 
        errors = []

    times = []
    for i in range(test_size):
        v = jax.device_put(jnp.complex128(jax.random.normal(jax.random.PRNGKey(i), (N_vec, N_samples))))
        jax.block_until_ready(v)
        
        start = time.perf_counter()
        u = run_contour(v)
        jax.block_until_ready(u)
        end = time.perf_counter()
        
        times.append(end - start)
        if check_error:
            u_true = P@v
            err = jnp.mean(jnp.abs(u - u_true))
            errors.append(err.item())

    if verbose:
        if check_error:
            return times, errors
        else:
            return times
    # Remove warmup period and compute statistics
    times = np.array(times[5:])  # Skip first 5 iterations
    errors = np.array(errors[5:])
    if check_error:
        return times.mean(), times.std(), errors.mean(), errors.std()
    else:
        return times.mean(), times.std()

if __name__ == "__main__":
    base_dim = 3
    Nss = [(base_dim**6,), 
            (base_dim**3,base_dim**3), 
            (base_dim**2, base_dim**2, base_dim**2)] # 1D, 2D, 3D
    Lss = [(base_dim**6-1,), 
            ((base_dim**3-1)*np.sqrt(2), (base_dim**3-1)*np.sqrt(2)), 
            ((base_dim**2-1)*np.sqrt(3), (base_dim**2-1)*np.sqrt(3), (base_dim**2-1)*np.sqrt(3))] # 1D, 2D, 3D

    test_size = 20 
    beta = 40
    N_poles = 20
    N_samples = 20

    for dim in range(len(Nss)):
        print(f'Scaling of {dim+1}D contour solver')
        Ns = Nss[dim]
        # Ls = tuple(item/5 for item in Lss[dim])
        Ls = Lss[dim]
        print(Ls)
        elapsed_time, errors = benchmark_contour(Ns, Ls,N_poles, N_samples, beta, test_size=test_size, check_error=True, verbose=True,tol=1e-4)
        # elapsed_time = benchmark_contour(Ns, Ls,N_poles, N_samples, beta, test_size=test_size, gen_N_poles=True, verbose=True,tol=1e-5) 

        print(elapsed_time) 
        print(errors)

    