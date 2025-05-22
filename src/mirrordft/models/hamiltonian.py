import jax
from jax import numpy as jnp
import numpy as np
import time
from .contour import *

jax.config.update("jax_enable_x64", True)

def genDiscretizedLaplacianEigenvalues(Ns, Ls):
    """
    Generate the discretized Laplacian in Fourier space.
    Args:
      Ns: (np.array [3]) Number of grid points in each dimension.
      Ls: (np.array [3]) Lengths of the domain in each dimension.

    Returns:
      D: (jnp.array [Ns]) Discretized Laplacian.
    """
    Ls = np.array(Ls)
    Ns = np.array(Ns)
    dim = len(Ns)
    dxs = Ls/Ns
    
    ks = [np.fft.fftfreq(Ns[i], d=dxs[i]) for i in range(dim)]
    Ks = np.meshgrid(*ks, indexing='ij')
    D = jnp.array(sum([-(2*np.pi*Ks[i])**2 for i in range(dim)]))
    return D

def genDiscretizedCosineExternalPotential(Ns, Ls):
    """
    Generate the discretized default cosine external potential.
    Args:
      Ns: (np.array [3]) Number of grid points in each dimension.
      Ls: (np.array [3]) Lengths of the domain in each dimension.

    Returns:
      D: (jnp.array [Ns[0]*Ns[1]*Ns[2]]) Discretized cosine external potential.
    """
    Ls = np.array(Ls)
    Ns = np.array(Ns)
    dim = len(Ns)
    ks = [np.linspace(0, Ns[i]-1, Ns[i])*Ls[i]/Ns[i] for i in range(dim)]
    Ks = np.meshgrid(*ks, indexing='ij')
    D = jnp.array(sum([jnp.cos(Ks[i]*2*np.pi/Ls[i]) for i in range(dim)]))
    D = D.flatten()
    return D

def dense_logm(P):
    w, V = jnp.linalg.eigh(P)
    return V @ jnp.diag(jnp.log(w)) @ V.T

def dense_tr_xlogx_m(P):
    w, _ = jnp.linalg.eigh(P)
    w = np.clip(w, 1e-20, 1e20)
    return jnp.sum(w * jnp.log(w))

def BinEntropy(P):
    return jnp.real(dense_tr_xlogx_m(P) + dense_tr_xlogx_m(jnp.eye(P.shape[0])-P))

def fftnProduct(D, v):
    """
    Multiply a vector by a diagonal matrix in Fourier space.
    
    This function performs the following steps:
    1. Reshapes the input vector into a multi-dimensional array
    2. Applies FFT to transform to Fourier space 
    3. Multiplies by the diagonal matrix
    4. Applies inverse FFT to transform back to real space
    5. Reshapes the result back to a vector

    Args:
        D: (jnp.array [Ns]) Diagonal matrix in Fourier space, where Ns is a tuple of 
           dimensions (e.g., (N1, N2, N3) for 3D)
        v: (jnp.array [N, n]) Input vector, where N = prod(Ns) and n is the number
           of columns (optional)

    Returns:
        (jnp.array [N, n]) or (jnp.array [N]) The transformed vector, flattened if n=1
    """
    # Get dimensions from the diagonal operator D
    Ns = D.shape  # Shape of spatial dimensions (e.g. N1,N2,N3 for 3D)
    dim = len(Ns) # Number of spatial dimensions
    
    # Reshape input vector v from [N,n] to [N1,N2,N3,n] for FFT
    # N = prod(Ns) is total spatial points, n is number of channels
    v = v.reshape(Ns + (-1,))  # Changed from list to tuple
    
    # Transform to Fourier space along spatial dimensions
    # Result has shape [N1,N2,N3,n] with complex values
    v = jnp.fft.fftn(v, axes=list(range(dim)))
    
    # Multiply by diagonal operator D in Fourier space
    # D[...,None] broadcasts D to match v's shape [N1,N2,N3,n]
    v = D[(...,) + (None,)] * v
    
    # Transform back to real space and reshape to original dimensions
    # First IFFT gives [N1,N2,N3,n], then reshape to [N,n]
    v = jnp.fft.ifftn(v, axes=list(range(dim))).reshape([-1, v.shape[-1]])
    
    # For single-channel case (n=1), return flattened 1D array
    if v.shape[-1] == 1:
        return v.flatten()
    
    return v

    """
    Constructs a unitary Fourier transform matrix of size N x N.
    The (j,k) entry is (1/sqrt(N))*exp(-2*pi*i*j*k/N).
    """
    n = jnp.arange(N)
    k = n.reshape((N, 1))
    F = jnp.exp(-2j * jnp.pi * k * n / N) / jnp.sqrt(N)
    return F

def genDiscretizedLaplacian(Ns, Ls):
    """
    Constructs the discretized Laplacian operator in real space using FFT.
    This is a highly efficient implementation that uses JAX's vmap for vectorization
    and avoids constructing the full Fourier matrix.
    
    Instead of explicitly constructing F and F*, we use FFT operations which are O(N log N)
    rather than the O(N²) complexity of the dense matrix approach.
    
    The operator in Fourier space is diagonal with eigenvalues:
       lambda(k1,k2,k3) = -[(2π*k1/L1)² + (2π*k2/L2)² + (2π*k3/L3)²]
    
    Args:
        Ns: (np.array [3]) Number of grid points in each dimension.
        Ls: (np.array [3]) Lengths of the domain in each dimension.
    
    Returns:
        K: (jnp.array [N, N]) Discretized Laplacian in real space.
    """
    N = np.prod(Ns)  # Total number of grid points
    
    # Get the eigenvalues in Fourier space
    eigenvalues = genDiscretizedLaplacianEigenvalues(Ns, Ls)
    
    # Create identity matrix
    I = jnp.eye(N)
    
    # Vectorize the FFT operation over columns of the identity matrix
    K = jax.vmap(lambda x: fftnProduct(eigenvalues, x.reshape(-1, 1)))(I.T)
    
    return K.T

def genDiscretizedYukawaInteractionEigenvalues(Ns, Ls, alpha):
    """
    Generate the discretized Yukawa Interaction in Fourier space.
    Args:
        Ns: (np.array [3]) Number of grid points in each dimension.
        Ls: (np.array [3]) Lengths of the domain in each dimension.
        alpha: (float) Screening length.

    Returns:
        D: (jnp.array [Ns]) Discretized Laplacian.
    """
    Ls = np.array(Ls)   
    Ns = np.array(Ns)
    dim = len(Ns)
    dxs = Ls/Ns
    
    ks = [np.fft.fftfreq(Ns[i], d=dxs[i]) for i in range(dim)]
    Ks = np.meshgrid(*ks, indexing='ij')
    D = jnp.array(sum([(2*np.pi*Ks[i])**2 for i in range(dim)])) + alpha**2
    delta_V = np.prod(Ls)/np.prod(Ns)
    
    # Note that we divide by delta_V^2 because we are using the Fourier 
    # transform and the discretization of the volume together. Hence, one comes 
    # from the discretization of the volume and the other comes from the waveplane-type basis.
    D = 1/D /delta_V *(alpha**2)

    return D

def genDiscretizedYukawaInteraction(Ns, Ls, alpha):
    """
    Constructs the discretized Yukawa Interaction operator in real space using FFT.
    This is a highly efficient implementation that uses JAX's vmap for vectorization
    and avoids constructing the full Fourier matrix.
    
    Instead of explicitly constructing F and F*, we use FFT operations which are O(N log N)
    rather than the O(N²) complexity of the dense matrix approach.
    
    Args:
        Ns: (np.array [3]) Number of grid points in each dimension.
        Ls: (np.array [3]) Lengths of the domain in each dimension.
        alpha: (float) Screening length.
    
    Returns:
        K: (jnp.array [N, N]) Discretized Yukawa interaction operator.
    """
    N = np.prod(Ns)  # Total number of grid points
    
    # Get the eigenvalues in Fourier space
    eigenvalues = genDiscretizedYukawaInteractionEigenvalues(Ns, Ls, alpha)
    
    # Create identity matrix
    I = jnp.eye(N)
    
    # Vectorize the FFT operation over columns of the identity matrix
    K = jax.vmap(lambda x: fftnProduct(eigenvalues, x.reshape(-1, 1)))(I.T)
    
    return K.T

def gen_centers(num_particles, Ns, seed=224):
    np.random.seed(seed)
    if len(Ns) == 1:
        return np.random.uniform(0, Ns[0], num_particles)
    else:
        return np.random.uniform(0, np.array(Ns), (num_particles, len(Ns)))

def gen_rho_ext(num_particles, Ns):
    centers = gen_centers(num_particles, Ns)
    rho_ext = np.zeros(Ns)
    if len(Ns) == 1:
        for i in range(num_particles):
            rho_ext[int(centers[i])] += 1
    else:
        for i in range(num_particles):
            indices = tuple(int(coord) for coord in centers[i])  # Convert coordinates to integer indices
            rho_ext[indices] += 1
    return rho_ext.flatten()

class Hamiltonian:
    def __init__(self, Ns, Ls, beta=1,mu=0, alpha=0, fourier=True, dense=False):
        """
        Initialize the Hamiltonian object.
        
        Args:
            Ns: (np.array [3]) Number of grid points in each dimension.
            Ls: (np.array [3]) Lengths of the domain in each dimension.
            beta: (float) Inverse temperature.
            alpha: (float) Screening length.
        """

        self.Ns = np.array(Ns)
        self.Ls = np.array(Ls)
        self.mu = mu
        self.beta = beta
        self.alpha = alpha

        self.fourier = fourier 
        self.dense = dense
        
        # Generate the discretized Laplacian and Yukawa interaction
        self.fourier_laplacian = genDiscretizedLaplacianEigenvalues(Ns, Ls) if fourier else None # [Ns]
        self.fourier_yukawa = genDiscretizedYukawaInteractionEigenvalues(Ns, Ls, alpha) if fourier else None # [Ns]
        self.dense_laplacian = genDiscretizedLaplacian(Ns, Ls) if dense else None # [N_vec, N_vec]
        self.dense_yukawa = genDiscretizedYukawaInteraction(Ns, Ls, alpha) if dense else None # [N_vec, N_vec]
        
        # Generate the discretized cosine external potential
        self.potential_external = genDiscretizedCosineExternalPotential(Ns, Ls) # [N_vec]

    def update_single_electron_effective_potential(self):
        pass 

    def update_external(self, potential_external):
        self.potential_external = potential_external # [N_vec]
        self.update_single_electron_effective_potential()

    # We require rho to be a 1D array for the following 3 functions!! 
    def potential_yukawa(self, rho):
        return jnp.real(fftnProduct(self.fourier_yukawa, rho)) # [N_vec]

    def energy_yukawa(self, rho):
        return jnp.real(jnp.sum(rho * self.potential_yukawa(rho))/2)

    def energy_external(self, rho):
        return jnp.real(jnp.sum(rho * self.potential_external))

    def update_external_yukawa(self, ratio):
        self.ratio = ratio
        n_particles = np.ceil(np.prod(self.Ls) * ratio).astype(int)
        self.density_external = gen_rho_ext(n_particles, self.Ns)
        self.potential_external = -self.potential_yukawa(self.density_external)
        self.update_single_electron_effective_potential()
    
    def update_external_yukawa_centers(self, centers, masses):
        rho_ext = np.zeros(self.Ns)
        for i in range(len(centers)):
            rho_ext[tuple(int(coord) for coord in centers[i])] += masses[i]
        self.density_external = rho_ext.flatten()
        self.potential_external = -self.potential_yukawa(self.density_external)
        self.update_single_electron_effective_potential()


class deterministicHamiltonian(Hamiltonian):
    def __init__(self, Ns, Ls, beta=1,mu=0, alpha=0, fourier=True, dense=True):
        super().__init__(Ns, Ls, beta, mu, alpha, fourier, dense)
        assert dense, "Only dense Hamiltonian is supported for deterministic Hamiltonian"
        self.key = 0
        self.N_vec = np.prod(Ns)

    def update_single_electron_effective_potential(self):
        self.C = - self.dense_laplacian/2 + jnp.diag(self.potential_external)


    # @partial(jit, static_argnums=(0,))
    def energy_kinetic(self, P):
        return jnp.real(jnp.trace( - P @ self.dense_laplacian) /2)
    
    # @partial(jit, static_argnums=(0,))
    def objective(self, H=None, mu=1, P=None):
        if H is not None:
            P = self.density_matrix(H)
        rho = jnp.real(jnp.diag(P))
        energy_external = self.energy_external(rho)
        energy_kinetic = self.energy_kinetic(P)
        energy_yukawa = self.energy_yukawa(rho)
        entropy = 1/self.beta * BinEntropy(P)

        energy_free = energy_kinetic + energy_external + energy_yukawa + entropy - mu * jnp.sum(rho)
        # return objective, energy_free 
        res = {
            "energy_free": energy_free,
            "energy_kinetic": energy_kinetic,
            "energy_external": energy_external,
            "energy_yukawa": energy_yukawa,
            "entropy": entropy,
            "sum_rho": sum(rho)
        }
        return res
    
    # @partial(jit, static_argnums=(0,))
    def density_matrix(self, H):
        w, V = jnp.linalg.eigh(H)
        w = jnp.clip(w, -500, 500)
        w1 = 1/(1+jnp.exp(self.beta*w))
        return jnp.real(V @ jnp.diag(w1) @ V.T) # [N_vec, N_vec]
    
    def density_function(self, H):
        P = self.density_matrix(H)
        return jnp.diag(P) # [N_vec]
    
    def gradient(self, H, cheat=False, N_samples=100):
        if cheat:
            w, V = jnp.linalg.eigh(H)
            w = jnp.clip(w, -500, 500)
            w1 = jnp.sqrt(1/(1+jnp.exp(self.beta*w)))
            P_half = V @ jnp.diag(w1) @ V.T
            v = jnp.complex128(jax.random.normal(jax.random.PRNGKey(self.key), (self.N_vec, N_samples))) # [N_vec, N_samples]
            fv = P_half @ v 
            rho = jnp.diag(fv.dot(fv.T))/N_samples 
            grad = self.C+ jnp.diag(self.potential_yukawa(rho))
            self.key += 1
        else:
            P = self.density_matrix(H)
            rho = jnp.diag(P)
            grad = self.C+ jnp.diag(self.potential_yukawa(rho))
        rho = jnp.real(rho)
        return grad,rho # [N_vec, N_vec]
    

class StochasticHamiltonian(Hamiltonian):
    def __init__(self, Ns, Ls, beta=1,mu=0, N_poles=100, alpha=0, fourier=True, dense=False):
        super().__init__(Ns, Ls, beta, mu, alpha, fourier, dense)
        assert fourier, "Only Fourier Hamiltonian is used for stochastic Hamiltonian"
        self.shifts = None 
        self.weights = None 
        self.key = 0
        self.N_poles = N_poles
        self.N_vec = np.prod(Ns)

        # generate initial shifts and weights
        self.update_poles_shifts(-1/2, 0)
    
    def update_poles_shifts(self, c_H, v_H):
        energy_min = np.max(self.fourier_laplacian)*c_H + np.min(v_H) # The minimum eigenvalue of the Hamiltonian
        energy_max = np.min(self.fourier_laplacian)*c_H + np.max(v_H) # The maximum eigenvalue of the Hamiltonian
        self.shifts, self.weights = gen_contour(energy_min, energy_max, self.beta, self.N_poles)
        square_root_fermi_dirac_weights = self.weights * complex_square_root_fermi_dirac(self.beta*self.shifts) # [N_poles]
        bin_entropy_fermi_dirac_weights = self.weights * complex_bin_entropy_fermi_dirac(self.beta*self.shifts) # [N_poles]
        combined_weights = jnp.stack([square_root_fermi_dirac_weights, bin_entropy_fermi_dirac_weights], axis=1) # [N_poles, 2]
        self.combined_weights = combined_weights

    # @partial(jit, static_argnums=(0,3))
    def density_function(self, c_H, v_H, N_samples=100, tol=1e-6):
        # Generate Gaussian random vectors
        v = jnp.complex128(jax.random.normal(jax.random.PRNGKey(self.key), (self.N_vec, N_samples))) # [N_vec, N_samples]
        
        # Compute the matrix-vector product
        fermi_dirac_weights = self.weights * complex_square_root_fermi_dirac(self.beta*self.shifts) # [N_poles]
        fv = contour_matvec(c_H, v_H, self.fourier_laplacian, v, tol, self.shifts, fermi_dirac_weights) # [N_vec, N_samples]
        
        # Update the key for the next random number generation
        self.key += 1 
        
        # Compute the estimate of the density function
        estimate_density_function = (jnp.diag(fv.dot(fv.T))/N_samples).flatten().real # [N_vec]

        return estimate_density_function,v


    def gradient(self, c_H, v_H, N_samples=100, tol=1e-6):

        estimate_density_function,v = self.density_function(c_H, v_H, N_samples, tol) # [N_vec]

        # Compute the gradient
        # Recall that we save H = c_H * K + diag^*(v_H), where c_H is a scalar and v_H is a vector.
        # We will decompose the gradient w.r.t c_H and v_H separately.

        gradient_c_H = -1/2 
        gradient_v_H = self.potential_yukawa(estimate_density_function) + self.potential_external # [N_vec]

        return gradient_c_H, gradient_v_H.real, estimate_density_function # [1], [N_vec], [N_vec]

    def objective(self, c_H, v_H, N_samples=100, tol=1e-5, mu=0):
        # Generate Gaussian random vectors
        v = jnp.complex128(jax.random.normal(jax.random.PRNGKey(self.key), (self.N_vec, N_samples)))
        self.key += 1
        
        combined_H_v = contour_matvec(c_H, v_H, self.fourier_laplacian, v, tol, self.shifts, self.combined_weights) # [N_vec, N_samples, 2]


        square_root_fermi_dirac_H_v = combined_H_v[:,:,0] # [N_vec, N_samples]
        bin_entropy_fermi_dirac_H_v = combined_H_v[:,:,1] # [N_vec, N_samples]
       
        estimate_density_function = (jnp.sum((square_root_fermi_dirac_H_v)**2, axis=1)/N_samples).real # [N_vec]

        grad_CH = - 1/2
        grad_vH = self.potential_yukawa(estimate_density_function) + self.potential_external # [N_vec]

        # Estimate the energy
        # There are three parts: sinple electron term, Hartree term, and entropy term
        # single electron term is tr(C*P) = tr(-1/2 K P) + v_{ext} \otimes rho
        
        # kinetic energy 
        energy_kinetic = -1/2 * jnp.sum(square_root_fermi_dirac_H_v * fftnProduct(self.fourier_laplacian, square_root_fermi_dirac_H_v)).real/N_samples # scalar

        # External potential energy
        energy_external = self.energy_external(estimate_density_function) # scalar

        # Hartree energy 
        energy_yukawa = self.energy_yukawa(estimate_density_function) # scalar

        # Entropy energy
        energy_entropy = 1/self.beta * jnp.sum(bin_entropy_fermi_dirac_H_v * v).real/N_samples # scalar

        # Total energy 
        energy_free = energy_kinetic + energy_external + energy_yukawa + energy_entropy  - mu * jnp.sum(estimate_density_function) # scalar

        res = {
            "energy_free": jnp.real(energy_free),
            "energy_kinetic": jnp.real(energy_kinetic),
            "energy_external": jnp.real(energy_external),
            "energy_yukawa": jnp.real(energy_yukawa),
            "entropy": jnp.real(energy_entropy),
            "sum_rho": jnp.sum(estimate_density_function),
            "grad_CH": grad_CH,
            "grad_vH": grad_vH,
            "density": estimate_density_function,
        }

        return res

if __name__ == "__main__":
    # Test the potential
    Ns = (11,11,11)
    Ls = (1,1,1)
    laplacian_fourier = genDiscretizedLaplacianEigenvalues(Ns, Ls)
    laplacian_dense = genDiscretizedLaplacian(Ns, Ls)
    yukawa_fourier = genDiscretizedYukawaInteractionEigenvalues(Ns, Ls, alpha=1)
    yukawa_dense = genDiscretizedYukawaInteraction(Ns, Ls, alpha=1)
    key = jax.random.PRNGKey(0)  # Add a PRNG key for reproducibility
    v = jax.random.uniform(key, shape=(Ns[0]*Ns[1]*Ns[2],))
    lap_fourier_v = fftnProduct(laplacian_fourier, v)
    lap_dense_v = laplacian_dense.dot(v)
    print(jnp.linalg.norm(lap_fourier_v - lap_dense_v))
    yukawa_fourier_v = fftnProduct(yukawa_fourier, v)
    yukawa_dense_v = yukawa_dense.dot(v)
    print(jnp.linalg.norm(yukawa_fourier_v - yukawa_dense_v))

    # Genereate two Hamiltonians
    Ns = (1281,)
    Ls = (10,)
    det_H = deterministicHamiltonian(Ns, Ls, beta=2, alpha=1)
    sto_H = StochasticHamiltonian(Ns, Ls, beta=2, alpha=1)
    sto_H.key = 24

    # Generate a random H
    c_H = 0
    # v_H = np.random.rand(Ns[0]*Ns[1]*Ns[2])
    # v_H = np.cos(10*np.pi*np.arange(Ns[0]*Ns[1]*Ns[2])/Ns[0]/Ns[1]/Ns[2])*10
    v_H = jax.random.normal(key, shape=(np.prod(Ns),))
    # v_H[100] = 10
    v_H = jnp.array(v_H)
    H = c_H * det_H.dense_laplacian + jnp.diag(v_H)
    sto_H.update_poles_shifts(c_H, v_H)

    # # Estimate density 
    rho_true = det_H.density_function(H)
    rho_esti = sto_H.density_function(c_H, v_H, N_samples=100)
    print(rho_true[:10])
    print(rho_esti[:10])

    # Estimate energy 
    print(det_H.objective(H))
    print(sto_H.objective(c_H, v_H, N_samples=20))

    # update external potential
    # det_H.update_external_yukawa(np.array([[0.2, 0.2, 0.2]]), np.array([10]))
    # print(det_H.potential_external[:10])
