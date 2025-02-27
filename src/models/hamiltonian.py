import jax
from jax import numpy as jnp
import numpy as np
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

    dxs = Ls/Ns
    
    ks = [np.fft.fftfreq(Ns[i], d=dxs[i]) for i in range(3)]
    Ks = np.meshgrid(*ks, indexing='ij')
    D = jnp.array(sum([-(2*np.pi*Ks[i])**2 for i in range(3)]))
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

    ks = [np.linspace(0, Ns[i]-1, Ns[i])*Ls[i]/Ns[i] for i in range(3)]
    Ks = np.meshgrid(*ks, indexing='ij')
    D = jnp.array(sum([jnp.cos(Ks[i]*2*np.pi/Ls[i]) for i in range(3)]))
    D = D.reshape([Ns[0]*Ns[1]*Ns[2], 1])
    return D

def dense_logm(P):
    w, V = jnp.linalg.eigh(P)
    return V @ jnp.diag(jnp.log(w)) @ V.T

def dense_tr_xlogx_m(P):
    w, _ = jnp.linalg.eigh(P)
    w = np.clip(w, 1e-20, 1e20)
    return jnp.sum(w * jnp.log(w))

def BinEntropy(P):
    return dense_tr_xlogx_m(P) + dense_tr_xlogx_m(jnp.eye(P.shape[0])-P)

def fftnProduct(D, v):
    """
    Multiply a vector 'v' by a diagonal matrix 'D' in 3D Fourier space.

    Args:
        D: (jnp.array [Ns]) Diagonal matrix in 3D Fourier space.
        v: (jnp.array [Ns[0]*Ns[1]*Ns[1], n]) Vector to be multiplied.
    
    Returns:
        v: (jnp.array [Ns[0]*Ns[1]*Ns[1], n]) Transformed vector.

    """
    # Reshape the input vector 'v' into a multi-dimensional array with spatial dimensions Ns[0], Ns[1], Ns[2]
    # The '-1' infers the size of any additional dimensions (e.g., multiple channels).
    Ns = D.shape
    v = jnp.fft.fftn(v.reshape([Ns[0], Ns[1], Ns[2], -1]), axes=[0, 1, 2])
    
    # Multiply the Fourier-transformed 'v' element-wise by the diagonal multiplier 'D'.
    # 'D' is broadcasted along an additional axis (None adds a new axis) to match the shape of 'v'.
    v = D[:, :, :, None] * v
    
    # Compute the inverse FFT along the same spatial axes to transform the product back to real space.
    # The result is then reshaped to a 2D array with shape [Ns[0]*Ns[1]*Ns[2], -1],
    # where the spatial dimensions are flattened.
    v = jnp.fft.ifftn(v, axes=[0, 1, 2]).reshape([Ns[0]*Ns[1]*Ns[2], -1])
    
    # Return the final transformed vector.
    return v

def fourier_matrix_1d(N):
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
    Constructs the discretized Laplacian operator in real space by transforming
    the diagonal operator in the planewave (Fourier) basis.
    
    The total number of grid points is N = N1 * N2 * N3.
    The Fourier transform matrix for the 3D grid is given by:
       F = F1 ⊗ F2 ⊗ F3
    where F1, F2, and F3 are the 1D Fourier matrices for each dimension.
    
    The diagonal eigenvalue for a mode (k1, k2, k3) is:
       lambda = - [ (2*pi*k1/L1)^2 + (2*pi*k2/L2)^2 + (2*pi*k3/L3)^2 ].
    
    The real-space Laplacian matrix is then:
       L = F^* D F.
    
    Args:
        Ns: (np.array [3]) Number of grid points in each dimension.
        Ls: (np.array [3]) Lengths of the domain in each dimension.
    
    Returns:
        K: (jnp.array [N, N]) Discretized Laplacian.
    """
    # Construct 1D Fourier matrices
    F1 = fourier_matrix_1d(Ns[0])
    F2 = fourier_matrix_1d(Ns[1])
    F3 = fourier_matrix_1d(Ns[2])
    
    # Construct the 3D Fourier matrix as a Kronecker product
    F = jnp.kron(jnp.kron(F1, F2), F3)  # Shape: (N1*N2*N3, N1*N2*N3)
    

    # For a grid on [0, L_i] discretized with N_i points, one common way to get the 
    # integer wave numbers is to use fftfreq and multiply by L_i.
    k1 = jnp.fft.fftfreq(Ns[0], d=Ls[0]/Ns[0]) * Ls[0]
    k2 = jnp.fft.fftfreq(Ns[1], d=Ls[1]/Ns[1]) * Ls[1]
    k3 = jnp.fft.fftfreq(Ns[2], d=Ls[2]/Ns[2]) * Ls[2]

    # Create a 3D grid of wave numbers.
    K1, K2, K3 = jnp.meshgrid(k1, k2, k3, indexing='ij')
    # Flatten them so that each mode corresponds to an entry
    K1 = K1.flatten()
    K2 = K2.flatten()
    K3 = K3.flatten()
    
    # Compute the eigenvalues for each mode:
    eigenvalues = -(((2 * jnp.pi * K1 / Ls[0])**2) +
                    ((2 * jnp.pi * K2 / Ls[1])**2) +
                    ((2 * jnp.pi * K3 / Ls[2])**2))
    
    # Form the diagonal matrix D
    D = jnp.diag(eigenvalues)
    
    # Since F is unitary (F^{-1} = F^*), the Laplacian in real space is:
    K = jnp.conjugate(F.T) @ D @ F
    return jnp.real(K)

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

    dxs = Ls/Ns
    
    ks = [np.fft.fftfreq(Ns[i], d=dxs[i]) for i in range(3)]
    Ks = np.meshgrid(*ks, indexing='ij')
    D = jnp.array(sum([(2*np.pi*Ks[i])**2 for i in range(3)])) + alpha**2
    D = 1/D / (Ls[0]*Ls[1]*Ls[2])*Ns[0]*Ns[1]*Ns[2]
    D = D.at[0,0,0].set(0)
    return D

def genDiscretizedYukawaInteraction(Ns, Ls, alpha):
    """
    Constructs the discretized Yukawa Interaction operator in real space by transforming
    the diagonal operator in the planewave (Fourier) basis.
    
    Args:
        Ns: (np.array [3]) Number of grid points in each dimension.
        Ls: (np.array [3]) Lengths of the domain in each dimension.
        alpha: (float) Screening length.
    
    Returns:
        K: (jnp.array [N, N]) Discretized Laplacian.
    """
    # Construct 1D Fourier matrices
    F1 = fourier_matrix_1d(Ns[0])
    F2 = fourier_matrix_1d(Ns[1])
    F3 = fourier_matrix_1d(Ns[2])
    
    # Construct the 3D Fourier matrix as a Kronecker product
    F = jnp.kron(jnp.kron(F1, F2), F3)  # Shape: (N1*N2*N3, N1*N2*N3)
    

    # For a grid on [0, L_i] discretized with N_i points, one common way to get the 
    # integer wave numbers is to use fftfreq and multiply by L_i.
    k1 = jnp.fft.fftfreq(Ns[0], d=Ls[0]/Ns[0]) * Ls[0]
    k2 = jnp.fft.fftfreq(Ns[1], d=Ls[1]/Ns[1]) * Ls[1]
    k3 = jnp.fft.fftfreq(Ns[2], d=Ls[2]/Ns[2]) * Ls[2]

    # Create a 3D grid of wave numbers.
    K1, K2, K3 = jnp.meshgrid(k1, k2, k3, indexing='ij')
    # Flatten them so that each mode corresponds to an entry
    K1 = K1.flatten()
    K2 = K2.flatten()
    K3 = K3.flatten()
    
    # Compute the eigenvalues for each mode:
    eigenvalues = (((2 * jnp.pi * K1 / Ls[0])**2) +
                ((2 * jnp.pi * K2 / Ls[1])**2) +
                ((2 * jnp.pi * K3 / Ls[2])**2)) + alpha**2

    eigenvalues = 1/eigenvalues / (Ls[0]*Ls[1]*Ls[2])*Ns[0]*Ns[1]*Ns[2]
    eigenvalues = jnp.where((K1 == 0) & (K2 == 0) & (K3 == 0), 0, eigenvalues)
    
    # Form the diagonal matrix D
    D = jnp.diag(eigenvalues)
    
    # Since F is unitary (F^{-1} = F^*), the Laplacian in real space is:
    K = jnp.conjugate(F.T) @ D @ F
    return jnp.real(K)    

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

        self.Ns = Ns
        self.Ls = Ls 
        self.mu = mu
        self.beta = beta
        self.alpha = alpha

        self.fourier = fourier 
        self.dense = dense
        
        # Generate the discretized Laplacian and Yukawa interaction
        self.fourier_laplacian = genDiscretizedLaplacianEigenvalues(Ns, Ls) if fourier else None
        self.fourier_yukawa = genDiscretizedYukawaInteractionEigenvalues(Ns, Ls, alpha) if fourier else None
        self.dense_laplacian = genDiscretizedLaplacian(Ns, Ls) if dense else None
        self.dense_yukawa = genDiscretizedYukawaInteraction(Ns, Ls, alpha) if dense else None
        
        # Generate the discretized cosine external potential
        self.potential_external = genDiscretizedCosineExternalPotential(Ns, Ls)


    def update_external(self, potential_external):
        self.potential_external = potential_external

    @partial(jit, static_argnums=(0))
    def potential_yukawa(self, rho):
        return fftnProduct(self.fourier_yukawa, rho)

    @partial(jit, static_argnums=(0))
    def energy_yukawa(self, rho):
        return jnp.sum(rho.flatten() * self.potential_yukawa(rho).flatten())/2

    @partial(jit, static_argnums=(0))
    def energy_external(self, rho):
        return jnp.sum(rho.flatten() * self.potential_external.flatten())

class deterministicHamiltonian(Hamiltonian):
    def __init__(self, Ns, Ls, beta=1,mu=0, alpha=0, fourier=True, dense=True):
        super().__init__(Ns, Ls, beta, mu, alpha, fourier, dense)
        assert dense, "Only dense Hamiltonian is supported for deterministic Hamiltonian"
        self.C = - self.dense_laplacian/2 + jnp.diag(self.potential_external)

    # @partial(jit, static_argnums=(0,))
    def energy_kinetic(self, P):
        return jnp.trace( - P @ self.dense_laplacian) /2 
    
    # @partial(jit, static_argnums=(0,))
    def objective(self, H):
        P = self.density_matrix(H)
        rho = jnp.diag(P)
        energy_external = self.energy_external(rho)
        energy_kinetic = self.energy_kinetic(P)
        energy_yukawa = self.energy_yukawa(rho)
        entropy = 1/self.beta * BinEntropy(P)

        energy_free = energy_kinetic + energy_external + energy_yukawa + entropy
        objective = energy_free - self.mu * jnp.sum(rho)
        # return objective, energy_free
        return objective, energy_free, energy_kinetic, energy_external, energy_yukawa, entropy
    
    # @partial(jit, static_argnums=(0,))
    def density_matrix(self, H):
        w, V = jnp.linalg.eigh(H)
        w = jnp.clip(w, -500, 500)
        w1 = 1/(1+jnp.exp(self.beta*w))
        return V @ jnp.diag(w1) @ V.T
    
    def density_function(self, H):
        P = self.density_matrix(H)
        return jnp.diag(P)
    
    @partial(jit, static_argnums=(0,))
    def gradient(self, H):
        P = self.density_matrix(H)
        rho = jnp.diag(P)
        grad = self.C + jnp.diag(self.potential_yukawa(rho))
        return grad

class StochasticHamiltonian(Hamiltonian):
    def __init__(self, Ns, Ls, beta=1,mu=0, N_poles=100, alpha=0, fourier=True, dense=False):
        super().__init__(Ns, Ls, beta, mu, alpha, fourier, dense)
        assert fourier, "Only Fourier Hamiltonian is used for stochastic Hamiltonian"
        self.shifts = None 
        self.weights = None 
        self.key = 0
        self.N_poles = N_poles
        self.N_vec = Ns[0]*Ns[1]*Ns[2]
    
    def update_poles_shifts(self, c_H, v_H):
        energy_min = np.max(self.fourier_laplacian)*c_H + np.min(v_H) # The minimum eigenvalue of the Hamiltonian
        energy_max = -np.min(self.fourier_laplacian)*c_H + np.max(v_H) # The maximum eigenvalue of the Hamiltonian
        self.shifts, self.weights = gen_contour(energy_min, energy_max, self.beta, self.N_poles)

    # @partial(jit, static_argnums=(0,))
    def density_function(self, c_H, v_H, N_samples=100, tol=1e-6):
        # Generate Gaussian random vectors
        v = jnp.complex128(jax.random.normal(jax.random.PRNGKey(self.key), (self.N_vec, N_samples)))
        
        # Compute the matrix-vector product
        fermi_dirac_weights = self.weights * complex_square_root_fermi_dirac(self.beta*self.shifts)
        fv = contour_matvec(c_H, v_H,self.Ns, self.fourier_laplacian, v, tol, self.shifts, fermi_dirac_weights)
        
        # Update the key for the next random number generation
        self.key += 1 
        
        # Compute the estimate of the density function
        estimate_density_function = (jnp.diag(fv.dot(fv.T))/N_samples).reshape([self.N_vec, 1]).real 

        return estimate_density_function

    # @partial(jit, static_argnums=(0,))
    def gradient(self, c_H, v_H, N_samples=100, tol=1e-6):

        estimate_density_function = self.density_function(c_H, v_H, N_samples, tol)

        # Compute the gradient
        # Recall that we save H = c_H K + diag^*(v_H). 
        # We will decompose the gradient w.r.t c_H and v_H separately.
        gradient_c_H = -1/2 
        gradient_v_H = fftnProduct(self.fourier_yukawa, estimate_density_function) + self.potential_external - self.mu

        return gradient_c_H, gradient_v_H

    # @partial(jit, static_argnums=(0,1,3))
    def objective(self, c_H, v_H, N_samples=100, tol=1e-6):
        # Generate Gaussian random vectors
        v = jnp.complex128(jax.random.normal(jax.random.PRNGKey(self.key), (self.N_vec, N_samples)))

        # Compute the matrix-vector product
        square_root_fermi_dirac_weights = self.weights * complex_square_root_fermi_dirac(self.beta*self.shifts)
        square_root_fermi_dirac_H_v = contour_matvec(c_H, v_H, self.Ns, self.fourier_laplacian, v, tol, self.shifts, square_root_fermi_dirac_weights)

        bin_entropy_fermi_dirac_weights = self.weights * complex_bin_entropy_fermi_dirac(self.beta*self.shifts)
        bin_entropy_fermi_dirac_H_v = contour_matvec(c_H, v_H,self.Ns, self.fourier_laplacian, v, tol, self.shifts, bin_entropy_fermi_dirac_weights)
        estimate_density_function = (jnp.diag(square_root_fermi_dirac_H_v.dot(square_root_fermi_dirac_H_v.T))/N_samples).reshape([self.N_vec, 1]).real 

        # Estimate the energy
        # There are three parts: sinple electron term, Hartree term, and entropy term
        # single electron term is tr(C*P) = tr(-1/2 K P) + v_{ext} \otimes rho
        
        # kinetic energy 
        energy_kinetic = -1/2 * jnp.trace(square_root_fermi_dirac_H_v.T.dot(fftnProduct(self.fourier_laplacian, square_root_fermi_dirac_H_v))).real/N_samples

        # External potential energy
        energy_external = self.energy_external(estimate_density_function)

        # Hartree energy 
        energy_yukawa = self.energy_yukawa(estimate_density_function)

        # Entropy energy
        energy_entropy = 1/self.beta * jnp.trace(v.T.dot(bin_entropy_fermi_dirac_H_v).real)/N_samples

        # Total energy
        energy_free = energy_kinetic + energy_external + energy_yukawa + energy_entropy

        # print(energy_kinetic, energy_external, energy_hartree, energy_entropy)
        # return jnp.real(energy_kinetic), jnp.real(energy_external), jnp.real(energy_hartree), jnp.real(energy_entropy)
        objective = energy_free - self.mu * jnp.sum(estimate_density_function)
        return objective, energy_free, [jnp.real(energy_kinetic), jnp.real(energy_external), jnp.real(energy_yukawa), jnp.real(energy_entropy)]


if __name__ == "__main__":

    # Genereate two Hamiltonians
    Ns = (11,11,11)
    Ls = (1,1,1)
    det_H = deterministicHamiltonian(Ns, Ls, beta=2)
    sto_H = StochasticHamiltonian(Ns, Ls, beta=2)
    sto_H.key = 24

    # Generate a random H
    c_H = 0
    # v_H = np.random.rand(Ns[0]*Ns[1]*Ns[2])
    v_H = np.cos(10*np.pi*np.arange(Ns[0]*Ns[1]*Ns[2])/Ns[0]/Ns[1]/Ns[2])*10
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
    print(sto_H.objective(c_H, v_H, N_samples=100))
