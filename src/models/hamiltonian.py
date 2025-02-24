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
    D = jnp.array(sum([jnp.cos(Ks[i]) for i in range(3)]))
    D = D.flatten()
    return D

def dense_logm(P):
    w, V = jnp.linalg.eigh(P)
    return V @ jnp.diag(jnp.log(w)) @ V.T

def dense_tr_xlogx_m(P):
    w, _ = jnp.linalg.eigh(P)
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
       lambda =  [ (2*pi*k1/L1)^2 + (2*pi*k2/L2)^2 + (2*pi*k3/L3)^2 ].
    
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
    D = 1/D / (Ls[0]*Ls[1]*Ls[2])
    D[0,0,0] = 0
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

    eigenvalues = 1/eigenvalues / (Ls[0]*Ls[1]*Ls[2])
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

        slef.fourier = fourier 
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

    @partial(jit, static_argnums=(0,))
    def potential_yukawa(self, rho):
        return fftnProduct(self.fourier_yukawa, rho)

    @partial(jit, static_argnums=(0,))
    def energy_yukawa(self, rho):
        return jnp.sum(rho * self.potential_yukawa(rho))/2

    @partial(jit, static_argnums=(0,))
    def energy_external(self, rho):
        return jnp.sum(rho.flatten() * self.potential_external)


class deterministicHamiltonian(Hamiltonian):
    def __init__(self, Ns, Ls, beta=1,mu=0, alpha=0, fourier=True, dense=True):
        super().__init__(Ns, Ls, beta, mu, alpha, fourier, dense)
        assert dense, "Only dense Hamiltonian is supported for deterministic Hamiltonian"
        self.C = - self.dense_laplacian/2 + jnp.diag(self.potential_external)

    @partial(jit, static_argnums=(0,))
    def energy_laplacian(self, P):
        return jnp.trace( - P @ self.dense_laplacian) /2 
    
    @partial(jit, static_argnums=(0,))
    def energy_free(self, H):
        P = self.density_matrix(H)
        rho = jnp.diag(P)
        return self.energy_external(rho) + self.energy_laplacian(P) + self.energy_yukawa(rho) + 1/self.beta * BinEntropy(P)
    
    @partial(jit, static_argnums=(0,))
    def objective(self, H):
        P = self.density_matrix(H)
        rho = jnp.diag(P)
        return self.energy_external(rho) + self.energy_laplacian(P) + self.energy_yukawa(rho) + 1/self.beta * BinEntropy(P) - self.mu * jnp.sum(rho)
    
    @partial(jit, static_argnums=(0,))
    def density_matrix(self, H):
        w, V = jnp.linalg.eigh(H)
        w1 = 1/(1+jnp.exp(self.beta*w))
        return V @ jnp.diag(w1) @ V.T
    
    @partial(jit, static_argnums=(0,))
    def gradient(self, H):
        P = self.density_matrix(H)
        rho = jnp.diag(P)
        grad = self.C + jnp.diag(self.potential_yukawa(rho))
        return grad

class StochasticHamiltonian(Hamiltonian):
    def __init__(self, Ns, Ls, beta=1,mu=0, N_poles=100, alpha=0, fourier=True, dense=False):
        super().__init__(Ns, Ls, beta, mu, alpha, fourier, dense)
        assert fourier, "Only Fourier Hamiltonian is supported for stochastic Hamiltonian"
        self.shifts = None 
        self.weights = None 
        self.key = 0
        self.N_poles = N_poles
        self.N_vec = Ns[0]*Ns[1]*Ns[2]
    
    def update_poles_shifts(self, c_H, v_H):
        Em = -np.max(self.fourier_laplacian)/2*c_H + np.min(v_H)
        EM = -np.min(self.fourier_laplacian)/2*c_H + np.max(v_H)
        self.shifts, self.weights = gen_contour(Em, EM, self.beta, self.N_poles, fermi_dirac=True)

    @partial(jit, static_argnums=(0,))
    def stochastic_estimates(self, c_H, v_H, N_samples=100, target='grad', tol=1e-6):

        # Generate Gaussian random vectors
        v = jnp.complex128(jnp.random.normal(jnp.random.PRNGKey(self.key), (self.N_vec, N_samples)))
        
        # Compute the matrix-vector product
        fv = contour_matvec(c_H, v_H,self.Ns, self.fourier_laplacian, v, tol, self.shifts, self.weights)
        
        self.key += 1 # Update the key for the next random number generation
        
        if target == 'grad': # Gradient
            return (jnp.diag(fv.dot(fv.T))/N_samples).rehsape([H.N_vec, 1]).real 
        
        elif target == 'energy': # Energy
            rho_esti = (jnp.diag(fv.dot(fv.T))/N_samples).rehsape([H.N_vec, 1]).real 
            





    

        


    


    
    
    

        
    



