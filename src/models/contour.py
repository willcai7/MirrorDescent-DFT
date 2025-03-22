import matplotlib.pyplot as plt
import scipyx as spx
import scipy.special
import numpy as np
import jax
import jax.numpy.linalg as jli
from jax import numpy as jnp
from jax import scipy as jsp
from jax import value_and_grad, vmap, lax, jit, random
from jax.experimental.sparse import BCOO
from jax.experimental.sparse import COO
from jax._src.api import block_until_ready
from functools import partial
jax.config.update("jax_enable_x64", True)

def gen_contour(Em, EM, beta, N, mu=0, function=None, clip=True):
    """
    Generate a contour for a matrix M based on its minimal and maximal eigenvalues.
    This contour is used to perform matrix-vector (matvec) operations.
    
    Args:
      Em: (Float) Minimal eigenvalue of M.
      EM: (Float) Maximal eigenvalue of M.
      mu: (Float) Chemical potential (default is 0).
      beta: (Float) Inverse temperature.
      N: (Int) Number of poles to compute.
      fermi_dirac: (Bool) If True, apply a Fermi-Dirac weighting to the poles.
    
    Returns:
      xi: (np.array [2N]) Poles along the contour.
      w: (np.array [2N]) Weights associated with each pole.
    """
    # Define a constant m using (pi/2)^2; used in the transformation.
    m = (np.pi/2) ** 2

    # Scale the minimal and maximal eigenvalues with the chemical potential and inverse temperature.
    if clip:
        low_b = max((Em - mu) * beta,-500)
        high_b = min((EM - mu) * beta,500)
    else:
        low_b = (Em - mu) * beta
        high_b = (EM - mu) * beta

    # Compute an effective maximum squared value, ensuring it is at least as large as the square of the largest scaled eigenvalue.
    # Adding m guarantees that the value is strictly positive.
    M = np.maximum(np.abs(low_b), np.abs(high_b)) ** 2 + m

    # Calculate the parameter k used for elliptic integral transformations.
    k = (np.sqrt(M / m) - 1) / (np.sqrt(M / m) + 1)
    # print("M",M)
    # print("m",m)
    # print("K",k)

    # Compute the complete elliptic integrals of the first kind:
    # K for modulus squared (k^2) and Kp for the complementary modulus (1 - k^2).
    K = scipy.special.ellipk(k ** 2)
    # print("K",K)
    Kp = scipy.special.ellipk(1 - k ** 2)
    # print("Kp",Kp)

    # Generate an array of complex parameters t along the contour.
    # t includes an imaginary shift (Kp/2 * 1j) and is uniformly spaced along the real part after a shift by -K.
    t = 1j * Kp / 2 - K + np.linspace(0.5, N - 0.5, N) * (2 * K / N)

    # Evaluate the Jacobi elliptic functions sn, cn, and dn for the array t.
    ej = spx.ellipj(t, k**2)
    sn = ej[0]  # Jacobi sine function
    cn = ej[1]  # Jacobi cosine function
    dn = ej[2]  # Delta amplitude

    # Apply a conformal mapping using the computed elliptic functions to obtain z in the complex plane.
    z = np.sqrt(m * M) * (1 / k + sn) / (1 / k - sn)

    # Compute the poles xi by taking the square root of the shifted z.
    xi = np.sqrt(z - m)
    # Generate the mirrored poles to obtain a symmetric contour.
    neg_xi = -xi

    # Calculate the weights for the positive poles.
    w_xi = (-2 * K * np.sqrt(m * M)) / (np.pi * N * k) * cn * dn / xi / ((1 / k - sn) ** 2)
    # Calculate the weights for the negative poles.
    w_neg_xi = (-2 * K * np.sqrt(m * M)) / (np.pi * N * k) * cn * dn / neg_xi / ((1 / k - sn) ** 2)

    # Combine the weights and poles from both halves of the contour.
    w = np.hstack([w_xi, w_neg_xi])
    xi = np.hstack([xi, -xi])

    # If merging values and weights is requested, modify the weights accordingly.
    if function == "fermi_dirac":
        w = w * complex_square_root_fermi_dirac(xi)
    
    elif function == "binary_entropy":
        w = w * complex_bin_entropy_fermi_dirac(xi)
        
    # Scale the poles and weights by beta and return.
    return xi/beta , w /beta 

def contour_f(x, xi, w):
    """
    Evaluate the contour integral of a function f along a contour defined by poles xi and weights w.
    Handles batches of x values efficiently.
    
    Args:
        x: (np.array [batch_size]) Input points to evaluate
        xi: (np.array [n_poles]) Contour poles
        w: (np.array [n_poles]) Contour weights
        
    Returns:
        (np.array [batch_size]) Contour integral evaluated at each x
    """
    # Reshape x to [batch_size, 1] and xi to [1, n_poles] for broadcasting
    x = np.asarray(x)[..., np.newaxis]
    xi = np.asarray(xi)[np.newaxis, :]
    w = np.asarray(w)[np.newaxis, :]
    
    # Compute 1/(xi-x) for all x,xi pairs at once
    denominators = 1.0 / (xi - x)
    
    # Multiply by weights and sum over poles dimension
    values = np.sum(w * denominators, axis=-1)
    
    return np.imag(values)


def complex_square_root_fermi_dirac(Z):
    """
    The holomorphic extension of square root of Fermi-Dirac function.
    Args:
      Z: (np.array [n,m]) The input of square root Fermi-Dirac function.

    Returns:
      S: (np.array [n,m]) The result of pointwise square root FD function applied to Z.
    """
    Z = jnp.asarray(Z)
    X = jnp.real(Z)/2
    Y = jnp.imag(Z)/2
    R = jnp.abs((1 - jnp.tanh(X + 1j * Y)) / 2)
    Theta = 2 * jnp.pi - 2 * jnp.pi * jnp.floor((Y - jnp.pi / 2) / jnp.pi) + jnp.angle(1 - jnp.tanh(X + 1j * Y))
    S_right = jnp.sqrt(R) * jnp.exp(1j * Theta / 2)
    S_left = jnp.sqrt((1 - jnp.tanh(X + 1j * Y)) / 2)
    S = S_left * (X <= 0) + S_right * (X > 0)
    return S

def complex_log_one_plus_exp(z):
    """
    Compute the complex logarithm of (1 + exp(z)) with branch cut adjustments
    to maintain continuity across the complex plane.
    
    This function computes two versions of the logarithm:
      - The standard logarithm log(1 + exp(z)) when the real part of z is non-positive.
      - An adjusted logarithm that corrects the phase when the real part of z is positive.
    
    Args:
        z : complex or numpy array of complex
            Input complex number(s).
    
    Returns:
        complex or numpy array of complex
            The computed complex logarithm of (1 + exp(z)) with appropriate branch adjustments.
    """
    # Extract the real and imaginary parts of z
    real_part = np.real(z)
    imag_part = np.imag(z)
    real_part = np.clip(real_part, -200, 200)
    z = real_part + 1j * imag_part
    
    # Compute the magnitude of (1 + exp(z))
    magnitude = np.abs(1 + np.exp(z))
    
    # Compute the standard logarithm of (1 + exp(z))
    standard_log = np.log(1 + np.exp(z))
    
    # Compute the phase (argument) of (1 + exp(z))
    phase_angle = np.angle(1 + np.exp(z))
    
    # Adjust the phase angle to ensure continuity (adjust branch cut)
    phase_angle += 2 * np.pi * np.ceil((imag_part - np.pi) / (2 * np.pi))
    
    # Compute the adjusted logarithm using the magnitude and modified phase angle
    adjusted_log = np.log(magnitude) + 1j * phase_angle
    
    # Select the appropriate logarithm based on the real part of z:
    # If real_part <= 0, use standard_log; otherwise, use adjusted_log.
    complex_log = standard_log * (real_part <= 0) + adjusted_log * (real_part > 0)
    
    return complex_log

def complex_entropy_fermi_dirac(z):
    """
    Compute the log(1 + exp(z))/(1+exp(z)) with branch cut adjustments
    to maintain continuity across the complex plane.
    
    This function computes two versions of the logarithm:
      - The standard logarithm log(1 + exp(z)) when the real part of z is non-positive.
      - An adjusted logarithm that corrects the phase when the real part of z is positive.
    
    Args:
        z : complex or numpy array of complex
            Input complex number(s).
    
    Returns:
        complex or numpy array of complex
            The computed complex logarithm of (1 + exp(z)) with appropriate branch adjustments.
    """
    # Extract the real and imaginary parts of z
    real_part = np.real(z)
    imag_part = np.imag(z)
    # fz = 1 + np.exp(z)
    real_part = np.clip(real_part, -200, 200)
    z = real_part + 1j * imag_part
    fz = 1 + np.exp(z)
    
    # Compute the magnitude of (1 + exp(z))
    magnitude = np.abs(fz)
    
    # Compute the standard logarithm of (1 + exp(z))
    standard_log = np.log(fz)
    
    # Compute the phase (argument) of (1 + exp(z))
    phase_angle = np.angle(fz)
    
    # Adjust the phase angle to ensure continuity (adjust branch cut)
    phase_angle += 2 * np.pi * np.ceil((imag_part - np.pi) / (2 * np.pi))
    
    # Compute the adjusted logarithm using the magnitude and modified phase angle
    adjusted_log = np.log(magnitude) + 1j * phase_angle
    
    # Select the appropriate logarithm based on the real part of z:
    # If real_part <= 0, use standard_log; otherwise, use adjusted_log.
    complex_log = standard_log * (real_part <= 0) + adjusted_log * (real_part > 0)

    return -complex_log / fz

def complex_bin_entropy_fermi_dirac(z):
    """
    Compute the binary entropy for the Fermi-Dirac distribution.
    """
    return complex_entropy_fermi_dirac(z) + complex_entropy_fermi_dirac(-z)


@jit
def contour_matvec(c_H, v_H, D, v, tol, shifts, weights, batch_size=20):
    """
    Compute matrix-vector products along a contour in batches using shift-and-invert.
    
    Args:
        c_H: (Float) Scalar coefficient for the Hamiltonian
        v_H: (jnp.array [N_vec]) Vector potential term
        D: (jnp.array [Ns]) Discretized Laplacian eigenvalues
        v: (jnp.array [N_vec, N_samples]) Input vectors to multiply with
        tol: (Float) Tolerance for the linear solver
        shifts: (jnp.array [N_poles]) Complex shifts along the contour
        weights: (jnp.array [N_poles]) Complex weights for the contour integration
        batch_size: (Int) Number of shifts to process in parallel
        
    Returns:
        (jnp.array [N_vec, N_samples]) Imaginary part of the weighted sum of matrix-vector products
    """
    results = []
    if len(weights.shape) > 1:  # Check if weights is a matrix
        for i in range(0, len(shifts), batch_size):
            batch_shifts = shifts[i:i+batch_size]
            batch_weights = weights[i:i+batch_size]
            x = vmap(shift_inv_system, in_axes=(None, None, None, None, 0, None))(c_H, v_H,  D, v, batch_shifts, tol)
            results.append(jnp.einsum('ijk,il->jkl', x, batch_weights))
    else:  # weights is a 1D array
        for i in range(0, len(shifts), batch_size):
            batch_shifts = shifts[i:i+batch_size]
            batch_weights = weights[i:i+batch_size]
            x = vmap(shift_inv_system, in_axes=(None, None, None, None, 0, None))(c_H, v_H,  D, v, batch_shifts, tol)
            results.append(jnp.einsum('ijk,i->jk', x, batch_weights))
    return jnp.imag(sum(results))
    
@jit
def shift_inv_system(c_H, v_H, D, v, shift, tol): 
    """
    Solve the shifted linear system (shift - H)x = v using BiCGSTAB with FFT preconditioning.
    
    Args:
        c_H: (Float) Scalar coefficient for the Hamiltonian
        v_H: (jnp.array [N_vec]) Vector potential term
        D: (jnp.array [Ns]) Discretized Laplacian eigenvalues
        v: (jnp.array [N_vec, N_samples]) Right-hand side vector
        shift: (complex) Complex shift
        tol: (Float) Tolerance for convergence
        
    Returns:
        (jnp.array [N_vec, N_samples]) Solution vector x satisfying (shift - H)x = v
    """
    Ns = D.shape
    @jit
    def linearmap(u):
        # Apply the shifted Hamiltonian operator (shift - H)u
        temp_v = v_H[:, None] * u
        shape = list(Ns) + [-1]
        axes = list(range(len(Ns)))
        u = jnp.fft.fftn(u.reshape(shape), axes=axes)
        u = jnp.multiply((shift - c_H * D)[..., None], u)
        u = jnp.fft.ifftn(u, axes=axes).reshape([-1, u.shape[-1]])
        u = u - temp_v
        return u

    @jit
    def precondition_linearmap(u):
        # Apply the inverse of the diagonal part as preconditioner
        shape = list(Ns) + [-1]
        axes = list(range(len(Ns)))
        u = jnp.fft.fftn(u.reshape(shape), axes=axes)
        u = jnp.multiply((1 / (shift - c_H * D  - jnp.mean(v_H)))[..., None], u)
        u = jnp.fft.ifftn(u, axes=axes).reshape([-1, u.shape[-1]])
        return u

    return jax.scipy.sparse.linalg.bicgstab(linearmap, v, tol=tol, M=precondition_linearmap, maxiter=1000)[0]


if __name__ == "__main__":
    EM = 1
    Em = -1
    mu = 0
    xi, w = gen_contour(Em, EM, 1, 4, mu=mu, function='fermi_dirac')
    # input = jnp.linspace(-10, 10, 100) + 1j*jnp.linspace(-10, 10, 100)[:, jnp.newaxis]
    # output = complex_square_root_fermi_dirac(input)
    # input_real = jnp.linspace(-10, 10, 100) 
    # input_real = jnp.complex128(input_real)
    # output_real = complex_square_root_fermi_dirac(input_real)
    # output_real = jnp.real(output_real)
    # true_output_real = jnp.sqrt(1/(1+jnp.exp(input_real)))

    # plt.figure(figsize=(10,4), dpi=100)
    # plt.subplot(1,2,1)
    # # plt.figure(figsize=(4,8), dpi=100)
    # plt.imshow(np.real(output), extent=(-7, 7, -7, 7))
    # plt.colorbar()
    # plt.scatter([0,0],[2,-2], color='red')
    # plt.title("Real part")
    # # plt.show()

    # plt.subplot(1,2,2)
    # # plt.figure(figsize=(8,8), dpi=100)
    # plt.imshow(np.imag(output), extent=(-7, 7, -7, 7))
    # plt.colorbar()
    # plt.scatter([0,0],[2,-2], color='red')
    # plt.title("Imaginary part")
    # plt.savefig("contour_real_imag.png")

    # plt.figure(figsize=(10,4), dpi=100)
    # plt.subplot(1,2,1)
    # plt.plot(input_real, output_real, label="implementation")
    # plt.plot(input_real, true_output_real, label="true")
    # plt.legend()
    # plt.title("Real part")
    # plt.xlabel("Input")
    # plt.ylabel("Output")

    # plt.subplot(1,2,2)
    # plt.semilogy(input_real, np.abs(output_real-true_output_real))
    # plt.title("Error")
    # plt.xlabel("Input")
    # plt.ylabel("Error")
    # plt.tight_layout()
    # plt.savefig("contour_real_error.png")
