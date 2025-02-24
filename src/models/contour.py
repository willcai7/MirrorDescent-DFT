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

def gen_contour(Em, EM, beta, N, mu=0, fermi_dirac=False):
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
    m = (np.pi / (2 * 1)) ** 2

    # Scale the minimal and maximal eigenvalues with the chemical potential and inverse temperature.
    low_b = (Em - mu) * beta
    high_b = (EM - mu) * beta

    # Compute an effective maximum squared value, ensuring it is at least as large as the square of the largest scaled eigenvalue.
    # Adding m guarantees that the value is strictly positive.
    M = np.maximum(np.abs(low_b), np.abs(high_b)) ** 2 + m

    # Calculate the parameter k used for elliptic integral transformations.
    k = (np.sqrt(M - m) - 1) / (np.sqrt(M - m) + 1)

    # Compute the complete elliptic integrals of the first kind:
    # K for modulus squared (k^2) and Kp for the complementary modulus (1 - k^2).
    K = scipy.special.ellipk(k ** 2)
    Kp = scipy.special.ellipk(1 - k ** 2)

    # Generate an array of complex parameters t along the contour.
    # t includes an imaginary shift (Kp/2 * 1j) and is uniformly spaced along the real part after a shift by -K.
    t = 1j * Kp / 2 - K + np.linspace(0.5, N - 0.5, N) * (2 * K / N)

    # Evaluate the Jacobi elliptic functions sn, cn, and dn for the array t.
    ej = spx.ellipj(t, k ** 2)
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

    # If Fermi-Dirac weighting is requested, modify the weights accordingly.
    if fermi_dirac:
        w = w * complex_square_root_fermidirac(xi)

    # Scale the poles and weights by beta and return.
    return xi / beta, w / beta

def contour_f(x,xi,w):
    """
    Evaluate the contour integral of a function f along a contour defined by poles xi and weights w.
    """
    value = sum(w/(xi-x))
    return np.imag(value)

def complex_square_root_fermidirac(Z):
    """
    A holomorphic extension of the square-root Fermi-Dirac function. 
    This function is holomorphic in the regions {Re(z) < 0} and {Re(z) > 0} 
    as well as in the strip {Re(z) = 1, -2 < Im(z) < 2}.

    Args:
        Z: (np.array) Complex number(s).

    Returns:
        S: (np.array) Holomorphic extension of the square-root Fermi-Dirac function.
    """
    # Extract half of the real part of Z
    X = np.real(Z) / 2

    # Extract half of the imaginary part of Z
    Y = np.imag(Z) / 2

    # Compute the magnitude component:
    # Calculate the absolute value of (1 - tanh(X + iY)) / 2.
    # This forms the basis for the square-root transformation.
    R = np.abs((1 - np.tanh(X + 1j * Y)) / 2)

    # Compute the adjusted phase angle:
    # Start with the phase of (1 - tanh(X + iY)) and adjust it by adding a multiple of 2*pi.
    # The floor function is used here to ensure the phase is continuously adjusted 
    # across branch cuts, aligning with the desired holomorphic extension.
    Theta = 2 * np.pi - 2 * np.pi * np.floor((Y - np.pi / 2) / np.pi) + np.angle(1 - np.tanh(X + 1j * Y))

    # Compute the "right" branch (for X >= 0) of the square-root Fermi-Dirac function:
    # Take the square root of the magnitude R and apply half the adjusted phase angle.
    S_right = np.sqrt(R) * np.exp(1j * Theta / 2)

    # Compute the "left" branch (for X <= 0) directly:
    # Here we take the square root of (1 - tanh(X + iY)) / 2 without phase adjustment.
    S_left = np.sqrt((1 - np.tanh(X + 1j * Y)) / 2)

    # Select the appropriate branch based on the sign of X:
    # Use S_left when the real part (after halving) is non-positive,
    # and S_right when it is non-negative.
    S = S_left * (X <= 0) + S_right * (X >= 0)

    # Return the holomorphic extension of the square-root Fermi-Dirac function.
    return S


# The function is jitted with JAX for efficiency, treating the first and third arguments as static.
@partial(jax.jit, static_argnums=(0, 2))
def shift_inv_system(c_H, v_H, Ns, D, v, shift, tol): 
    """
    Solve a shifted linear system using the BiCGStab iterative solver. Here the linear system is defined by an abstract operator. 
    $$ A = shift - c_H * F* D* F^* - diag(v_H) $$
    We are solving the system $A x = v$ for $x$.
    
    Args:
        c_H: (real) Coefficient for the Laplacian.
        v_H: (np.array [Ns[0]*Ns[1]*Ns[2]]) Diagonal operator.
        Ns: (np.array [3]) Number of grid points in each dimension.
        D: (np.array [Ns[0]*Ns[1]*Ns[2]]) Discretized Laplacian eigenvalues.
        v: (np.array [Ns[0]*Ns[1]*Ns[2], n]) Right-hand side vector.
        shift: (np.array) Shift value for the linear system.
        tol: (real) Convergence tolerance for the iterative solver.

    Returns:
        x: (np.array [Ns[0]*Ns[1]*Ns[2], n]) Solution to the linear system.

    """
    # Define the linear operator for the shifted system
    @jit
    def linearmap(u):
        """
        Apply the linear operator to a vector u.
        The operator is defined as A = shift - c_H * F*D*F^* - diag(v_H).
        """

        # Multiply u by the vector v_H (expanded along a new axis) to create a term to subtract later.
        temp_v = v_H[:, None] * u

        # Reshape u to a multi-dimensional array with spatial dimensions Ns[0], Ns[1], Ns[2] 
        # and perform a forward FFT along the spatial axes.
        u = jnp.fft.fftn(u.reshape([Ns[0], Ns[1], Ns[2], -1]), axes=[0, 1, 2])

        # Multiply the FFT result element-wise by (shift - c_H * D).
        # The extra axis is added to match the dimensions for broadcasting.
        u = jnp.multiply((shift - c_H * D)[:, :, :, None], u)

        # Transform the product back to real space using the inverse FFT,
        # then flatten the spatial dimensions.
        u = jnp.fft.ifftn(u, axes=[0, 1, 2]).reshape([Ns[0] * Ns[1] * Ns[2], -1])

        # Subtract the previously computed term (temp_v) to complete the linear mapping.
        u = u - temp_v
        return u

    # Define the preconditioner for the system, which approximates the inverse of the operator.
    @jit
    def precondition_linearmap(u):
        """
        Apply the preconditioner to a vector u.
        The preconditioner is defined as the reciprocal of (shift - c_H * F*D*F^*).
        """
        # Reshape u to the spatial grid dimensions and perform a forward FFT.
        u = jnp.fft.fftn(u.reshape([Ns[0], Ns[1], Ns[2], -1]), axes=[0, 1, 2])

        # Multiply element-wise by the reciprocal of (shift - c_H * D) to precondition the system.
        u = jnp.multiply((1 / (shift - c_H * D))[:, :, :, None], u)

        # Perform the inverse FFT to transform back to real space and flatten the spatial dimensions.
        u = jnp.fft.ifftn(u, axes=[0, 1, 2]).reshape([Ns[0] * Ns[1] * Ns[2], -1])
        return u

    # Use the BiCGStab iterative solver from JAX's sparse linear algebra library to solve the linear system.
    # - 'linearmap' defines the linear operator.
    # - 'v' is the right-hand side vector.
    # - 'tol' sets the convergence tolerance.
    # - 'M=precondition_linearmap' provides the fixed preconditioner.
    # - 'maxiter=100' sets the maximum number of iterations.
    # The solver returns a tuple where the first element is the solution.
    return jax.scipy.sparse.linalg.bicgstab(linearmap, v, tol=tol, M=precondition_linearmap, maxiter=100)[0]

# The function is jitted with JAX for efficiency, treating the first and third arguments as static.

@partial(jax.jit, static_argnums=(0,2))
def contour_matvec(c_H, v_H, Ns, D, v, tol, shifts, weights):
    """
    Perform a matrix-vector product using a contour integral method.
    The method involves solving a shifted linear system for each shift in the contour.
    The results are then combined using the weights associated with each shift.

    Args:
        c_H: (real) Coefficient for the Laplacian.
        v_H: (np.array [Ns[0]*Ns[1]*Ns[2]]) Diagonal operator.
        Ns: (np.array [3]) Number of grid points in each dimension.
        D: (np.array [Ns[0]*Ns[1]*Ns[2]]) Discretized Laplacian eigenvalues.
        v: (np.array [Ns[0]*Ns[1]*Ns[2], n]) Right-hand side vector.
        tol: (real) Convergence tolerance for the iterative solver.
        shifts: (np.array [2N]) Shift values for the contour.
        weights: (np.array [2N]) Weights associated with each shift.
    
    Returns:
        x: (np.array [Ns[0]*Ns[1]*Ns[2], n]) Result of the contour matrix-vector product.
    """
    
    # Apply the shift_inv_system function to each element in the 'shifts' array.
    # vmap vectorizes the computation over shifts (in_axes=0 for shifts) while keeping
    # other parameters (c_H, v_H, Ns, D, v, tol) fixed across all evaluations.
    x = vmap(shift_inv_system, in_axes=(None, None, None, None, None, 0, None))(
        c_H, v_H, Ns, D, v, shifts, tol
    )
    
    # Use Einstein summation to combine the results 'x' weighted by the 'weights' vector.
    # The einsum 'ijk,i->jk' multiplies each slice along the first axis (corresponding to each shift)
    # by the corresponding weight and then sums over that axis.
    weighted_sum = jnp.einsum('ijk,i->jk', x, weights)
    
    # Return only the imaginary part of the weighted sum.
    # This extracts the final contour matrix-vector product result.
    return jnp.imag(weighted_sum)