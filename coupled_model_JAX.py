import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from microstructure_generator import generate_microstructure_from_files

jax.config.update("jax_enable_x64", True)

# Problem setup
N = 512
L = 1.0
dx = L / N

MODE_CUTOFF = N // 2
MODE_SHAPE = "radial"

# Thermal loading
G_macro = jnp.array([0.1, 0.02])
T_mean = 250.0
T_ref = 20.0

# Mechanical loading
E_macro = jnp.array([[0.05, 0.00],
                     [0.00, 0.00]])

# Mechanical material parameters
E_matrix = 14.401103984819736
nu_matrix = 0.22001897533206832
E_incl = 72.00551992409868
nu_incl = 0.22001897533206832

# Thermal material parameters
lambda_matrix = 1.0
lambda_incl = 2.0
alpha_matrix = 5.0e-5
alpha_incl = 1.0e-4

# Iteration parameters
thermal_max_iter = 1000
mechanical_max_iter = 1000
thermal_tol = 1e-8
mechanical_tol = 1e-8
thermal_relaxation = 1.0
mechanical_relaxation = 1.0

# Heat source S(X)
S = jnp.zeros((N, N))

nx = N // 2
ny = N // 2

S = S.at[nx, ny].set(0.0)
S = S - jnp.mean(S)
S_hat = jnp.fft.fft2(S, axes=(0, 1))

# Compatibility check: periodic Poisson problems require zero mean source.
if abs(float(jnp.mean(S))) > 1e-14:
    raise ValueError("For periodic boundary conditions, the heat source S must have zero mean.")

## 2D COMPOSITE RVE GEOMETRY

x = jnp.linspace(0.0, L, N, endpoint=False)
y = jnp.linspace(0.0, L, N, endpoint=False)
X, Y = jnp.meshgrid(x, y, indexing="ij")

#cx = 0.5
#cy = 0.5
#radius = 0.23

#r = jnp.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
#inclusion = r < radius

sample_id = 8281
phase = generate_microstructure_from_files(X, Y, sample_id=sample_id)

## FOURIER GRID
freq = 2.0 * jnp.pi * jnp.fft.fftfreq(N, d=dx)
kx, ky = jnp.meshgrid(freq, freq, indexing="ij")

kvec = jnp.stack([kx, ky], axis=-1)
k2 = kx**2 + ky**2

mode_index = jnp.fft.fftfreq(N) * N
mx, my = jnp.meshgrid(mode_index, mode_index, indexing="ij")

if MODE_SHAPE == "radial":
    fourier_mask = (mx**2 + my**2) <= MODE_CUTOFF**2
elif MODE_SHAPE == "square":
    fourier_mask = (jnp.abs(mx) <= MODE_CUTOFF) & (jnp.abs(my) <= MODE_CUTOFF)
else:
    raise ValueError("MODE_SHAPE must be 'radial' or 'square'.")

fourier_mask = fourier_mask.at[0, 0].set(True)

#######################################################################
# HELPER FUNCTIONS
#######################################################################

def lame_parameters(E, nu):
    mu = E / (2.0 * (1.0 + nu))
    lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam, mu


def apply_mode_filter(field_hat, mask):
    return jnp.where(mask[(...,) + (None,) * (field_hat.ndim - 2)], field_hat, 0.0)


@jax.jit
def heat_flux(lambda_field, gradT):
    """
    Fourier/PDF sign convention:
        Q = -lambda(X) grad T
    Q is a 2D vector field.
    """
    return -lambda_field[..., None] * gradT


@jax.jit
def perturbation_field(lambda0, lambda_field, gradT):
    """
    With
        Q = -lambda0 grad T + P
    so
        P = lambda0 grad T + Q
          = (lambda0 - lambda(X)) grad T
    P is also a 2D vector field.
    """
    Q = heat_flux(lambda_field, gradT)
    return lambda0 * gradT + Q


@jax.jit
def fft_gradient_update(P_hat, S_hat, mask, kvec, k2, lambda0):
    """
    Solve the Fourier-space scalar temperature equation for nonzero modes:
        T_hat(k) = [S_hat(k) - i k.P_hat(k)] / [lambda0 |k|^2]

    Then compute the perturbation gradient:
        grad(T)_hat(k) = i k T_hat(k)

    The k=0 mode is set to zero here because the imposed mean gradient
    is added separately as G_macro in real space.
    """
    P_hat = apply_mode_filter(P_hat, mask)
    S_hat = jnp.where(mask, S_hat, 0.0 + 0.0j)

    k_dot_P_hat = jnp.einsum("...i,...i->...", kvec, P_hat)
    active = mask & (k2 >= 1e-14)

    numerator = S_hat - 1j * k_dot_P_hat
    T_hat = jnp.where(active, numerator / (lambda0 * k2), 0.0 + 0.0j)
    T_hat = T_hat.at[0, 0].set(0.0 + 0.0j)

    gradT_hat = 1j * kvec * T_hat[..., None]
    gradT_hat = apply_mode_filter(gradT_hat, mask)

    gradT_fluct = jnp.fft.ifft2(gradT_hat, axes=(0, 1)).real
    return gradT_fluct, T_hat


@jax.jit
def compute_flux_residual(Q, S, mask, kvec):
    """
    Compute relative equilibrium residual in Fourier space:
        || div Q || / || Q ||
    where div Q = i k.Q_hat
    Note that heat source will be included here as well if it exists
    """
    S_hat = jnp.fft.fft2(S, axes=(0, 1))
    S_hat = apply_mode_filter(S_hat, mask)
    Q_hat = jnp.fft.fft2(Q, axes=(0, 1))
    Q_hat = apply_mode_filter(Q_hat, mask)

    div_Q_hat = 1j * jnp.einsum("...i,...i->...", kvec, Q_hat)

    residual_hat = -div_Q_hat + S_hat
    numerator = jnp.linalg.norm(residual_hat)
    denominator = jnp.linalg.norm(S_hat) + jnp.linalg.norm(Q_hat) + 1e-14
    return numerator / denominator


@jax.jit
def thermal_strain(alpha_field, dT):
    """
    Compute the thermal strain
    For the computation of the mechanical strain
    eps_thermal = alpha delta_T I
    """
    I = jnp.eye(2, dtype=dT.dtype)
    return alpha_field[..., None, None] * dT[..., None, None] * I


@jax.jit
def stress_isotropic(eps, lam_field, mu_field, alpha_field, dT):
    """ sigma = lambda tr(eps - eps_thermal) I + 2 mu (eps - eps_thermal) """
    I = jnp.eye(2, dtype=eps.dtype)
    eps_mech = eps - thermal_strain(alpha_field, dT)
    tr = jnp.trace(eps_mech, axis1=-2, axis2=-1)

    sigma = (lam_field[..., None, None] * tr[..., None, None] * I) + (2.0 * mu_field[..., None, None] * eps_mech)

    return sigma


@jax.jit
def stress_reference(eps, lam0, mu0):
    tr = jnp.trace(eps, axis1=-2, axis2=-1)
    I = jnp.eye(2, dtype=eps.dtype)

    sigma0 = (lam0 * tr[..., None, None] * I) + (2.0 * mu0 * eps)

    return sigma0


@jax.jit
def fft_strain_update(tau_hat, mask, kvec, k2, lam0, mu0):
    # remove high-frequency modes before solving
    tau_hat = apply_mode_filter(tau_hat, mask)

    # Acoustic tensor:
    # A_ij = mu0 |k|^2 delta_ij + (lam0 + mu0) k_i k_j
    I = jnp.eye(2, dtype=tau_hat.real.dtype)

    A = (mu0 * k2[..., None, None]) * I + ((lam0 + mu0) * kvec[..., :, None] * kvec[..., None, :])
    
    # rhs_i = i * k_j * tau_ij
    rhs = 1j * jnp.einsum("...j,...ij->...i", kvec, tau_hat)

    # Filter for active modes
    # k = 0 and truncated modes are inactive.
    active = mask & (k2 >= 1e-14)

    # For inactive modes, replace A by identity and rhs by zero
    A_safe = jnp.where(active[..., None, None], A, I)
    rhs_safe = jnp.where(active[..., None], rhs, 0.0 + 0.0j)

    # Shape check point:
    # kvec     : (N, N, 2)
    # A_safe   : (N, N, 2, 2)
    # rhs_safe : (N, N, 2)
    # u_hat    : (N, N, 2)
    u_hat = jnp.linalg.solve(A_safe, rhs_safe[..., None])[..., 0]
    u_hat = jnp.where(active[..., None], u_hat, 0.0 + 0.0j)

    # Convert u_hat to eps_hat
    eps_hat = -0.5j * ((kvec[..., None, :] * u_hat[..., :, None]) + (kvec[..., :, None] * u_hat[..., None, :]))
    
    # Enforce zero fluctuation strain at mode 0
    eps_hat = eps_hat.at[0, 0, :, :].set(0.0 + 0.0j)
    
    # Remove noise outside selected modes
    eps_hat = apply_mode_filter(eps_hat, mask)

    eps_fluct = jnp.fft.ifft2(eps_hat, axes=(0, 1)).real

    return eps_fluct


@jax.jit
def compute_equilibrium_residual(sigma, mask, kvec):
    """
    Compute || div sigma || / || sigma || in Fourier space    
    """
    sigma_hat = jnp.fft.fft2(sigma, axes=(0, 1))
    sigma_hat = apply_mode_filter(sigma_hat, mask)

    div_sigma_hat = 1j * jnp.einsum("...j,...ij->...i", kvec, sigma_hat)

    return jnp.linalg.norm(div_sigma_hat) / (jnp.linalg.norm(sigma_hat) + 1e-14)


#######################################################################
# Calculating material parameters
#######################################################################

#lambda0 = 0.5 * (lambda_matrix + lambda_incl)
#lambda_field = jnp.where(inclusion, lambda_incl, lambda_matrix)
#alpha = jnp.where(inclusion, alpha_incl, alpha_matrix)

#lam_matrix, mu_matrix = lame_parameters(E_matrix, nu_matrix)
#lam_incl, mu_incl = lame_parameters(E_incl, nu_incl)

#lam0 = 0.5 * (lam_matrix + lam_incl)
#mu0 = 0.5 * (mu_matrix + mu_incl)

#lam = jnp.where(inclusion, lam_incl, lam_matrix)
#mu = jnp.where(inclusion, mu_incl, mu_matrix)

lambda_field = lambda_matrix + phase * (lambda_incl - lambda_matrix)
alpha = alpha_matrix + phase * (alpha_incl - alpha_matrix)

E_field = E_matrix + phase * (E_incl - E_matrix)
nu_field = nu_matrix + phase * (nu_incl - nu_matrix)

mu = E_field / (2.0 * (1.0 + nu_field))
lam = (E_field * nu_field) / ((1.0 + nu_field) * (1.0 - 2.0 * nu_field))

lambda0 = 0.5 * (lambda_matrix + lambda_incl)

lam_matrix_const, mu_matrix_const = lame_parameters(E_matrix, nu_matrix)
lam_incl_const, mu_incl_const = lame_parameters(E_incl, nu_incl)

lam0 = 0.5 * (lam_matrix_const + lam_incl_const)
mu0 = 0.5 * (mu_matrix_const + mu_incl_const)


#######################################################################
# THERMAL SOLVER
#######################################################################

gradT = jnp.broadcast_to(G_macro, (N, N, 2)).copy()
thermal_history = []
T_hat = jnp.zeros((N, N), dtype=jnp.complex128)

for it in range(thermal_max_iter):
    P = perturbation_field(lambda0, lambda_field, gradT)
    P_hat = jnp.fft.fft2(P, axes=(0, 1))

    gradT_fluct_new, T_hat = fft_gradient_update(P_hat, S_hat, fourier_mask, kvec, k2, lambda0)
    gradT_new = G_macro[None, None, :] + gradT_fluct_new

    diff = jnp.linalg.norm(gradT_new - gradT) / (jnp.linalg.norm(gradT) + 1e-14)
    gradT = (1.0 - thermal_relaxation) * gradT + thermal_relaxation * gradT_new

    Q = heat_flux(lambda_field, gradT)
    residual_active = compute_flux_residual(Q, S, fourier_mask, kvec)

    thermal_history.append((float(diff), float(residual_active)))

    if it % 20 == 0:
        print(f"thermal iter {it:4d} | gradient change = {diff:.3e} | active flux residual = {residual_active:.3e}")

    if diff < thermal_tol:
        print(f"Thermal converged at iteration {it}")
        break

Q = heat_flux(lambda_field, gradT)
Q_macro = jnp.mean(Q, axis=(0, 1))
G_mean = jnp.mean(gradT, axis=(0, 1))
lambda_eff_xx = -Q_macro[0] / G_macro[0]

T_fluct = jnp.fft.ifft2(T_hat, axes=(0, 1)).real
dT = T_mean + T_fluct - T_ref

#######################################################################
# MECHANICAL SOLVER
#######################################################################

eps = jnp.broadcast_to(E_macro, (N, N, 2, 2)).copy()
mechanical_history = []

for it in range(mechanical_max_iter):
    # eps_mech is calculated in the stress_isotropic function
    sigma = stress_isotropic(eps, lam, mu, alpha, dT)
    sigma0 = stress_reference(eps, lam0, mu0)

    tau = sigma0 - sigma
    tau_hat = jnp.fft.fft2(tau, axes=(0, 1))

    eps_fluct_new = fft_strain_update(tau_hat, fourier_mask, kvec, k2, lam0, mu0)
    eps_new = E_macro[None, None, :, :] + eps_fluct_new

    diff = jnp.linalg.norm(eps_new - eps) / (jnp.linalg.norm(eps) + 1e-14)
    eps = (1.0 - mechanical_relaxation) * eps + mechanical_relaxation * eps_new

    sigma = stress_isotropic(eps, lam, mu, alpha, dT)
    residual_active = compute_equilibrium_residual(sigma, fourier_mask, kvec)

    mechanical_history.append((float(diff), float(residual_active)))

    if it % 20 == 0:
        print(f"mechanical iter {it:4d} | strain change = {diff:.3e} | active residual = {residual_active:.3e}")

    if diff < mechanical_tol:
        print(f"Mechanical converged at iteration {it}")
        break

## Result #############################################################

sigma = stress_isotropic(eps, lam, mu, alpha, dT)
sigma_macro = jnp.mean(sigma, axis=(0, 1))
eps_th = thermal_strain(alpha, dT)

print("\nApplied macroscopic temperature gradient <grad T>:")
print(G_macro)

print("\nMean gradient check <grad T>:")
print(G_mean)

print("\nHomogenized/macroscopic heat flux <Q>:")
print(Q_macro)

print("\nEffective conductivity component lambda_eff_xx:")
print(lambda_eff_xx)

print("\nApplied macroscopic strain E_macro:")
print(E_macro)

print("\nHomogenized/macroscopic stress <sigma>:")
print(sigma_macro)

print("\nMean total strain check <eps>:")
print(jnp.mean(eps, axis=(0, 1)))

print("\nMean thermal strain check <eps_th>:")
print(jnp.mean(eps_th, axis=(0, 1)))

## Plot ###############################################################
thermal_history = np.array(thermal_history)
mechanical_history = np.array(mechanical_history)

plt.figure(figsize=(5, 4))
#plt.imshow(np.asarray(inclusion).T, origin="lower", extent=[0, L, 0, L])

plt.imshow(np.asarray(phase).T, origin="lower", extent=[0, L, 0, L])
plt.colorbar(label="continuous phase field")
plt.title(f"Continuous Fourier microstructure, sample {sample_id}")

plt.colorbar(label="phase: 0=matrix, 1=inclusion")
plt.title("2D composite RVE")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 4))
plt.imshow(np.asarray(T_fluct).T, origin="lower", extent=[0, L, 0, L])
plt.colorbar(label=r"$T^{\mathrm{fluct}}$")
plt.title("Periodic temperature fluctuation")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 4))
plt.imshow(np.asarray(gradT[:, :, 0]).T, origin="lower", extent=[0, L, 0, L])
plt.colorbar(label=r"$\partial T / \partial x$")
plt.title(r"Local temperature-gradient field $\nabla T_x$")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 4))
plt.imshow(np.asarray(eps[:, :, 0, 0]).T, origin="lower", extent=[0, L, 0, L])
plt.colorbar(label=r"$\varepsilon_{xx}$")
plt.title(r"Local total strain field $\varepsilon_{xx}$")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 4))
plt.imshow(np.asarray(eps_th[:, :, 0, 0]).T, origin="lower", extent=[0, L, 0, L])
plt.colorbar(label=r"$\varepsilon^{th}_{xx}$")
plt.title(r"Local thermal strain field $\varepsilon^{th}_{xx}$")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 4))
plt.imshow(np.asarray(sigma[:, :, 0, 0]).T, origin="lower", extent=[0, L, 0, L])
plt.colorbar(label=r"$\sigma_{xx}$")
plt.title(r"Local stress field $\sigma_{xx}$")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 4))
plt.semilogy(thermal_history[:, 0], label="gradient change")
plt.semilogy(thermal_history[:, 1], label="active-mode flux residual")
plt.xlabel("iteration")
plt.ylabel("value")
plt.title("Thermal convergence history")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 4))
plt.semilogy(mechanical_history[:, 0], label="strain change")
plt.semilogy(mechanical_history[:, 1], label="active-mode residual")
plt.xlabel("iteration")
plt.ylabel("value")
plt.title("Mechanical convergence history")
plt.legend()
plt.tight_layout()
plt.show()
