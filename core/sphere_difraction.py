import numpy as np
import scipy.special as sp
from .parameters import ExperimentParameters
import hashlib
import os
import time

def xi(n: int, z: complex) -> complex: # x * h1_n(x)
    result = z * (sp.spherical_jn(n, z) + 1j * sp.spherical_yn(n, z))
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result

def xi_derivative(n: int, z: complex) -> complex: # derivative of x * h1_n(x)
    result = z * (sp.spherical_jn(n, z, derivative=True) + 1j * sp.spherical_yn(n, z, derivative=True)) + ( sp.spherical_jn(n, z) + 1j * sp.spherical_yn(n, z))
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result

def psi(n: int, z:complex) -> complex: # x * j_n(x)
    result = z * sp.spherical_jn(n, z)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result

def psi_derivative(n: int, z: complex) -> complex: # derivative of x * j_n(x)
    result = z * sp.spherical_jn(n, z, derivative=True) + sp.spherical_jn(n, z)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result

def assoc_legendre_derivative(n: int, m: int, x: float) -> float:
    # Use the same identity as in the vectorized variant.
    # Accepts either scalar or array-like `n` and returns an array-like result.
    # Protect the x value near +-1 to avoid division by zero.
    x_safe = (x - 1e-10) if np.isclose(x, 1.0) else x
    n_arr = np.asarray(n)
    # lpmv can accept array `n` and returns values accordingly
    result = (n_arr * x_safe * sp.lpmv(m, n_arr, x_safe) - (n_arr + m) * sp.lpmv(m, n_arr - 1, x_safe)) / (x_safe ** 2 - 1.0)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result

def assoc_legendre_derivative_vectorized(n: int, m: int, x: np.ndarray) -> np.ndarray:
    x = np.where(np.isclose(x, 1.0), x - 1e-10, x)
    result = (n * x * sp.lpmv(m, n, x) - (n + m) * sp.lpmv(m, n - 1, x)) / (x ** 2 - 1.0)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result



def _calculate_R_for_conducting_center(k, eps, r, n_max = 128):
        
    arg_R = k * np.sqrt(eps[1]) * r[0]
    Orders = np.arange(n_max)

    R_m = - psi(Orders, arg_R) / xi(Orders, arg_R)
    R_e = - psi_derivative(Orders, arg_R) / xi_derivative(Orders, arg_R)

    return R_m, R_e
    
def _calculate_R_for_dielectric_center(k, eps, r, n_max = 128):

    R_m = np.zeros(n_max, dtype=np.complex128)
    R_e = np.zeros(n_max, dtype=np.complex128)
    arg_R1 = k * np.sqrt(eps[1]) * r[0]
    arg_R0 = k * np.sqrt(eps[0]) * r[0]

    if(abs(arg_R1 - arg_R0) < 1e-14):
        return R_m, R_e

    Orders = np.arange(n_max)
    
    numerator = np.sqrt(eps[1]) * psi_derivative(Orders, arg_R1) * psi(Orders, arg_R0) - np.sqrt(eps[0]) * psi(Orders, arg_R1) * psi_derivative(Orders, arg_R0)
    denominator = np.sqrt(eps[0]) * xi(Orders, arg_R1) * psi_derivative(Orders, arg_R0) - np.sqrt(eps[1]) * xi_derivative(Orders, arg_R1) * psi(Orders, arg_R0)
    R_m =  np.divide(numerator, denominator, where=denominator != 0)
    
    
    numerator = np.sqrt(eps[0]) * psi_derivative(Orders, arg_R1) * psi(Orders, arg_R0) - np.sqrt(eps[1]) * psi(Orders, arg_R1) * psi_derivative(Orders, arg_R0)
    denominator = np.sqrt(eps[1]) * xi(Orders, arg_R1) * psi_derivative(Orders, arg_R0) - np.sqrt(eps[0]) * xi_derivative(Orders, arg_R1) * psi(Orders, arg_R0)
    R_e = np.divide(numerator, denominator, where=denominator != 0)

    return R_m, R_e
    


def calculate_coefficients(
        params: ExperimentParameters,
        n_max: int = 128,
        threshold: float = 1e-50,
    ) -> tuple[np.ndarray, np.ndarray, int]:
    
    k = params.k
    eps = params.eps
    r = params.r
    conducting_core = params.conducting_core

    layers_number = len(r)
    D_e = np.zeros(n_max, dtype=np.complex128)
    D_m = np.zeros(n_max, dtype=np.complex128)

    T_e = [np.eye(2, dtype=np.complex128) for _ in range(n_max)]
    T_m = [np.eye(2, dtype=np.complex128) for _ in range(n_max)]

    Orders = np.arange(n_max)
    for i in range(1, layers_number):

        if(abs(eps[i] - eps[i + 1]) < 1e-14):
            continue

        arg_A = k * np.sqrt(eps[i]) * r[i]
        arg_B = k * np.sqrt(eps[i + 1]) * r[i]
            
        A_m = np.array([
            [                             psi(Orders, arg_A),                              xi(Orders, arg_A)],
            [np.sqrt(eps[i]) * psi_derivative(Orders, arg_A), np.sqrt(eps[i]) * xi_derivative(Orders, arg_A)]
        ], dtype=np.complex128)

        A_m = A_m.transpose(2, 0, 1)

        A_e = A_m.copy()    
        A_e[:, 0, :] *= eps[i]  # A_e is A_m with first row multiplied by eps[i]

        B_m = np.array([
            [                                 psi(Orders, arg_B),                                  xi(Orders, arg_B)],
            [np.sqrt(eps[i + 1]) * psi_derivative(Orders, arg_B), np.sqrt(eps[i + 1]) * xi_derivative(Orders, arg_B)]
        ], dtype=np.complex128)
            
        B_m = B_m.transpose(2, 0, 1)

        B_e = B_m.copy()    
        B_e[:, 0, :] *= eps[i + 1]  # B_e is B_m with first row multiplied by eps[i + 1]

        for n in range(1, n_max):
            try:
                T_m[n] = np.linalg.solve(B_m[n], A_m[n]) @ T_m[n]
                T_e[n] = np.linalg.solve(B_e[n], A_e[n]) @ T_e[n]
            except np.linalg.LinAlgError:
                # B is singular at this order — boundary condition is degenerate,
                # leave T[n] unchanged (identity propagation for this layer)
                pass
    
    if(conducting_core):
        R_m, R_e = _calculate_R_for_conducting_center(k, eps, r, n_max)
    else:
        R_m, R_e = _calculate_R_for_dielectric_center(k, eps, r, n_max)

    converged_at = n_max - 1
    for n in range(1, n_max):
        c_n = (1j ** (n - 1)) / (k ** 2) * (2 * n + 1) / (n * (n + 1))
        
        D_e[n] = c_n * (T_e[n][1, 0] + T_e[n][1, 1] * R_e[n]) / (T_e[n][0, 0] + T_e[n][0, 1] * R_e[n])
        D_m[n] = c_n * (T_m[n][1, 0] + T_m[n][1, 1] * R_m[n]) / (T_m[n][0, 0] + T_m[n][0, 1] * R_m[n])

        # if absolute value of D becomes very small -> stop loop
        if  np.abs(D_e[n]) < threshold or \
            np.abs(D_m[n]) < threshold :
            converged_at = n
            break

    return D_e, D_m, converged_at

def calculate_S(params: ExperimentParameters, M = 3600):
    """
    Compute the far-field scattering amplitude functions S_theta and S_phi
    for M evenly spaced angles over [0, 2pi].

    Angles 0 and pi are handled separately via asymptotic formulas 
    because the general term is singular there.

    Parameters
    ----------
    params : ExperimentParameters
        Sphere and wave parameters.
    M : int
        Number of angular samples. Must be positive.

    Returns
    -------
    S_th, S_ph : ndarray of shape (M,), dtype complex128
        Scattering amplitudes at angles m * 2pi/M for m in 0..M-1.
    """
    assert M > 0, 'M has to be positive integer'

    D_e, D_m, N = calculate_coefficients(params)

    n = np.arange(1, N)
    i_pow  = (-1j) ** n               # (-i)^n
    n_pair = n * (n + 1) / 2          # n(n+1)/2 
    
    # Number of unique angles to compute before mirroring.
    # The series is symmetric: S(2pi - theta) = S(theta), so we only
    # compute the first half and reflect.
    n_unique = (M + 1) // 2
    # If M is even, angle=pi lands exactly on a sample and needs special treatment.
    has_pi   = (M % 2 == 0)
    n_compute = n_unique + has_pi

    S_th = np.zeros(n_compute, dtype=np.complex128)
    S_ph = np.zeros(n_compute, dtype=np.complex128)

    # --- theta = 0 (forward scattering): use asymptotic limit ---
    S_ph[0] = np.sum(i_pow * n_pair * (D_m[1:N] + D_e[1:N]))
    S_th[0] = -S_ph[0]

    theta_step = 2 * np.pi / M
    for m in range(1, n_unique):
        theta = m * theta_step
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)

        first_terms = sp.lpmv(1, np.arange(N), cos_th)
        if (not np.isclose(sin_th, 0.0)):
            first_terms /= sin_th
        second_terms = assoc_legendre_derivative(np.arange(N), 1, cos_th) * sin_th
        
        # Replace invalid entries
        first_terms = np.where(np.isfinite(first_terms), first_terms, 0.0)
        second_terms = np.where(np.isfinite(second_terms), second_terms, 0.0)

        S_th[m] += np.sum(i_pow * (D_m[1:N] * first_terms[1:N] - D_e[1:N] * second_terms[1:N]))
        S_ph[m] += np.sum(i_pow * (D_m[1:N] * second_terms[1:N] - D_e[1:N] * first_terms[1:N]))

    # --- theta = pi (backscattering): use asymptotic limit ---
    if has_pi:
        alternating = (-1) ** n
        S_th[-1] = np.sum(i_pow * alternating * n_pair * (D_e[1:N] - D_m[1:N]))
        S_ph[-1] = S_th[-1]
        
    # --- Mirror over theta = pi to fill the second half ---
    # Exclude endpoints (0 and pi) from the reflection to avoid duplication.
    interior_th = S_th[1:-1] if has_pi else S_th[1:]
    interior_ph = S_ph[1:-1] if has_pi else S_ph[1:]
    S_th = np.concatenate([S_th, interior_th[::-1]])
    S_ph = np.concatenate([S_ph, interior_ph[::-1]])

    return S_th, S_ph


def calculate_electric_field_far(ExperimentParam: ExperimentParameters, phi = np.pi * 0) -> tuple[list[complex], list[complex]]:
    k = ExperimentParam.k
    
    S_th, S_ph = calculate_S(ExperimentParam)
    
    x = ExperimentParam.r[-1] * 10
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    E_theta = S_th * k**2 * (np.exp(1j * k * x) * cos_phi / (k * x))
    E_phi =   S_ph * k**2 * (np.exp(1j * k * x) * sin_phi / (k * x))
    
    return E_theta, E_phi


# deprecated, better use the next one
def calculate_electric_field_close(ExperimentParam: ExperimentParameters, limits:list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # arrays of complex128 elements
    k = ExperimentParam.k
    phi = 0

    N_x = N_y = 300
    x_min, x_max, y_min, y_max = limits
    x_values = np.linspace(x_min, x_max, N_x)
    y_values = np.linspace(y_min, y_max, N_y)
    
    vE_theta = np.zeros((N_x,N_y),dtype=np.complex128)
    vE_phi = np.zeros((N_x,N_y),dtype=np.complex128)
    vE_r = np.zeros((N_x,N_y),dtype=np.complex128)
    
    vD_e, vD_m, N = calculate_coefficients(ExperimentParam)

    cos_phi = np.cos(phi)

    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            R = np.sqrt(x**2 + y**2)
            if R < ExperimentParam.r[-1] or x == 0:
                continue

            theta = np.arctan2(y, x)
            E_r, E_th, E_ph = 0, 0, 0
            
            cos_th = np.cos(theta)
            sin_th = np.sin(theta)
            
            kR = k * R
            inv_R2 = 1 / R**2

            for n in range(1, N):

                E_r += vD_e[n] * n * (n + 1) * inv_R2 * xi(n, kR) * sp.lpmv(1, n, cos_th) * cos_phi

                E_th += vD_e[n] * xi_derivative(n, kR) * (assoc_legendre_derivative(n, 1, cos_th) * (-sin_th)) + 1j * vD_m[n] / sin_th * xi(n, kR) * sp.lpmv(1,n,cos_th)

                E_ph += vD_e[n] / sin_th * xi_derivative(n, kR) *sp.lpmv(1,n,cos_th) + 1j * vD_m[n] * xi(n, kR) * (assoc_legendre_derivative(n, 1, cos_th) * (-sin_th))
            
            E_th *=  k / R * np.cos(phi)
            E_ph *= -k / R * np.sin(phi)
            
            vE_r[i,j] = E_r
            vE_theta[i,j] = E_th
            vE_phi[i,j] = E_ph

    return vE_r, vE_theta, vE_phi


def calculate_electric_field_close_vectorized(ExperimentParam: ExperimentParameters, coordinates_limits:list[float])-> tuple[np.ndarray, np.ndarray, np.ndarray]:  # arrays of complex128 elements
        
    def load_or_compute_field_terms(cos_th: np.ndarray, kR: np.ndarray, N: int):

        def generate_cache_filename(cos_th: np.ndarray, kR: np.ndarray, N: int, cache_dir="field_cache"):
            os.makedirs(cache_dir, exist_ok=True)
            h = hashlib.sha256()
            h.update(cos_th.astype(np.float32).tobytes())
            h.update(kR.astype(np.float32).tobytes())
            h.update(str(N).encode())
            hash_str = h.hexdigest()
            return os.path.join(cache_dir, f"field_cache_{hash_str}.npz")
        
        filename = generate_cache_filename(cos_th, kR, N)

        if os.path.exists(filename):
            data = np.load(filename, allow_pickle=True)
            return data['Pnm'], data['dPnm'], data['hankel'], data['hankel_deriv']

        shape = kR.shape
        Pnm = np.empty((N, *shape), dtype=np.float64)
        dPnm = np.empty_like(Pnm)
        hankel = np.empty((N, *shape), dtype=np.complex128)
        hankel_deriv = np.empty_like(hankel)

        for n in range(1, N):
            Pnm[n] = sp.lpmv(1, n, cos_th)
            dPnm[n] = assoc_legendre_derivative_vectorized(n, 1, cos_th)

            jn = sp.spherical_jn(n, kR)
            yn = sp.spherical_yn(n, kR)
            jn_p = sp.spherical_jn(n, kR, derivative=True)
            yn_p = sp.spherical_yn(n, kR, derivative=True)

            hankel[n] = kR * (jn + 1j * yn)
            hankel_deriv[n] = kR * (jn_p + 1j * yn_p) + (jn + 1j * yn)


        np.savez_compressed(filename, Pnm=Pnm, dPnm=dPnm, hankel=hankel, hankel_deriv=hankel_deriv)
        return Pnm, dPnm, hankel, hankel_deriv

    start = time.time()
    k = ExperimentParam.k
    phi = 0

    cos_phi = np.cos(phi)

    N_x = N_y = 200
    x_min, x_max, y_min, y_max = coordinates_limits
    x_values = np.linspace(x_min, x_max, N_x)
    y_values = np.linspace(y_min, y_max, N_y)

    start_coefficient = time.time()
    vD_e, vD_m, N = calculate_coefficients(ExperimentParam)
    end_coefficients = time.time()

    X, Y = np.meshgrid(x_values, y_values, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    mask = (R >= ExperimentParam.r[-1]) & (X != 0)

    E_r = np.zeros_like(R, dtype=np.complex128)
    E_theta = np.zeros_like(R, dtype=np.complex128)
    E_phi = np.zeros_like(R, dtype=np.complex128)

    cos_th = np.cos(theta)
    sin_th = np.sin(theta)
    kR = k * R
    start_field_terms = time.time()
    Pnm_vals, dPnm_vals, hankel_vals, hankel_deriv_vals = load_or_compute_field_terms(cos_th, kR, N)
    end_field_terms = time.time()
    inv_R2 = np.zeros_like(R)
    inv_R2[mask] = 1 / R[mask]**2
    for n in range(1, N):

        term_Er = vD_e[n] * n * (n + 1) * inv_R2 * hankel_vals[n] * Pnm_vals[n] * cos_phi
        term_Eth = (
            vD_e[n] * hankel_deriv_vals[n] * dPnm_vals[n] * (-sin_th) +
            1j * vD_m[n] / sin_th * hankel_vals[n] * Pnm_vals[n]
        )
        #term_Eph = (vD_e[n] / sin_th * hankel_deriv * Pnm +
        #            1j * vD_m[n] * hankel * dPnm * (-sin_th))

        E_r[mask] += term_Er[mask]
        E_theta[mask] += term_Eth[mask]
        #E_phi[mask] += term_Eph[mask]

    E_theta *= k / R * cos_phi
    #E_phi *= -k / R * np.sin(phi)
    end = time.time()

    # print("coefficients tiem", end_coefficients - start_coefficient)
    # print("field terms", end_field_terms - start_field_terms)
    # print("loop", end - end_field_terms)
    # print("whole time", end-start)

    return E_r, E_theta, E_phi