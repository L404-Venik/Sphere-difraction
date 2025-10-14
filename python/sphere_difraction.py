import numpy as np
from numpy.linalg import solve
import scipy.special as sp
import hashlib
import os
import time

N_max = 50  # Global variable, defining summation limit
iHO = 1  # order of Hankel function

def xi(n: int, z: complex) -> complex: # x * h1_n(x)
    return z * ( sp.spherical_jn(n, z) + 1j * sp.spherical_yn(n, z))

def xi_derivative(n: int, z: complex) -> complex: # derivative of x * h1_n(x)
    return z * (sp.spherical_jn(n, z, derivative=True) + 1j * sp.spherical_yn(n, z, derivative=True)) + ( sp.spherical_jn(n, z) + 1j * sp.spherical_yn(n, z))

def psi(n: int, z:complex) -> complex: # x * j_n(x)
    return z * sp.spherical_jn(n, z)

def psi_derivative(n: int, z: complex) -> complex: # derivative of x * j_n(x)
    return z * sp.spherical_jn(n, z, derivative=True) + sp.spherical_jn(n, z)

def assoc_legendre_derivative(n: int, m: int, x: float) -> float:
    if np.isclose(x,1.0):
        x -= 1e-10

    return (n * x * sp.lpmv(m, n, x) - (n + m) * sp.lpmv(m, n - 1, x)) / (x ** 2 - 1)

def assoc_legendre_derivative_vectorized(n: int, m: int, x: np.ndarray) -> np.ndarray:
    x = np.where(np.isclose(x, 1.0), x - 1e-10, x)
    return (n * x * sp.lpmv(m, n, x) - (n + m) * sp.lpmv(m, n - 1, x)) / (x ** 2 - 1)

def calculate_coefficients(k: float, eps: list[complex], r: list[float], conducting_center: bool = True) -> tuple[list[complex], list[complex]]:
    
    def calculate_R_for_conducting_center(k, eps, r):
        
        arg_R = k * np.sqrt(eps[1]) * r[0]
        Orders = np.arange(N_max)

        R_m = - psi(Orders, arg_R) / xi(Orders, arg_R)
        R_e = - psi_derivative(Orders, arg_R) / xi_derivative(Orders, arg_R)

        return R_m, R_e
    
    def calculate_R_for_dielectric_center(k, eps, r):
        
        R_m = np.zeros(N_max, dtype=np.complex128)
        R_e = np.zeros(N_max, dtype=np.complex128)
        arg_R1 = k * np.sqrt(eps[1]) * r[0]
        arg_R0 = k * np.sqrt(eps[0]) * r[0]

        if(abs(arg_R1 - arg_R0) < 1e-14):
            return R_m, R_e

        Orders = np.arange(N_max)
        
        R_m = (np.sqrt(eps[1]) * psi_derivative(Orders, arg_R1) * psi(Orders, arg_R0) - np.sqrt(eps[0]) * psi(Orders, arg_R1) * psi_derivative(Orders, arg_R0)) / \
                    (np.sqrt(eps[0]) * xi(Orders, arg_R1) * psi_derivative(Orders, arg_R0) - np.sqrt(eps[1]) * xi_derivative(Orders, arg_R1) * psi(Orders, arg_R0))
        
        R_e = (np.sqrt(eps[0]) * psi_derivative(Orders, arg_R1) * psi(Orders, arg_R0) - np.sqrt(eps[1]) * psi(Orders, arg_R1) * psi_derivative(Orders, arg_R0)) / \
                    (np.sqrt(eps[1]) * xi(Orders, arg_R1) * psi_derivative(Orders, arg_R0) - np.sqrt(eps[0]) * xi_derivative(Orders, arg_R1) * psi(Orders, arg_R0))

        return R_m, R_e
    
    layers_number = len(r)
    D_e = np.zeros(N_max, dtype=np.complex128)
    D_m = np.zeros(N_max, dtype=np.complex128)
    
    T_e = [np.eye(2, dtype=np.complex128) for _ in range(N_max)]
    T_m = [np.eye(2, dtype=np.complex128) for _ in range(N_max)]
    
    Orders = np.arange(N_max)
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

        for n in range(1, N_max):
            T_m[n] @= np.linalg.inv(B_m[n]) @ A_m[n]
            T_e[n] @= np.linalg.inv(B_e[n]) @ A_e[n]
    
    if(conducting_center):
        R_m, R_e = calculate_R_for_conducting_center(k, eps, r)
    else:
        R_m, R_e = calculate_R_for_dielectric_center(k, eps, r)

    for n in range(1, N_max):
        c_n = (1j ** (n - 1)) / (k ** 2) * (2 * n + 1) / (n * (n + 1))
        
        D_e[n] = c_n * (T_e[n][1, 0] + T_e[n][1, 1] * R_e[n]) / (T_e[n][0, 0] + T_e[n][0, 1] * R_e[n])
        D_m[n] = c_n * (T_m[n][1, 0] + T_m[n][1, 1] * R_m[n]) / (T_m[n][0, 0] + T_m[n][0, 1] * R_m[n])

        if(np.abs(D_e[n]) < 1e-40 and np.abs(D_m[n]) < 1e-40): # stop when values below threshold
            break
    
    return n, D_e, D_m


def calculate_S(N, D_e, D_m):

    M = 1000
    S_th = np.zeros(M, dtype=np.complex128)
    S_ph = np.zeros(M, dtype=np.complex128)

    i_pow = (-1j) ** np.arange(1, N) # Vector of (-1j)**n
    for m in range(M):
        theta = 0.0001 + m / M * 2 * np.pi

        cos_th = np.cos(theta)
        sin_th = np.sin(theta)

        first_terms = sp.lpmv(1, np.arange(N), cos_th) / sin_th
        second_terms = assoc_legendre_derivative(np.arange(N), 1, cos_th) * sin_th

        # Replace invalid entries
        first_terms = np.where(np.isfinite(first_terms), first_terms, 0.0)
        second_terms = np.where(np.isfinite(second_terms), second_terms, 0.0)

        S_th[m] += np.sum(i_pow * (D_m[1:N] * first_terms[1:N] - D_e[1:N] * second_terms[1:N]))
        S_ph[m] += np.sum(i_pow * (D_m[1:N] * second_terms[1:N] - D_e[1:N] * first_terms[1:N]))

    return S_th, S_ph


def calculate_electric_field_far(r: list[float], eps_compl: list[complex], lambda_: float, conducting_center: bool = True) -> tuple[list[complex], list[complex]]:
    k = 2 * np.pi / lambda_
    
    N, D_e, D_m = calculate_coefficients(k, eps_compl, r, conducting_center)
    S_th, S_ph = calculate_S(N, D_e, D_m)
    
    x = r[-1] * 10
    phi = 0
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    E_theta = S_th * k**2 * (np.exp(1j * k * x) * cos_phi / (k * x))
    E_phi =   S_ph * k**2 * (np.exp(1j * k * x) * sin_phi / (k * x))
    
    E_theta[0] = E_theta[-1]
    E_phi[0] = E_phi[-1]
    
    return E_theta, E_phi



def calculate_electric_field_close_radial(r: list[float], eps_compl: list[complex], lambda_: float, R: float) -> tuple[list[complex], list[complex], list[complex]]:
    k = 2 * np.pi / lambda_
    phi = 0
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    N, vD_e, vD_m = calculate_coefficients(k, eps_compl, r)
    
    M = 1000
    E_r = np.zeros(M, dtype=np.complex128)
    E_theta = np.zeros(M, dtype=np.complex128)
    E_phi = np.zeros(M, dtype=np.complex128)

    for m in range(M):
        theta = 0.0001 + 2 * np.pi * m / M
        
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)
        
        for n in range(1, N):

            E_r[m] += vD_e[n] * n * (n + 1) / R**2 * xi(n, k*R) * sp.lpmv(1, n, cos_th) * cos_phi

            E_theta[m] += vD_e[n] * xi_derivative(n, k * R) * (assoc_legendre_derivative(n, 1, cos_th) * (-sin_th)) + 1j * vD_m[n] /sin_th * xi(n, k*R) * sp.lpmv(1,n,cos_th)

            E_phi[m] += vD_e[n] / sin_th * xi_derivative(n, k*R) *sp.lpmv(1,n,cos_th) + 1j * vD_m[n] * xi(n, k * R) * (assoc_legendre_derivative(n, 1, cos_th) * (-sin_th))
        
    
    E_theta *=  k / R * cos_phi
    E_phi *= -k / R * sin_phi
    
    E_theta[0] = E_theta[-1]
    E_phi[0] = E_phi[-1]
    E_r[0] = E_r[-1]

    return E_r, E_theta, E_phi


def calculate_electric_field_close(r: list[float], eps_compl: list[complex], lambda_: float, limits:list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # arrays of complex128 elements
    k = 2 * np.pi / lambda_
    phi = 0

    N_x = N_y = 300
    x_min, x_max, y_min, y_max = limits
    x_values = np.linspace(x_min, x_max, N_x)
    y_values = np.linspace(y_min, y_max, N_y)
    
    vE_theta = np.zeros((N_x,N_y),dtype=np.complex128)
    vE_phi = np.zeros((N_x,N_y),dtype=np.complex128)
    vE_r = np.zeros((N_x,N_y),dtype=np.complex128)
    
    N, vD_e, vD_m = calculate_coefficients(k, eps_compl, r)

    cos_phi = np.cos(phi)

    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            R = np.sqrt(x**2 + y**2)
            if R < r[-1] or x == 0:
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



def calculate_electric_field_close_vectorized(r: list[float], eps_compl: list[complex], lambda_: float, limits:list[float], conducting_center: bool = True)-> tuple[np.ndarray, np.ndarray, np.ndarray]:  # arrays of complex128 elements
        
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
    k = 2 * np.pi / lambda_
    phi = 0

    cos_phi = np.cos(phi)

    N_x = N_y = 200
    x_min, x_max, y_min, y_max = limits
    x_values = np.linspace(x_min, x_max, N_x)
    y_values = np.linspace(y_min, y_max, N_y)

    start_coefficient = time.time()
    N, vD_e, vD_m = calculate_coefficients(k, eps_compl, r, conducting_center)
    end_coefficients = time.time()

    X, Y = np.meshgrid(x_values, y_values, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    mask = (R >= r[-1]) & (X != 0)

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