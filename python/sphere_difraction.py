import numpy as np
import scipy.special as sp
import hashlib
import os
#import time

N = 40  # Global variable, defining summation limit
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
        R_m = np.empty(N, dtype=np.complex128)
        R_e = np.empty(N, dtype=np.complex128)
        arg_R = k * np.sqrt(eps[1]) * r[0]
        for n in range(1, N):
            R_m[n] = - psi(n, arg_R) / xi(n, arg_R)
            R_e[n] = - psi_derivative(n, arg_R) / xi_derivative(n, arg_R)

        return R_m, R_e
    
    def calculate_R_for_dielectric_center(k, eps, r):
        R_m = np.empty(N, dtype=np.complex128)
        R_e = np.empty(N, dtype=np.complex128)
        arg_R1 = k * np.sqrt(eps[1]) * r[0]
        arg_R0 = k * np.sqrt(eps[0]) * r[0]
        for n in range(1, N):

            R_m[n] = (np.sqrt(eps[1]) * psi_derivative(n, arg_R1) * psi(n, arg_R0) - np.sqrt(eps[0]) * psi(n, arg_R1) * psi_derivative(n, arg_R0)) / \
                     (np.sqrt(eps[0]) * xi(n, arg_R1) * psi_derivative(n, arg_R0) - np.sqrt(eps[1]) * xi_derivative(n, arg_R1) * psi(n, arg_R0))
            
            R_e[n] = (np.sqrt(eps[0]) * psi_derivative(n, arg_R1) * psi(n, arg_R0) - np.sqrt(eps[1]) * psi(n, arg_R1) * psi_derivative(n, arg_R0)) / \
                     (np.sqrt(eps[1]) * xi(n, arg_R1) * psi_derivative(n, arg_R0) - np.sqrt(eps[0]) * xi_derivative(n, arg_R1) * psi(n, arg_R0))

        return R_m, R_e
    
    layers_number = len(r)
    D_e, D_m = [0], [0]
    
    T_e = [np.eye(2, dtype=np.complex128) for _ in range(N)]
    T_m = [np.eye(2, dtype=np.complex128) for _ in range(N)]
    
    for i in range(layers_number):
        for n in range(1, N):
            arg_A = k * np.sqrt(eps[i]) * r[i]
            arg_B = k * np.sqrt(eps[i + 1]) * r[i]
            
            A_m = np.array([
                [                              psi(n,arg_A),                              xi(n, arg_A)],
                [np.sqrt(eps[i]) * psi_derivative(n, arg_A), np.sqrt(eps[i]) * xi_derivative(n, arg_A)]
            ], dtype=np.complex128)
            
            A_e = np.array([
                [eps[i] * A_m[0, 0], eps[i] * A_m[0, 1]],
                [         A_m[1, 0],          A_m[1, 1]]
            ], dtype=np.complex128)
            
            B_m = np.array([
                [                                 psi(n, arg_B),                                  xi(n, arg_B)],
                [np.sqrt(eps[i + 1]) * psi_derivative(n, arg_B), np.sqrt(eps[i + 1]) * xi_derivative(n, arg_B)]
            ], dtype=np.complex128)
            
            B_e = np.array([
                [eps[i + 1] * B_m[0, 0], eps[i + 1] * B_m[0, 1]],
                [             B_m[1, 0],              B_m[1, 1]]
            ], dtype=np.complex128)
            
            T_m[n] @= np.linalg.inv(B_m) @ A_m
            T_e[n] @= np.linalg.inv(B_e) @ A_e
    
    if(conducting_center):
        R_m, R_e = calculate_R_for_conducting_center(k, eps, r)
    else:
        R_m, R_e = calculate_R_for_dielectric_center(k, eps, r)

    for n in range(1, N):
        c_n = (1j ** (n - 1)) / (k ** 2) * (2 * n + 1) / (n * (n + 1))
        
        d_e_n = c_n * (T_e[n][1, 0] + T_e[n][1, 1] * R_e[n]) / (T_e[n][0, 0] + T_e[n][0, 1] * R_e[n])
        d_m_n = c_n * (T_m[n][1, 0] + T_m[n][1, 1] * R_m[n]) / (T_m[n][0, 0] + T_m[n][0, 1] * R_m[n])
        
        D_e.append(d_e_n)
        D_m.append(d_m_n)
    
    return D_e, D_m


def calculate_electric_field_far(r: list[float], eps_compl: list[complex], lambda_: float) -> tuple[list[complex], list[complex]]:
    k = 2 * np.pi / lambda_
    x = r[-1] * 10
    phi = 0
    
    vE_theta = []
    vE_phi = []
    
    vD_e, vD_m = calculate_coefficients(k, eps_compl, r)
    
    for theta in np.arange(0.0001, 2 * np.pi, np.pi / (5 * 36.0)):
        E_th, E_ph = 0, 0
        S_th, S_ph = 0, 0
        
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)
        
        i_pow = -1j
        
        for n in range(1, N):
            first_term = sp.lpmv(1, n, cos_th) / sin_th
            second_term = assoc_legendre_derivative(n, 1, cos_th) * sin_th
            
            if not np.isfinite(first_term):
                first_term = 0.0
            if not np.isfinite(second_term):
                second_term = 0.0
            
            S_th += i_pow * (vD_m[n] * first_term - vD_e[n] * second_term)
            S_ph += i_pow * (vD_m[n] * second_term - vD_e[n] * first_term)
            
            i_pow *= -1j
        
        S_th *= k * k
        S_ph *= k * k
        
        E_th = S_th * (np.exp(1j * k * x) * np.cos(phi) / (k * x))
        E_ph = S_ph * (np.exp(1j * k * x) * np.sin(phi) / (k * x))
        
        vE_theta.append(E_th)
        vE_phi.append(E_ph)
    
    vE_theta[0] = vE_theta[1]
    #vE_theta[-1] = vE_theta[-2]
    vE_theta[len(vE_theta) // 2] = vE_theta[len(vE_theta) // 2 - 1]
    
    vE_phi[0] = vE_phi[1]
    #vE_phi[-1] = vE_phi[-2]
    vE_phi[len(vE_phi) // 2] = vE_phi[len(vE_phi) // 2 - 1]
    
    return vE_theta, vE_phi



def calculate_electric_field_close_radial(r: list[float], eps_compl: list[complex], lambda_: float, R: float) -> tuple[list[complex], list[complex], list[complex]]:
    k = 2 * np.pi / lambda_
    phi = 0
    
    vE_theta = []
    vE_phi = []
    vE_r = []
    
    vD_e, vD_m = calculate_coefficients(k, eps_compl, r)
    

    for theta in np.arange(0.001, 2 * np.pi, np.pi / (5 * 36.0)):
        E_r, E_th, E_ph = 0, 0, 0
        
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)
        
        for n in range(1, N):

            E_r += vD_e[n] * n * (n + 1) / R**2 * xi(n, k*R) * sp.lpmv(1, n, cos_th) * np.cos(phi)

            E_th += vD_e[n] * xi_derivative(n, k * R) * (assoc_legendre_derivative(n, 1, cos_th) * (-sin_th)) + 1j * vD_m[n] /sin_th * xi(n, k*R) * sp.lpmv(1,n,cos_th)

            E_ph += vD_e[n] / sin_th * xi_derivative(n, k*R) *sp.lpmv(1,n,cos_th) + 1j * vD_m[n] * xi(n, k * R) * (assoc_legendre_derivative(n, 1, cos_th) * (-sin_th))
        
        E_th *=  k / R * np.cos(phi)
        E_ph *= -k / R * np.sin(phi)
        
        vE_r.append(E_r)
        vE_theta.append(E_th)
        vE_phi.append(E_ph)
    
    vE_theta[0] = vE_theta[1]
    # #vE_theta[-1] = vE_theta[-2]
    vE_theta[len(vE_theta) // 2] = vE_theta[len(vE_theta) // 2 - 1]
    
    vE_phi[0] = vE_phi[1]
    # #vE_phi[-1] = vE_phi[-2]
    vE_phi[len(vE_phi) // 2] = vE_phi[len(vE_phi) // 2 - 1]
    
    vE_r[0] = vE_r[1]
    # #vE_r[-1] = vE_r[-2]
    vE_r[len(vE_r) // 2] = vE_r[len(vE_r) // 2 - 1]

    return vE_r, vE_theta, vE_phi


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
    
    vD_e, vD_m = calculate_coefficients(k, eps_compl, r)

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



def calculate_electric_field_close_vectorized(r: list[float], eps_compl: list[complex], lambda_: float, limits:list[float])-> tuple[np.ndarray, np.ndarray, np.ndarray]:  # arrays of complex128 elements
        
    def generate_cache_filename(cos_th: np.ndarray, kR: np.ndarray, N: int, cache_dir="field_cache"):
        os.makedirs(cache_dir, exist_ok=True)
        h = hashlib.sha256()
        h.update(cos_th.astype(np.float32).tobytes())
        h.update(kR.astype(np.float32).tobytes())
        h.update(str(N).encode())
        hash_str = h.hexdigest()
        return os.path.join(cache_dir, f"field_cache_{hash_str}.npz")

    def load_or_compute_field_terms(cos_th: np.ndarray, kR: np.ndarray, N: int):
        filename = generate_cache_filename(cos_th, kR, N)

        if os.path.exists(filename):
            data = np.load(filename, allow_pickle=True)
            #print(f"[cache] Loaded field terms from {filename}")
            return data['Pnm'], data['dPnm'], data['hankel'], data['hankel_deriv']

        #print(f"[compute] Generating field terms and saving to {filename}")
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


    k = 2 * np.pi / lambda_
    phi = 0

    cos_phi = np.cos(phi)

    N_x = N_y = 200
    x_min, x_max, y_min, y_max = limits
    x_values = np.linspace(x_min, x_max, N_x)
    y_values = np.linspace(y_min, y_max, N_y)
    
    vD_e, vD_m = calculate_coefficients(k, eps_compl, r)

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
    Pnm_vals, dPnm_vals, hankel_vals, hankel_deriv_vals = load_or_compute_field_terms(cos_th, kR, N)

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
    

    return E_r, E_theta, E_phi