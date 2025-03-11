import numpy as np
import scipy.special as sp

N = 25  # глобальная переменная, определяющая предел суммирования
iHO = 1  # порядок функции Ханкеля

def sph_hankel(n: int, z: complex) -> complex:
    return  sp.spherical_jn(n, z) + 1j * sp.spherical_yn(n, z)

def sph_bessel_derivative(n: int, z: complex) -> complex:
    return z * sp.spherical_jn(n, z, derivative=True) + sp.spherical_jn(n, z)
    #return (n+1) * sp.spherical_jn(n, z) - z * sp.spherical_jn(n + 1, z)

def sph_hankel_derivative(n: int, z: complex) -> complex:
    return z * (sp.spherical_jn(n, z, derivative=True) + 1j * sp.spherical_yn(n, z, derivative=True)) + sph_hankel(n, z)

def assoc_legendre_derivative(n: int, m: int, x: float) -> float:
    if np.isclose(x,1.0):
        x - 0e-10
    return (n * x * sp.lpmv(m, n, x) - (n + m) * sp.lpmv(m, n - 1, x)) / (x ** 2 - 1)


def calculate_coefficients(k: float, eps: list[complex], r: list[float]) -> tuple[list[complex], list[complex]]:
    layers_number = len(r)
    D_e, D_m = [0], [0]
    
    T_e = [np.eye(2, dtype=np.complex128) for _ in range(N)]
    T_m = [np.eye(2, dtype=np.complex128) for _ in range(N)]
    
    for j in range(layers_number):
        for n in range(1, N):
            arg_A = k * np.sqrt(eps[j]) * r[j]
            arg_B = k * np.sqrt(eps[j + 1]) * r[j]
            
            A_m = np.array([
                [arg_A * sp.spherical_jn(n, arg_A),                 arg_A * sph_hankel(n, arg_A)],
                [np.sqrt(eps[j]) * sph_bessel_derivative(n, arg_A), np.sqrt(eps[j]) * sph_hankel_derivative(n, arg_A)]
            ], dtype=np.complex128)
            
            A_e = np.array([
                [eps[j] * A_m[0, 0], eps[j] * A_m[0, 1]],
                [A_m[1, 0],          A_m[1, 1]]
            ], dtype=np.complex128)
            
            B_m = np.array([
                [arg_B * sp.spherical_jn(n, arg_B),                     arg_B * sph_hankel(n, arg_B)],
                [np.sqrt(eps[j + 1]) * sph_bessel_derivative(n, arg_B), np.sqrt(eps[j + 1]) * sph_hankel_derivative(n, arg_B)]
            ], dtype=np.complex128)
            
            B_e = np.array([
                [eps[j + 1] * B_m[0, 0], eps[j + 1] * B_m[0, 1]],
                [B_m[1, 0],              B_m[1, 1]]
            ], dtype=np.complex128)
            
            T_m[n] @= np.linalg.inv(B_m) @ A_m
            T_e[n] @= np.linalg.inv(B_e) @ A_e
    
    arg_R = k * np.sqrt(eps[1]) * r[0]
    for n in range(1, N):
        c_n = (1j ** (n - 1)) / (k ** 2) * (2 * n + 1) / (n * (n + 1))
        
        R_m = -sp.spherical_jn(n, arg_R) / sph_hankel(n, arg_R)
        R_e = -sph_bessel_derivative(n, arg_R) / sph_hankel_derivative(n, arg_R)
        
        d_e_n = c_n * (T_e[n][1, 0] + T_e[n][1, 1] * R_e) / (T_e[n][0, 0] + T_e[n][0, 1] * R_e)
        d_m_n = c_n * (T_m[n][1, 0] + T_m[n][1, 1] * R_m) / (T_m[n][0, 0] + T_m[n][0, 1] * R_m)
        
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
    
    for theta in np.arange(0, 2 * np.pi, np.pi / (5 * 36.0)):
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
