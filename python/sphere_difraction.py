import numpy as np
import scipy.special as sp

N = 25  # глобальная переменная, определяющая предел суммирования
iHO = 1  # порядок функции Ханкеля

def sph_hankel(n: int, z: complex) -> complex:
    return  sp.spherical_jn(n, z) + 1j * sp.spherical_yn(n, z)

def sph_bessel_derivative(n: int, z: complex) -> complex: # derivative of x * j_n(x)
    return z * sp.spherical_jn(n, z, derivative=True) + sp.spherical_jn(n, z)
    #return (n+1) * sp.spherical_jn(n, z) - z * sp.spherical_jn(n + 1, z)

def sph_hankel_derivative(n: int, z: complex) -> complex: # derivative of x * h_n(x)
    return z * (sp.spherical_jn(n, z, derivative=True) + 1j * sp.spherical_yn(n, z, derivative=True)) + sph_hankel(n, z)

def assoc_legendre_derivative(n: int, m: int, x: float) -> float:
    if np.isclose(x,1.0):
        x -= 1e-10

    return (n * x * sp.lpmv(m, n, x) - (n + m) * sp.lpmv(m, n - 1, x)) / (x ** 2 - 1)


def calculate_coefficients(k: float, eps: list[complex], r: list[float]) -> tuple[list[complex], list[complex]]:
    layers_number = len(r)
    D_e, D_m = [0], [0]
    
    T_e = [np.eye(2, dtype=np.complex128) for _ in range(N)]
    T_m = [np.eye(2, dtype=np.complex128) for _ in range(N)]
    
    for i in range(layers_number):
        for n in range(1, N):
            arg_A = k * np.sqrt(eps[i]) * r[i]
            arg_B = k * np.sqrt(eps[i + 1]) * r[i]
            
            A_m = np.array([
                [arg_A * sp.spherical_jn(n, arg_A),                 arg_A * sph_hankel(n, arg_A)],
                [np.sqrt(eps[i]) * sph_bessel_derivative(n, arg_A), np.sqrt(eps[i]) * sph_hankel_derivative(n, arg_A)]
            ], dtype=np.complex128)
            
            A_e = np.array([
                [eps[i] * A_m[0, 0], eps[i] * A_m[0, 1]],
                [         A_m[1, 0],          A_m[1, 1]]
            ], dtype=np.complex128)
            
            B_m = np.array([
                [arg_B * sp.spherical_jn(n, arg_B),                     arg_B * sph_hankel(n, arg_B)],
                [np.sqrt(eps[i + 1]) * sph_bessel_derivative(n, arg_B), np.sqrt(eps[i + 1]) * sph_hankel_derivative(n, arg_B)]
            ], dtype=np.complex128)
            
            B_e = np.array([
                [eps[i + 1] * B_m[0, 0], eps[i + 1] * B_m[0, 1]],
                [             B_m[1, 0],              B_m[1, 1]]
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

def calculate_electric_field_close(r: list[float], eps_compl: list[complex], lambda_: float) -> tuple[list[complex], list[complex]]:
    k = 2 * np.pi / lambda_
    phi = 0

    N_x = N_y = 200
    x_values = np.linspace(2 * -r[-1], 2 * r[-1], N_x)
    y_values = np.linspace(2 * -r[-1], 2 * r[-1], N_y)
    
    vE_theta = np.zeros((N_x,N_y),dtype=np.complex128)
    vE_phi = np.zeros((N_x,N_y),dtype=np.complex128)
    vE_r = np.zeros((N_x,N_y),dtype=np.complex128)
    
    vD_e, vD_m = calculate_coefficients(k, eps_compl, r)
    

    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            R = np.sqrt(x**2 + y**2)
            if R < r[-1] or x == 0:
                continue

            theta = np.arctan2(y, x)
            E_r, E_th, E_ph = 0, 0, 0
            
            cos_th = np.cos(theta)
            sin_th = np.sin(theta)
            
            for n in range(1, N):

                E_r += vD_e[n] * n * (n + 1) / R**2 * sph_hankel(n, k*R) * sp.lpmv(1, n, cos_th) * np.cos(phi)

                E_th += vD_e[n] * sph_hankel_derivative(n, k * R) * (assoc_legendre_derivative(n, 1, cos_th) * (-sin_th)) + 1j * vD_m[n] /sin_th * sph_hankel(n, k*R) * sp.lpmv(1,n,cos_th)

                E_ph += vD_e[n] / sin_th * sph_hankel_derivative(n, k*R) *sp.lpmv(1,n,cos_th) + 1j * vD_m[n] * sph_hankel(n, k * R) * (assoc_legendre_derivative(n, 1, cos_th) * (-sin_th))
            
            E_th *=  k / R * np.cos(phi)
            E_ph *= -k / R * np.sin(phi)
            
            vE_r[i,j] = E_r
            vE_theta[i,j] = E_th
            vE_phi[i,j] = E_ph
    
    # vE_theta[0] = vE_theta[1]
    # #vE_theta[-1] = vE_theta[-2]
    # vE_theta[len(vE_theta) // 2] = vE_theta[len(vE_theta) // 2 - 1]
    
    # vE_phi[0] = vE_phi[1]
    # #vE_phi[-1] = vE_phi[-2]
    # vE_phi[len(vE_phi) // 2] = vE_phi[len(vE_phi) // 2 - 1]
    
    # vE_r[0] = vE_r[1]
    # #vE_r[-1] = vE_r[-2]
    # vE_r[len(vE_r) // 2] = vE_r[len(vE_r) // 2 - 1]

    return vE_r, vE_theta, vE_phi


def calculate_electric_field_close_vectorized(r: list[float], eps_compl: list[complex], lambda_: float, limits:list[float]):
    
    def assoc_legendre_derivative_vectorized(n: int, m: int, x: float) -> float:
        x[np.isclose(x,1.0)] -= 1e-10
        
        return (n * x * sp.lpmv(m, n, x) - (n + m) * sp.lpmv(m, n - 1, x)) / (x ** 2 - 1)
    
    k = 2 * np.pi / lambda_
    phi = 0

    N_x = N_y = 300
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

    for n in range(1, N):
        Pnm = sp.lpmv(1, n, cos_th)
        dPnm = assoc_legendre_derivative_vectorized(n, 1, cos_th)

        hankel = sph_hankel(n, k * R)
        hankel_deriv = sph_hankel_derivative(n, k * R)

        term_Er = vD_e[n] * n * (n + 1) / R**2 * hankel * Pnm * np.cos(phi)
        term_Eth = (
            vD_e[n] * hankel_deriv * dPnm * (-sin_th) +
            1j * vD_m[n] / sin_th * hankel * Pnm
        )
        #term_Eph = (vD_e[n] / sin_th * hankel_deriv * Pnm +
        #            1j * vD_m[n] * hankel * dPnm * (-sin_th))

        E_r += np.where(mask, term_Er, 0)
        E_theta += np.where(mask, term_Eth, 0)
        #E_phi += np.where(mask, term_Eph, 0)

    E_theta *= k / R * np.cos(phi)
    #E_phi *= -k / R * np.sin(phi)

    
    return E_r, E_theta, E_phi