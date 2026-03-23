import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
c = const.speed_of_light

def plot_field_scaterring(S_th, S_ph = None):
    num_points = len(S_th)

    if S_ph is None:
        y_values = [abs(c) for c in S_th]
    else:
        y_values = [abs(c) for c in S_ph[:num_points // 2]] + [abs(c) for c in S_th[num_points // 2:]]
        
    y_values = np.array(y_values)

    theta = np.linspace(0, 2 * np.pi, num_points)

    plt.figure(figsize=(6, 6), dpi=200)

    ax = plt.subplot(111, projection='polar')
    ax.plot(theta, y_values, linestyle='-', linewidth=0.5)
    ax.set_yticklabels([])

    ax.set_title('Диаграмма рассеяния')

    plt.show()

def plot_field_distribution(E, limits, r, eps, lmbd, conducting_center = True):
    y_min, y_max, x_min, x_max = limits

    plt.figure(figsize=(7, 6))
    plt.imshow(E/ (1 + E), origin='lower', aspect='auto', cmap="viridis",extent=[y_min, y_max, x_min, x_max])
    plt.colorbar(label='Модуль E')
    if(conducting_center):
        plt.title("Распределение рассеянного электрического поля на проводящем шаре, покрытом слоями диэлектрика")
    else:
        plt.title("Распределение рассеянного электрического поля на шаре, состоящем из слоёв диэлектриков")
    plt.xlabel('Z')
    plt.ylabel('Y')
    
    info_text = f"λ = {lmbd:.5f}\n"
    for i, (radius, epsilon) in enumerate(zip(r, eps), start=1):
        if i == 1:
            continue
        info_text += f"Слой {i-1}: r = {radius:.5f}, ε = {epsilon}\n"

    plt.text(0.05, 0.12, info_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def plot_radar_cross_section(S, k, y_params = [-30, 30, 10], save_path = ""):
    plt.rcParams.update({'font.size': 14})

    y_values = np.array([abs(c)**2 for c in S][:]) * 4.0 * np.pi * k**2
    num_points = len(y_values) //2

    Right_bound = 180
    theta = np.linspace(0, Right_bound, num_points)
    x_ticks = np.linspace(0, Right_bound, 7)
    y_min, y_max, step = y_params[0], y_params[1], y_params[2]
    y_ticks = np.arange(y_min, y_max + 10, step=step)

    plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)
    ax.grid()
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_ylim(y_min, y_max)

    RCS = 10 * np.log10(y_values[:num_points][::-1])
    ax.plot(theta, RCS, linestyle='-')

    #ax.set_title('RCS')
    ax.set_xbound(0, Right_bound)
    ax.set_ylabel('dBm^2', fontsize='x-large')
    ax.set_xlabel('Theta (deg)', fontsize='x-large')

    if(save_path == ""):
        plt.show()
    else:
        plt.savefig(save_path)

def plot_radar_cross_sections(S_theta, S_phi, k, y_params = [-30, 30, 10], save_path = ""):
    plt.rcParams.update({'font.size': 14})

    Y1_values = np.array([abs(c)**2 for c in S_theta][:]) * 4.0 * np.pi * k**2
    Y2_values = np.array([abs(c)**2 for c in S_phi][:]) * 4.0 * np.pi * k**2
    num_points = len(Y1_values) //2

    Right_bound = 180
    theta = np.linspace(0, Right_bound, num_points)
    x_ticks = np.linspace(0, Right_bound, 7)
    y_min, y_max, step = y_params[0], y_params[1], y_params[2]
    y_ticks = np.arange(y_min, y_max + 10, step=step)

    plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)
    ax.grid()
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_ylim(y_min, y_max)

    ax.plot(theta, 10 * np.log10(Y1_values[:num_points][::-1]), linestyle='-', label='$S_{\\theta}$')
    ax.plot(theta, 10 * np.log10(Y2_values[:num_points][::-1]), linestyle='-', label='$S_{\\phi}$')

    #ax.set_title('RCS')
    ax.legend()
    ax.set_xbound(0, Right_bound)
    ax.set_ylabel('dBm^2', fontsize='x-large')
    ax.set_xlabel('Theta (deg)', fontsize='x-large')

    if(save_path == ""):
        plt.show()
    else:
        plt.savefig(save_path)

def plot_back_scattering(S, wave_lengts, scale = "linear", sphere_radius = None, save_path = None, info_text = None):
    k = 2 * np.pi / wave_lengts
    frequencies = c / wave_lengts / 1e9

    plt.figure(figsize=(18, 6))
    ax = plt.subplot(111)
    ax.grid()

    if S.ndim == 1:
        y_values = np.abs(S)**2 * 4.0 * np.pi * k**2
        #y_values = 10 * np.log10(y_values)
        x = np.arange(len(y_values))
        ax.plot(x, y_values, linestyle='-')

    elif S.ndim == 2:
        x = np.arange(S.shape[1])
        for i in range(S.shape[0]):
            y_values = np.abs(S[i])**2 * 4.0 * np.pi * k**2
            ax.plot(x, y_values, linestyle='-')

    # choose evenly spaced tick positions
    n_ticks = 10
    tick_pos = np.linspace(0, len(x)-1, n_ticks, dtype=int)
    tick_labels = np.round(frequencies[tick_pos], 3)
    
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)
    ax.set_xbound(0, x[-1])

    
    # y_min, y_max, step = -10, 30, 10
    # y_ticks = np.arange(y_min, y_max + step, step=step)
    # ax.set_yticks(y_ticks)
    # ax.set_ylim(y_min, y_max)

    # add line acording to sphere radius
    if sphere_radius != None:
        freq_target = c / sphere_radius / 1e9 
        idx = np.argmin(np.abs(frequencies - freq_target))
        ax.axvline(x=idx, linestyle='--', color='red', linewidth=2)

    if info_text != None:
        plt.text(0.02, 0.95, info_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    #ax.set_title(f'ЭПР')
    ax.set_ylabel('RCS (m^2)', fontsize='x-large')
    ax.set_yscale(scale)
    ax.set_xlabel('Frequencies (GHz)', fontsize='x-large')

    plt.show()