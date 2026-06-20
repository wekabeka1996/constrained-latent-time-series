import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Конфігурація ---
TRAJ_FILES = [
    'results/generation_1/analysis/interpolation_detailed_1.csv',
    'results/generation_1/analysis/interpolation_detailed_2.csv',
]

# --- Функція для обчислення коренів ---
def compute_roots(phi, theta):
    # AR: 1 - phi1*z - phi2*z^2 = 0
    ar_coeffs = np.array([1, -phi[0], -phi[1]])
    ar_roots = np.roots(ar_coeffs)
    # MA: 1 + theta1*z + theta2*z^2 + theta3*z^3 = 0
    ma_coeffs = np.array([1, theta[0], theta[1], theta[2]])
    ma_roots = np.roots(ma_coeffs)
    return ar_roots, ma_roots

for traj_idx, csv_path in enumerate(TRAJ_FILES, 1):
    df = pd.read_csv(csv_path)
    # Знаходимо перший перехід is_param_valid: True -> False
    valid = df['is_param_valid'].values
    if not np.any(~valid):
        print(f'Trajectory {traj_idx}: All points valid')
        continue
    first_invalid = np.argmax(~valid)
    # λ до і після втрати валідності
    idx_before = max(first_invalid-1, 0)
    idx_after = first_invalid
    for label, idx in [('before', idx_before), ('after', idx_after)]:
        phi = [df['phi1'].iloc[idx], df['phi2'].iloc[idx]]
        theta = [df['theta1'].iloc[idx], df['theta2'].iloc[idx], df['theta3'].iloc[idx]]
        ar_roots, ma_roots = compute_roots(phi, theta)
        plt.figure(figsize=(5,5))
        unit_circle = plt.Circle((0,0), 1, color='gray', fill=False, linestyle='--')
        ax = plt.gca()
        ax.add_artist(unit_circle)
        plt.scatter(ar_roots.real, ar_roots.imag, c='red', label='AR roots', s=80)
        plt.scatter(ma_roots.real, ma_roots.imag, c='blue', label='MA roots', s=80)
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.title(f'Traj {traj_idx} λ={df["lambda"].iloc[idx]:.2f} ({label})')
        plt.legend()
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'results/generation_1/analysis/roots_traj{traj_idx}_{label}.png', dpi=200, bbox_inches='tight')
        plt.close()
    print(f'Trajectory {traj_idx}: roots visualized for λ={df["lambda"].iloc[idx_before]:.2f} (before), λ={df["lambda"].iloc[idx_after]:.2f} (after)')
