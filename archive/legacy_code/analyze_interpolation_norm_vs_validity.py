import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Конфігурація ---
TRAJ_FILES = [
    'results/generation_1/analysis/interpolation_detailed_1.csv',
    'results/generation_1/analysis/interpolation_detailed_2.csv',
    'results/generation_1/analysis/interpolation_detailed_3.csv',
    'results/generation_1/analysis/interpolation_detailed_4.csv',
    'results/generation_1/analysis/interpolation_detailed_5.csv',
]
Z_FILES = [
    'data/generated/z_interp_1.npy',
    'data/generated/z_interp_2.npy',
    'data/generated/z_interp_3.npy',
    'data/generated/z_interp_4.npy',
    'data/generated/z_interp_5.npy',
]

plt.figure(figsize=(12, 8))
for i, (csv_path, z_path) in enumerate(zip(TRAJ_FILES, Z_FILES), 1):
    df = pd.read_csv(csv_path)
    Z = np.load(z_path)
    lambdas = df['lambda'].values
    norms = np.sum(Z**2, axis=1)
    validity = df['is_param_valid'].astype(int).values
    ax = plt.subplot(2, 3, i)
    ax.plot(lambdas, norms, 'b-', label=r'$||z_{interp}(\lambda)||^2$')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$||z||^2$', color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax2 = ax.twinx()
    ax2.plot(lambdas, validity, 'r--', label='validity', alpha=0.7)
    ax2.set_ylabel('is_param_valid', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.set_title(f'Trajectory {i}')
    ax.set_ylim(bottom=0)
    ax2.set_ylim(-0.1, 1.1)
    if i == 1:
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
plt.tight_layout()
plt.suptitle('Norm of Latent Vector vs Model Validity along Interpolation', y=1.02)
plt.savefig('results/generation_1/analysis/norm_vs_validity_by_trajectory.png', dpi=200, bbox_inches='tight')
plt.close()
print('Візуалізацію norm_vs_validity_by_trajectory.png збережено.')
