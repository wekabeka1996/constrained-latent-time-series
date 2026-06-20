import os
import numpy as np

# === КОНФІГУРАЦІЯ ===
ARCHITECTURE_IDS = ['baseline', 'ID_1', 'ID_2', 'ID_3']
N_TRAJ = 5
# Для baseline — особливий шлях
BASELINE_FISHER_DIR = 'results/generation_1/fisher_metrics'
ARCH_ROOT = 'results/generation_2/robustness/arch'

for arch_id in ARCHITECTURE_IDS:
    for traj in range(1, N_TRAJ+1):
        if arch_id == 'baseline':
            fisher_path = os.path.join(BASELINE_FISHER_DIR, f'fisher_metric_{traj}.npy')
            out_dir = os.path.join(ARCH_ROOT, 'baseline')
        else:
            fisher_path = os.path.join(ARCH_ROOT, arch_id, f'fisher_metric_traj{traj}.npy')
            out_dir = os.path.join(ARCH_ROOT, arch_id)
        if not os.path.exists(fisher_path):
            continue
        g = np.load(fisher_path)  # (n_points, d_z, d_z)
        n_points, d_z, _ = g.shape
        U_all = np.zeros((n_points, d_z, d_z))
        for k in range(n_points):
            U, S, Vt = np.linalg.svd(g[k])
            U_all[k] = U
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f'U_traj{traj}.npy'), U_all)
        print(f'[+] Saved U_traj{traj}.npy for {arch_id}')

for arch_id in ['ID_1', 'ID_2', 'ID_3']:
    for traj in range(1, N_TRAJ+1):
        fisher_path = os.path.join(ARCH_ROOT, arch_id, f'fisher_metrics_traj{traj}.npy')
        out_dir = os.path.join(ARCH_ROOT, arch_id)
        if not os.path.exists(fisher_path):
            continue
        g = np.load(fisher_path)  # (n_points, d_z, d_z)
        n_points, d_z, _ = g.shape
        U_all = np.zeros((n_points, d_z, d_z))
        for k in range(n_points):
            U, S, Vt = np.linalg.svd(g[k])
            U_all[k] = U
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f'U_traj{traj}.npy'), U_all)
        print(f'[+] Saved U_traj{traj}.npy for {arch_id}')
