import os
import numpy as np
import pandas as pd
from typing import List, Dict

# === КОНФІГУРАЦІЯ ===
ARCHITECTURE_IDS = ['baseline', 'ID_1', 'ID_2', 'ID_3']
N_TRAJ = 5
ARCH_ROOT = 'results/generation_2/robustness/arch'
SUMMARY_CSV = 'results/generation_2/robustness/architecture_geometric_summary.csv'
ALIGN_CSV = 'results/generation_2/robustness/architecture_alignment_vs_baseline.csv'

# === ДОПОМІЖНІ ФУНКЦІЇ ===
def load_singular_values(arch_id: str, traj: int) -> np.ndarray:
    s_path = os.path.join(ARCH_ROOT, arch_id, f'S_traj{traj}.npy')
    if not os.path.exists(s_path):
        s_path = os.path.join(ARCH_ROOT, arch_id, 'S.npy')  # fallback
    return np.load(s_path)

def load_geodesic_lengths(arch_id: str) -> List[float]:
    path = os.path.join(ARCH_ROOT, arch_id, 'geodesic_lengths.npy')
    if os.path.exists(path):
        return np.load(path).tolist()
    return []

def load_baseline_singular_values(traj: int) -> np.ndarray:
    """Завантажити спектри S для baseline з generation_1, якщо потрібно — обчислити SVD по fisher_metric_*.npy."""
    import numpy as np
    fisher_path = os.path.join('results/generation_1/fisher_metrics', f'fisher_metric_{traj}.npy')
    if not os.path.exists(fisher_path):
        raise FileNotFoundError(fisher_path)
    g = np.load(fisher_path)  # (n_points, d_z, d_z)
    s_list = []
    for k in range(g.shape[0]):
        s = np.linalg.svd(g[k], compute_uv=False)
        s_list.append(s)
    return np.stack(s_list, axis=0)  # (n_points, d_z)

def compute_metrics_for_arch(arch_id: str) -> Dict:
    # Агрегація по траєкторіях
    all_s = []
    if arch_id == 'baseline':
        for traj in range(1, N_TRAJ+1):
            try:
                s = load_baseline_singular_values(traj)
                all_s.append(s)
            except Exception:
                continue
    else:
        for traj in range(1, N_TRAJ+1):
            try:
                s = load_singular_values(arch_id, traj)
                all_s.append(s)
            except Exception:
                continue
    if not all_s:
        return None
    all_s = np.concatenate(all_s, axis=0)  # (N, d_z)
    eff_rank = np.mean((all_s > 1e-6).sum(axis=1))
    sigma1 = np.mean(all_s[:,0])
    sigma2 = np.mean(all_s[:,1])
    kappa = np.mean(all_s[:,0] / (all_s[:,1] + 1e-12))
    geod_lengths = load_geodesic_lengths(arch_id)
    avg_geod = np.mean(geod_lengths) if geod_lengths else np.nan
    return {
        'architecture_id': arch_id,
        'avg_eff_rank': eff_rank,
        'avg_sigma1': sigma1,
        'avg_sigma2': sigma2,
        'avg_kappa_eff': kappa,
        'avg_geodesic_length_std_path': avg_geod
    }

def principal_vectors_from_s(s: np.ndarray) -> np.ndarray:
    """Повертає перші два сингулярні вектори (u1, u2) для кожної точки траєкторії."""
    # s: (n_points, d_z) — тут потрібно також U, але якщо є лише s, повертаємо None
    return None

def load_principal_vectors(arch_id: str, traj: int, k: int = None):
    """Завантажити U (сингулярні вектори) для архітектури і траєкторії, обмежити по k компонентам."""
    u_path = os.path.join(ARCH_ROOT, arch_id, f'U_traj{traj}.npy')
    if os.path.exists(u_path):
        U = np.load(u_path)  # (n_points, d_z, d_z)
        if k is not None:
            U = U[:, :, :k]
        return U
    return None

def angle_between_vectors(u: np.ndarray, v: np.ndarray) -> float:
    """Кут у градусах між двома векторами."""
    cos_theta = np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def compute_alignment_metrics(arch_id: str, baseline_id: str = 'baseline') -> dict:
    """Обчислити alignment-метрики між arch_id і baseline по всіх траєкторіях у спільному підпросторі."""
    angles_u1, angles_u2 = [], []
    for traj in range(1, N_TRAJ+1):
        # Визначаємо k = min(d_z_baseline, d_z_arch)
        u_path_base = os.path.join(ARCH_ROOT, baseline_id, f'U_traj{traj}.npy')
        u_path_arch = os.path.join(ARCH_ROOT, arch_id, f'U_traj{traj}.npy')
        if not (os.path.exists(u_path_base) and os.path.exists(u_path_arch)):
            continue
        U_base = np.load(u_path_base)  # (n_points, d_z_base, d_z_base)
        U_arch = np.load(u_path_arch)  # (n_points, d_z_arch, d_z_arch)
        k = min(U_base.shape[1], U_arch.shape[1])
        for kvec in range(min(2, k)):
            for t in range(U_base.shape[0]):
                u_base = U_base[t, :, kvec][:k]
                u_arch = U_arch[t, :, kvec][:k]
                # Нормалізуємо
                u_base = u_base / (np.linalg.norm(u_base) + 1e-12)
                u_arch = u_arch / (np.linalg.norm(u_arch) + 1e-12)
                angle = angle_between_vectors(u_base, u_arch)
                if kvec == 0:
                    angles_u1.append(angle)
                else:
                    angles_u2.append(angle)
    return {
        'architecture_id_tested': arch_id,
        'baseline_architecture_id': baseline_id,
        'avg_angle_u1_vs_baseline_deg': np.mean(angles_u1) if angles_u1 else np.nan,
        'std_angle_u1_vs_baseline_deg': np.std(angles_u1) if angles_u1 else np.nan,
        'avg_angle_u2_vs_baseline_deg': np.mean(angles_u2) if angles_u2 else np.nan,
        'std_angle_u2_vs_baseline_deg': np.std(angles_u2) if angles_u2 else np.nan,
        'hausdorff_distance_to_baseline': np.nan  # TODO: реалізувати
    }

def main():
    summary_rows = []
    align_rows = []
    for arch_id in ARCHITECTURE_IDS:
        metrics = compute_metrics_for_arch(arch_id)
        if metrics:
            summary_rows.append(metrics)
        else:
            print(f'[!] No data for {arch_id}')
        if arch_id != 'baseline':
            align = compute_alignment_metrics(arch_id, 'baseline')
            align_rows.append(align)
    df = pd.DataFrame(summary_rows)
    df.to_csv(SUMMARY_CSV, index=False)
    print(f'[+] Saved {SUMMARY_CSV}')
    df_align = pd.DataFrame(align_rows)
    df_align.to_csv(ALIGN_CSV, index=False)
    print(f'[+] Saved {ALIGN_CSV}')

if __name__ == '__main__':
    main()
