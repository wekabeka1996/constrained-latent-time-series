"""
Обчислення інформаційної метрики Фішера g_ij(z) вздовж інтерполяційних траєкторій AR-GARCH.

Цей модуль реалізує завдання 1.1 з дослідження VAE латентного простору:
- Завантаження interpolation_detailed_*.csv файлів
- Обчислення апроксимованої метрики Фішера через Якобіан g_ij ≈ J^T J
- Збереження результатів для подальшого аналізу геодезичних довжин
"""

import os
import numpy as np
import pandas as pd
from typing import Callable
from numpy.typing import NDArray

# --- Налаштування ---
CSV_DIR = 'results/generation_1/analysis/'
Z_DIR = 'data/generated/'
CSV_PATTERN = 'interpolation_detailed_{}.csv'
Z_PATTERN = 'z_interp_{}.npy'
N_TRAJECTORIES = 5
EPS = 1e-4

# --- Ядро: чисельний Якобіан ---
def numerical_jacobian(theta_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]], 
                      z: NDArray[np.float64], 
                      eps: float = EPS) -> NDArray[np.float64]:
    """
    Обчислює чисельний Якобіан функції theta_fn по z.
    
    Args:
        theta_fn: функція z -> theta, де theta - параметри моделі
        z: латентний вектор shape (d,)
        eps: крок для чисельного диференціювання
        
    Returns:
        J: Якобіан shape (n_theta, d)
    """
    d = len(z)
    theta0 = theta_fn(z)
    n_theta = len(theta0)
    J = np.zeros((n_theta, d))
    for i in range(d):
        z_plus = np.array(z, copy=True)
        z_minus = np.array(z, copy=True)
        z_plus[i] += eps
        z_minus[i] -= eps
        J[:, i] = (theta_fn(z_plus) - theta_fn(z_minus)) / (2 * eps)
    return J

# --- Основна функція ---
def main() -> None:
    """Основна функція для обчислення Fisher метрики для всіх траєкторій."""
    for idx in range(1, N_TRAJECTORIES + 1):
        csv_path = os.path.join(CSV_DIR, CSV_PATTERN.format(idx))
        z_path = os.path.join(Z_DIR, Z_PATTERN.format(idx))
        
        if not os.path.exists(csv_path):
            print(f"[!] {csv_path} not found, skipping.")
            continue
            
        if not os.path.exists(z_path):
            print(f"[!] {z_path} not found, skipping.")
            continue
            
        # Завантажуємо параметри з CSV та латентні координати з NPY
        df = pd.read_csv(csv_path)
        Z: NDArray[np.float64] = np.load(z_path).astype(np.float64)
        
        # Витягуємо параметри моделей
        param_cols = ['phi1', 'phi2', 'theta1', 'theta2', 'theta3']
        Theta: NDArray[np.float64] = df[param_cols].values.astype(np.float64)
        
        n_points: int = Z.shape[0]
        d_z: int = Z.shape[1]
        
        print(f"[+] Processing trajectory {idx}: {n_points} points, latent dim = {d_z}")
        
        fisher_metrics: NDArray[np.float64] = np.zeros((n_points, d_z, d_z))        # Для кожної точки λ: обчислити Якобіан dθ/dz через скінченні різниці
        for k in range(n_points):
            # Функція θ(z) через локальну лінійну інтерполяцію
            # Оскільки ми маємо дискретні точки, використовуємо локальний градієнт
            if k == 0:
                # Перша точка: використовуємо наступну
                dz = Z[k+1] - Z[k]
                dtheta = Theta[k+1] - Theta[k]
            elif k == n_points - 1:
                # Остання точка: використовуємо попередню
                dz = Z[k] - Z[k-1]
                dtheta = Theta[k] - Theta[k-1]
            else:
                # Середні точки: центральна різниця
                dz = (Z[k+1] - Z[k-1]) / 2.0
                dtheta = (Theta[k+1] - Theta[k-1]) / 2.0
            
            # Обчислюємо Якобіан dθ/dz
            # Якщо dz має нульові компоненти, регуляризуємо
            eps_reg = 1e-8
            J = np.zeros((dtheta.shape[0], dz.shape[0]))
            for i in range(dtheta.shape[0]):
                for j in range(dz.shape[0]):
                    if abs(dz[j]) > eps_reg:
                        J[i, j] = dtheta[i] / dz[j]
                    else:
                        J[i, j] = 0.0
            
            # Fisher metric: g_ij = J^T @ J
            fisher_metrics[k] = J.T @ J
              # Зберігаємо результат
        output_dir = 'results/generation_1/fisher_metrics'
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f'fisher_metric_{idx}.npy')
        np.save(out_path, fisher_metrics)
        print(f'[+] Saved Fisher metric for trajectory {idx} to {out_path}')

if __name__ == '__main__':
    main()
