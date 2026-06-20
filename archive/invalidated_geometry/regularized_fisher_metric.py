# filepath: c:\Users\user\OneDrive\Документы\ДослідженняСимуляційРинку\src\regularized_fisher_metric.py
"""
Завдання 1.2: Метричне згортання Fisher матриці

Цей модуль реалізує регуляризовану Fisher метрику через сингулярний розклад:
- Виконує SVD для g(z) = UΣU^T
- Будує згорнуту метрику g^(r) з r головними компонентами
- Рекалькулює геодезичні довжини з покращеною стабільністю
- Аналізує числа обумовленості та інформаційні напрямки
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import pandas as pd
from sklearn.decomposition import PCA

def load_fisher_data() -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Завантажує всі дані Fisher метрик та латентних координат.
    
    Returns:
        Dict з ключами 1-5, значення (Z, Fisher_metrics, lambdas)
    """
    data = {}
    
    for idx in range(1, 6):
        # Латентні координати
        z_path = f'data/generated/z_interp_{idx}.npy'
        Z = np.load(z_path)
        
        # Fisher метрики
        fisher_path = f'results/generation_1/fisher_metrics/fisher_metric_{idx}.npy'
        
        if os.path.exists(fisher_path):
            raise ValueError(
                f"CRITICAL ERROR: Detected invalidated legacy Fisher metric artifact '{fisher_path}'. "
                "These artifacts are classified as INVALID_DO_NOT_CLAIM due to mathematical bugs in "
                "the legacy Jacobian formulation. Please regenerate them using the canonical geometry "
                "implementation in src/geometry.py."
            )
            
        Fisher_metrics = np.load(fisher_path)
        
        # Параметр λ (рівномірний від 0 до 1)
        n_points = Z.shape[0]
        lambdas = np.linspace(0, 1, n_points)
        
        data[idx] = (Z, Fisher_metrics, lambdas)
        print(f"✅ Завантажено траєкторію {idx}: {Z.shape}, Fisher {Fisher_metrics.shape}")
    
    return data

def compute_svd_decomposition(Fisher_metrics: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Виконує сингулярний розклад для всіх Fisher матриць вздовж траєкторії.
    
    Args:
        Fisher_metrics: (n_points, d_z, d_z) - Fisher матриці
        
    Returns:
        U: (n_points, d_z, d_z) - унітарні матриці
        S: (n_points, d_z) - сингулярні значення
        condition_numbers: (n_points,) - числа обумовленості
    """
    n_points, d_z, _ = Fisher_metrics.shape
    U = np.zeros((n_points, d_z, d_z))
    S = np.zeros((n_points, d_z))
    condition_numbers = np.zeros(n_points)
    
    for k in range(n_points):
        try:
            # SVD розклад g = U @ diag(s) @ U.T
            u, s, vt = np.linalg.svd(Fisher_metrics[k])
            
            U[k] = u
            S[k] = s
            
            # Число обумовленості
            condition_numbers[k] = s[0] / (s[-1] + 1e-12)
            
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"SVD computation failed for Fisher metric at step {k}: {e}") from e
    
    return U, S, condition_numbers

def build_regularized_metric(U: np.ndarray, S: np.ndarray, r: int = 2) -> np.ndarray:
    """
    Будує регуляризовану Fisher метрику g^(r) з r головними компонентами.
    
    g^(r)(z) = Σ_{i=1}^r σ_i u_i u_i^T
    
    Args:
        U: (n_points, d_z, d_z) - унітарні матриці
        S: (n_points, d_z) - сингулярні значення  
        r: кількість головних компонент
        
    Returns:
        g_regularized: (n_points, d_z, d_z) - регуляризовані метрики
    """
    n_points, d_z, _ = U.shape
    g_regularized = np.zeros((n_points, d_z, d_z))
    
    for k in range(n_points):
        # Беремо r найбільших сингулярних значень
        for i in range(min(r, d_z)):
            u_i = U[k, :, i]  # i-й головний вектор
            sigma_i = S[k, i]  # i-е сингулярне значення
            
            # g^(r) += σ_i * u_i * u_i^T
            g_regularized[k] += sigma_i * np.outer(u_i, u_i)
    
    return g_regularized

def compute_geodesic_length_regularized(Z: np.ndarray, g_regularized: np.ndarray) -> float:
    """
    Обчислює геодезичну довжину з регуляризованою метрикою.
    
    L^(r) = Σ_k √(Δz_k^T · g^(r)(z_k) · Δz_k)
    """
    total_length = 0.0
    n_points = Z.shape[0]
    
    for k in range(n_points - 1):
        delta_z = Z[k+1] - Z[k]
        g_k = g_regularized[k]
        
        # Додаємо мінімальну регуляризацію для стабільності
        g_k_stable = g_k + 1e-8 * np.eye(g_k.shape[0])
        val = delta_z.T @ g_k_stable @ delta_z
        if val < 0:
            raise ValueError(
                f"Fisher metric tensor is not positive semi-definite at step {k}: "
                f"delta_z^T g_k delta_z = {val} < 0. Computation failed fast."
            )
        geodesic_element = np.sqrt(val)
        total_length += geodesic_element
    
    return total_length

def analyze_principal_directions(U: np.ndarray, S: np.ndarray, lambdas: np.ndarray) -> Dict:
    """
    Аналізує поведінку головних напрямків u_1, u_2 вздовж траєкторії.
    
    Returns:
        Dict з результатами аналізу
    """
    n_points, d_z, _ = U.shape
    
    results = {
        'u1_vectors': U[:, :, 0],  # Перший головний напрямок
        'u2_vectors': U[:, :, 1] if d_z > 1 else np.zeros((n_points, d_z)),  # Другий головний напрямок
        'sigma1': S[:, 0],  # Найбільші сингулярні значення
        'sigma2': S[:, 1] if d_z > 1 else np.zeros(n_points),  # Другі за величиною
        'explained_variance_ratio': np.zeros(n_points),
        'principal_angles': np.zeros(n_points - 1)  # Кути між сусідніми u1
    }
    
    # Обчислюємо частку поясненої варіації для r=2
    for k in range(n_points):
        total_variance = np.sum(S[k])
        if total_variance > 1e-12:
            results['explained_variance_ratio'][k] = (S[k, 0] + (S[k, 1] if d_z > 1 else 0)) / total_variance
        else:
            results['explained_variance_ratio'][k] = 0.0
    
    # Обчислюємо кути між сусідніми головними напрямками
    for k in range(n_points - 1):
        u1_current = U[k, :, 0]
        u1_next = U[k+1, :, 0]
        
        # Кут між векторами
        cos_angle = np.clip(np.dot(u1_current, u1_next), -1.0, 1.0)
        results['principal_angles'][k] = np.arccos(np.abs(cos_angle))  # Беремо абсолютне значення
    
    return results

def create_comparison_visualization(trajectory_idx: int, data_dict: Dict, save_dir: str):
    """
    Створює візуалізацію порівняння оригінальної та регуляризованої метрик.
    """
    Z, Fisher_original, lambdas = data_dict['original']
    g_reg, analysis = data_dict['regularized'], data_dict['analysis']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Регуляризована Fisher метрика - Траєкторія {trajectory_idx}')
    
    # 1. Сингулярні значення
    axes[0,0].semilogy(lambdas, analysis['sigma1'], 'b-', label='σ₁', linewidth=2)
    if analysis['sigma2'] is not None:
        axes[0,0].semilogy(lambdas, analysis['sigma2'], 'r-', label='σ₂', linewidth=2)
    axes[0,0].set_xlabel('λ')
    axes[0,0].set_ylabel('Сингулярні значення')
    axes[0,0].set_title('Головні компоненти Fisher метрики')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Пояснена варіація
    axes[0,1].plot(lambdas, analysis['explained_variance_ratio'], 'g-', linewidth=2)
    axes[0,1].set_xlabel('λ')
    axes[0,1].set_ylabel('Частка варіації')
    axes[0,1].set_title('Пояснена варіація (r=2)')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_ylim([0, 1])
    
    # 3. Кути між головними напрямками
    axes[0,2].plot(lambdas[:-1], np.degrees(analysis['principal_angles']), 'purple', linewidth=2)
    axes[0,2].set_xlabel('λ')
    axes[0,2].set_ylabel('Кут (градуси)')
    axes[0,2].set_title('Поворот головного напрямку u₁')
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Порівняння чисел обумовленості
    # Обчислюємо числа обумовленості для регуляризованої метрики
    kappa_original = []
    kappa_regularized = []
    for k in range(len(Fisher_original)):
        try:
            s_orig = np.linalg.svd(Fisher_original[k], compute_uv=False)
            kappa_orig = s_orig[0] / (s_orig[-1] + 1e-12)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"SVD failed for original metric in visualization at step {k}: {e}") from e
        kappa_original.append(kappa_orig)
        
        # Регуляризована метрика  
        try:
            s_reg = np.linalg.svd(g_reg[k], compute_uv=False)
            kappa_reg = s_reg[0] / (s_reg[-1] + 1e-12)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"SVD failed for regularized metric in visualization at step {k}: {e}") from e
        kappa_regularized.append(kappa_reg)
    
    axes[1,0].semilogy(lambdas, kappa_original, 'r-', label='Оригінальна g', linewidth=2)
    axes[1,0].semilogy(lambdas, kappa_regularized, 'b-', label='Регуляризована g⁽ʳ⁾', linewidth=2)
    axes[1,0].set_xlabel('λ')
    axes[1,0].set_ylabel('Число обумовленості κ')
    axes[1,0].set_title('Покращення стабільності')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Головні напрямки u1 в PCA проєкції
    # Проєктуємо латентні координати в 2D
    pca = PCA(n_components=2)
    Z_2d = pca.fit_transform(Z)
    
    # Проєктуємо головні напрямки
    u1_2d = np.zeros((len(analysis['u1_vectors']), 2))
    for k in range(len(analysis['u1_vectors'])):
        u1_full = analysis['u1_vectors'][k]
        u1_2d[k] = pca.transform(u1_full.reshape(1, -1))[0]
        # Нормалізуємо для візуалізації
        norm = np.linalg.norm(u1_2d[k])
        if norm > 1e-12:
            u1_2d[k] /= norm
    
    # Траєкторія з векторами напрямків
    axes[1,1].plot(Z_2d[:, 0], Z_2d[:, 1], 'k-', alpha=0.7, linewidth=1)
    axes[1,1].scatter(Z_2d[:, 0], Z_2d[:, 1], c=lambdas, cmap='viridis', s=30)
    
    # Додаємо стрілки головних напрямків (через кожну 3-тю точку)
    for k in range(0, len(Z_2d), 3):
        scale = 0.05
        axes[1,1].arrow(Z_2d[k, 0], Z_2d[k, 1], 
                       scale * u1_2d[k, 0], scale * u1_2d[k, 1],
                       head_width=0.01, head_length=0.01, fc='red', ec='red', alpha=0.7)
    
    axes[1,1].set_xlabel('PC1')
    axes[1,1].set_ylabel('PC2') 
    axes[1,1].set_title('Головні напрямки u₁ в латентному просторі')
    
    # 6. Детермінанти метрик
    det_original = [np.linalg.det(Fisher_original[k]) for k in range(len(Fisher_original))]
    det_regularized = [np.linalg.det(g_reg[k]) for k in range(len(g_reg))]
    
    axes[1,2].semilogy(lambdas, np.abs(det_original), 'r-', label='det(g)', linewidth=2)
    axes[1,2].semilogy(lambdas, np.abs(det_regularized), 'b-', label='det(g⁽ʳ⁾)', linewidth=2)
    axes[1,2].set_xlabel('λ')
    axes[1,2].set_ylabel('|Детермінант|')
    axes[1,2].set_title('Інформаційна щільність')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/regularized_analysis_trajectory_{trajectory_idx}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Основна функція виконання завдання 1.2"""
    print("🔧 Завдання 1.2: Метричне згортання Fisher матриці")
    print("=" * 60)
    
    # Створюємо папку для результатів
    output_dir = 'results/generation_1/regularized_fisher'
    os.makedirs(output_dir, exist_ok=True)
    
    # Завантажуємо дані
    print("\n📂 Завантаження даних...")
    all_data = load_fisher_data()
    
    # Параметри регуляризації
    r_components = 2  # Кількість головних компонент
    
    # Результати для всіх траєкторій
    results_summary = []
    
    print(f"\n🧮 Обробка траєкторій з r={r_components} компонентами...")
    
    for idx in range(1, 6):
        print(f"\n--- Траєкторія {idx} ---")
        
        Z, Fisher_original, lambdas = all_data[idx]
        
        # 1. SVD розклад
        U, S, condition_numbers_orig = compute_svd_decomposition(Fisher_original)
        print(f"✅ SVD розклад виконано")
        
        # 2. Побудова регуляризованої метрики
        g_regularized = build_regularized_metric(U, S, r=r_components)
        print(f"✅ Регуляризована метрика g^({r_components}) побудована")
        
        # 3. Рекалькуляція геодезичних довжин
        geodesic_orig = compute_geodesic_length_regularized(Z, Fisher_original)
        geodesic_reg = compute_geodesic_length_regularized(Z, g_regularized)
        euclidean_length = np.sum([np.linalg.norm(Z[k+1] - Z[k]) for k in range(len(Z)-1)])
        
        print(f"📏 Геодезична (оригінал): {geodesic_orig:.6f}")
        print(f"📏 Геодезична (регуляр.): {geodesic_reg:.6f}")
        print(f"📏 Евклідова: {euclidean_length:.6f}")
        
        # 4. Аналіз головних напрямків
        analysis = analyze_principal_directions(U, S, lambdas)
        
        mean_explained_var = np.mean(analysis['explained_variance_ratio'])
        mean_angle_change = np.mean(analysis['principal_angles'])
        
        print(f"📊 Середня пояснена варіація (r=2): {mean_explained_var:.3f}")
        print(f"📊 Середня зміна кута u1: {np.degrees(mean_angle_change):.1f}°")
        
        # 5. Числа обумовленості
        kappa_orig_mean = np.mean(condition_numbers_orig[condition_numbers_orig < np.inf])
        
        kappa_reg = []
        for k in range(len(g_regularized)):
            try:
                s_reg = np.linalg.svd(g_regularized[k], compute_uv=False)
                kappa_reg.append(s_reg[0] / (s_reg[-1] + 1e-12))
            except np.linalg.LinAlgError as e:
                raise RuntimeError(f"SVD computation failed in kappa_reg at step {k}: {e}") from e
        kappa_reg_mean = np.mean([k for k in kappa_reg if k < np.inf])
        
        print(f"📉 Число обумовленості (оригінал): {kappa_orig_mean:.2e}")
        print(f"📉 Число обумовленості (регуляр.): {kappa_reg_mean:.2e}")
        print(f"📈 Покращення стабільності: {kappa_orig_mean/kappa_reg_mean:.1f}x")
        
        # Збираємо результати
        results_summary.append({
            'trajectory': idx,
            'geodesic_original': geodesic_orig,
            'geodesic_regularized': geodesic_reg,
            'euclidean_length': euclidean_length,
            'ratio_orig_eucl': geodesic_orig / euclidean_length,
            'ratio_reg_eucl': geodesic_reg / euclidean_length,
            'condition_number_original': kappa_orig_mean,
            'condition_number_regularized': kappa_reg_mean,
            'stability_improvement': kappa_orig_mean / kappa_reg_mean,
            'explained_variance_r2': mean_explained_var,
            'mean_angle_change_deg': np.degrees(mean_angle_change)
        })
        
        # 6. Створюємо візуалізацію
        data_dict = {
            'original': (Z, Fisher_original, lambdas),
            'regularized': g_regularized,
            'analysis': analysis
        }
        create_comparison_visualization(idx, data_dict, output_dir)
        print(f"✅ Візуалізація збережена")
        
        # Зберігаємо регуляризовані метрики
        np.save(f'{output_dir}/regularized_fisher_metric_{idx}.npy', g_regularized)
        np.save(f'{output_dir}/singular_values_{idx}.npy', S)
        np.save(f'{output_dir}/principal_vectors_{idx}.npy', U)
    
    # Збереження підсумкових результатів
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv(f'{output_dir}/regularized_fisher_summary.csv', index=False)
    
    print(f"\n📊 Підсумкові результати:")
    print(results_df.round(6))
    
    # Створюємо підсумкову візуалізацію
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Підсумок: Регуляризована Fisher метрика (r=2)')
    
    # 1. Порівняння геодезичних довжин
    x = results_df['trajectory']
    axes[0,0].bar(x - 0.2, results_df['geodesic_original'], 0.4, label='Оригінальна g', alpha=0.7)
    axes[0,0].bar(x + 0.2, results_df['geodesic_regularized'], 0.4, label='Регуляризована g⁽ʳ⁾', alpha=0.7)
    axes[0,0].set_xlabel('Траєкторія')
    axes[0,0].set_ylabel('Геодезична довжина')
    axes[0,0].set_title('Порівняння геодезичних довжин')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Покращення стабільності
    axes[0,1].bar(x, results_df['stability_improvement'], alpha=0.7, color='green')
    axes[0,1].set_xlabel('Траєкторія')
    axes[0,1].set_ylabel('Коефіцієнт покращення')
    axes[0,1].set_title('Покращення числа обумовленості')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Пояснена варіація
    axes[1,0].bar(x, results_df['explained_variance_r2'], alpha=0.7, color='orange')
    axes[1,0].set_xlabel('Траєкторія')
    axes[1,0].set_ylabel('Частка варіації')
    axes[1,0].set_title('Пояснена варіація (r=2)')
    axes[1,0].set_ylim([0, 1])
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Співвідношення довжин
    axes[1,1].plot(x, results_df['ratio_orig_eucl'], 'ro-', label='g / Евклідова', linewidth=2)
    axes[1,1].plot(x, results_df['ratio_reg_eucl'], 'bo-', label='g⁽ʳ⁾ / Евклідова', linewidth=2)
    axes[1,1].set_xlabel('Траєкторія')
    axes[1,1].set_ylabel('Співвідношення')
    axes[1,1].set_title('Геодезична / Евклідова довжина')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/regularized_fisher_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Завдання 1.2 виконано!")
    print(f"📁 Результати збережено в: {output_dir}/")
    print(f"📈 Середнє покращення стабільності: {results_df['stability_improvement'].mean():.1f}x")
    print(f"📊 Середня пояснена варіація (r=2): {results_df['explained_variance_r2'].mean():.3f}")

if __name__ == '__main__':
    main()
