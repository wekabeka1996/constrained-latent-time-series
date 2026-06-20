"""
Аналіз результатів обчислення інформаційної метрики Фішера.

Цей модуль реалізує аналіз результатів завдання 1.1:
- Обчислення геодезичних довжин вздовж траєкторій
- Порівняння з Евклідовими довжинами
- Візуалізація Fisher metric у проєкції (UMAP/t-SNE)
- Аналіз сингулярних значень метрики
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Try to import UMAP, fall back to alternatives if not available
try:
    import umap
    umap_available = True
except ImportError:
    umap_available = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")

def load_trajectory_data(idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Завантажує дані для траєкторії idx.
    
    Returns:
        Z: латентні координати (n_points, d_z)
        Fisher_metrics: метрики Фішера (n_points, d_z, d_z)
        Theta: параметри моделей (n_points, n_params)
    """
    z_path = f'data/generated/z_interp_{idx}.npy'
    fisher_path = f'results/generation_1/fisher_metrics/fisher_metric_{idx}.npy'
    csv_path = f'results/generation_1/analysis/interpolation_detailed_{idx}.csv'
    
    Z = np.load(z_path)
    Fisher_metrics = np.load(fisher_path)
    
    df = pd.read_csv(csv_path)
    param_cols = ['phi1', 'phi2', 'theta1', 'theta2', 'theta3']
    Theta = df[param_cols].values
    
    return Z, Fisher_metrics, Theta

def compute_geodesic_length(Z: np.ndarray, Fisher_metrics: np.ndarray) -> float:
    """
    Обчислює геодезичну довжину вздовж траєкторії.
    
    L = ∑_k √(Δz_k^T · g_k · Δz_k)
    """
    total_length = 0.0
    n_points = Z.shape[0]
    
    for k in range(n_points - 1):
        delta_z = Z[k+1] - Z[k]
        g_k = Fisher_metrics[k]
        
        # Регуляризація для уникнення проблем з виродженими матрицями
        g_k_reg = g_k + 1e-6 * np.eye(g_k.shape[0])
        
        try:
            geodesic_element = np.sqrt(delta_z.T @ g_k_reg @ delta_z)
            total_length += geodesic_element
        except:
            # Якщо виникла помилка, використовуємо Евклідову відстань
            euclidean_element = np.linalg.norm(delta_z)
            total_length += euclidean_element
    
    return total_length

def compute_euclidean_length(Z: np.ndarray) -> float:
    """Обчислює Евклідову довжину траєкторії."""
    return np.sum([np.linalg.norm(Z[k+1] - Z[k]) for k in range(len(Z)-1)])

def analyze_singular_values(Fisher_metrics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Аналізує сингулярні значення метрики Фішера вздовж траєкторії.
    
    Returns:
        singular_values: (n_points, d_z) - сингулярні значення для кожної точки
        condition_numbers: (n_points,) - числа обумовленості
    """
    n_points, d_z, _ = Fisher_metrics.shape
    singular_values = np.zeros((n_points, d_z))
    condition_numbers = np.zeros(n_points)
    
    for k in range(n_points):
        try:
            U, s, Vt = np.linalg.svd(Fisher_metrics[k])
            singular_values[k] = s
            condition_numbers[k] = s[0] / (s[-1] + 1e-12)  # max/min
        except:
            singular_values[k] = np.zeros(d_z)
            condition_numbers[k] = np.inf
    
    return singular_values, condition_numbers

def visualize_trajectory_in_2d(Z: np.ndarray, method: str = 'pca') -> np.ndarray:
    """
    Візуалізує латентну траєкторію в 2D через dimensionality reduction.
    
    Args:
        Z: латентні координати (n_points, d_z)
        method: 'umap', 'tsne', або 'pca'
        
    Returns:
        Z_2d: проєкція в 2D (n_points, 2)
    """
    if method == 'umap' and umap_available:
        reducer = umap.UMAP(n_neighbors=min(10, len(Z)-1), min_dist=0.1, random_state=42)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(10, len(Z)-1))
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    else:
        print(f"Method {method} not available, falling back to PCA")
        reducer = PCA(n_components=2, random_state=42)
    
    Z_2d = reducer.fit_transform(Z)
    return Z_2d

def main():
    """Основна функція аналізу."""
    print("🔍 Аналіз інформаційної метрики Фішера")
    print("=" * 50)
    
    # Створюємо папку для візуалізацій
    viz_dir = 'results/generation_1/fisher_analysis'
    os.makedirs(viz_dir, exist_ok=True)
    
    # Аналіз всіх траєкторій
    results = []
    
    for idx in range(1, 6):
        print(f"\n📊 Траєкторія {idx}")
        print("-" * 30)
        
        try:
            Z, Fisher_metrics, Theta = load_trajectory_data(idx)
            
            # Обчислюємо довжини
            geodesic_length = compute_geodesic_length(Z, Fisher_metrics)
            euclidean_length = compute_euclidean_length(Z)
            
            # Аналіз сингулярних значень
            singular_values, condition_numbers = analyze_singular_values(Fisher_metrics)
            
            print(f"Геодезична довжина: {geodesic_length:.6f}")
            print(f"Евклідова довжина:  {euclidean_length:.6f}")
            print(f"Відношення Geo/Eucl: {geodesic_length/euclidean_length:.6f}")
            print(f"Середнє число обумовленості: {np.mean(condition_numbers[condition_numbers < np.inf]):.2e}")
            
            # Зберігаємо результати
            results.append({
                'trajectory': idx,
                'geodesic_length': geodesic_length,
                'euclidean_length': euclidean_length,
                'ratio': geodesic_length / euclidean_length,
                'mean_condition_number': np.mean(condition_numbers[condition_numbers < np.inf])
            })
            
            # Візуалізація в 2D
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Траєкторія {idx} - Аналіз Fisher Metric', fontsize=14)
            
            # PCA проєкція
            Z_pca = visualize_trajectory_in_2d(Z, 'pca')
            axes[0,0].plot(Z_pca[:, 0], Z_pca[:, 1], 'o-', alpha=0.7)
            axes[0,0].set_title('PCA проєкція латентного простору')
            axes[0,0].set_xlabel('PC1')
            axes[0,0].set_ylabel('PC2')
            
            # Сингулярні значення
            axes[0,1].plot(singular_values.T, alpha=0.7)
            axes[0,1].set_title('Сингулярні значення Fisher metric')
            axes[0,1].set_xlabel('Номер сингулярного значення')
            axes[0,1].set_ylabel('Значення')
            axes[0,1].set_yscale('log')
            
            # Числа обумовленості
            lambdas = np.linspace(0, 1, len(condition_numbers))
            axes[1,0].plot(lambdas, condition_numbers, 'o-')
            axes[1,0].set_title('Число обумовленості Fisher metric')
            axes[1,0].set_xlabel('λ')
            axes[1,0].set_ylabel('Condition number')
            axes[1,0].set_yscale('log')
            
            # Детермінант Fisher metric
            determinants = [np.linalg.det(Fisher_metrics[k]) for k in range(len(Fisher_metrics))]
            axes[1,1].plot(lambdas, np.abs(determinants), 'o-')
            axes[1,1].set_title('|Детермінант| Fisher metric')
            axes[1,1].set_xlabel('λ')
            axes[1,1].set_ylabel('|det(g)|')
            axes[1,1].set_yscale('log')
            
            plt.tight_layout()
            plt.savefig(f'{viz_dir}/trajectory_{idx}_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"❌ Помилка для траєкторії {idx}: {e}")
    
    # Зберігаємо зведену таблицю
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{viz_dir}/fisher_metrics_summary.csv', index=False)
    
    print(f"\n📊 Зведені результати:")
    print(results_df.to_string(index=False))
    
    # Загальна візуалізація
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.bar(results_df['trajectory'], results_df['geodesic_length'], alpha=0.7, label='Геодезична')
    plt.bar(results_df['trajectory'], results_df['euclidean_length'], alpha=0.7, label='Евклідова')
    plt.xlabel('Траєкторія')
    plt.ylabel('Довжина')
    plt.title('Порівняння довжин траєкторій')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(results_df['trajectory'], results_df['ratio'], 'o-')
    plt.xlabel('Траєкторія')
    plt.ylabel('Геодезична / Евклідова')
    plt.title('Відношення довжин')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/summary_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Результати збережено в {viz_dir}/")

if __name__ == '__main__':
    main()
