import sys
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import os
import warnings
warnings.filterwarnings('ignore')

# Налаштування шрифтів для підтримки українських символів
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_latent_vectors():
    """Завантаження латентних векторів з файлів."""
    print("Завантаження латентних векторів...")
    files_to_load = {
        'z_train_ar.npy': 'Тренувальні AR',
        'z_train_garch.npy': 'Тренувальні GARCH',
        'z_generated.npy': 'Згенеровані ARMA-GARCH',
        'z_interp.npy': 'Інтерполяційний шлях'
    }
    # Додаємо всі z_interp_{i}.npy
    for i in range(1, 20):
        fname = f'z_interp_{i}.npy'
        if os.path.exists(fname):
            files_to_load[fname] = f'Інтерполяційний шлях {i}'
    data = {}
    for filename, label in files_to_load.items():
        if os.path.exists(filename):
            vectors = np.load(filename)
            data[label] = vectors
            print(f"✓ {filename}: {vectors.shape} ({label})")
        else:
            print(f"✗ {filename} не знайдено")
    return data

def perform_tsne(data_dict, perplexity=30, n_iter=1000, random_state=42):
    """Виконання t-SNE для зменшення розмірності до 2D."""
    print(f"\nВиконання t-SNE (perplexity={perplexity})...")
    
    # Об'єднання всіх даних
    all_data = []
    labels = []
    colors = []
    
    color_map = {
        'Тренувальні AR': 'blue',
        'Тренувальні GARCH': 'red', 
        'Згенеровані ARMA-GARCH': 'green',
        'Інтерполяційний шлях': 'gold'
    }
    # Додаємо кольори для багатьох траєкторій
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab10')
    interp_idx = 0
    for label, vectors in data_dict.items():
        if vectors.shape[0] > 0:
            all_data.append(vectors)
            labels.extend([label] * vectors.shape[0])
            if label.startswith('Інтерполяційний шлях'):
                color = cmap(interp_idx % 10)
                colors.extend([color] * vectors.shape[0])
                interp_idx += 1
            else:
                colors.extend([color_map.get(label, 'black')] * vectors.shape[0])
    
    if not all_data:
        print("Немає даних для візуалізації!")
        return None, None, None
    
    all_data = np.vstack(all_data)
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, 
                random_state=random_state, verbose=1)
    tsne_results = tsne.fit_transform(all_data)
    
    print("t-SNE завершено.")
    return tsne_results, labels, colors

def perform_umap(data_dict, n_neighbors=15, min_dist=0.1, random_state=42):
    """Виконання UMAP для зменшення розмірності до 2D."""
    print(f"\nВиконання UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    
    # Об'єднання всіх даних
    all_data = []
    labels = []
    colors = []
    
    color_map = {
        'Тренувальні AR': 'blue',
        'Тренувальні GARCH': 'red',
        'Згенеровані ARMA-GARCH': 'green',
        'Інтерполяційний шлях': 'gold'
    }
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab10')
    interp_idx = 0
    for label, vectors in data_dict.items():
        if vectors.shape[0] > 0:
            all_data.append(vectors)
            labels.extend([label] * vectors.shape[0])
            if label.startswith('Інтерполяційний шлях'):
                color = cmap(interp_idx % 10)
                colors.extend([color] * vectors.shape[0])
                interp_idx += 1
            else:
                colors.extend([color_map.get(label, 'black')] * vectors.shape[0])
    
    if not all_data:
        print("Немає даних для візуалізації!")
        return None, None, None
    
    all_data = np.vstack(all_data)
    
    # UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, 
                       n_components=2, random_state=random_state)
    umap_results = reducer.fit_transform(all_data)
    
    print("UMAP завершено.")
    return umap_results, labels, colors

def create_visualization(embeddings, labels, colors, method_name, save_path=None):
    """Створення візуалізації латентного простору."""
    plt.figure(figsize=(12, 10))
    
    # Визначаємо всі інтерполяційні шляхи
    interp_labels = [l for l in set(labels) if l.startswith('Інтерполяційний шлях')]
    unique_labels = list(set(labels) - set(interp_labels))
    
    # Малюємо основні групи
    for label in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == label]
        label_embeddings = embeddings[idx]
        label_colors = [colors[i] for i in idx]
        plt.scatter(label_embeddings[:, 0], label_embeddings[:, 1], 
                   c=label_colors[0], label=label, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    # Кольори для траєкторій
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab10')
    for j, interp_label in enumerate(sorted(interp_labels)):
        interp_indices = [i for i, l in enumerate(labels) if l == interp_label]
        interp_emb = embeddings[interp_indices]
        color = cmap(j % 10)
        plt.plot(interp_emb[:, 0], interp_emb[:, 1], color=color, marker='o', markersize=10, linewidth=3, label=interp_label, zorder=10)
        # Додаємо мітки lambda
        n_points = interp_emb.shape[0]
        for k in range(n_points):
            lambda_val = k / (n_points - 1) if n_points > 1 else 0
            plt.text(interp_emb[k, 0], interp_emb[k, 1], f'{lambda_val:.2f}', fontsize=8, color=color, weight='bold', zorder=20)
    
    plt.title(f'Візуалізація латентного простору VAE ({method_name})', fontsize=16, pad=20)
    plt.xlabel(f'{method_name} компонента 1', fontsize=12)
    plt.ylabel(f'{method_name} компонента 2', fontsize=12)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    
    # Додаємо статистику
    stats_text = f"Всього точок: {len(labels)}\n"
    for label in unique_labels:
        count = labels.count(label)
        stats_text += f"{label}: {count}\n"
    for interp_label in interp_labels:
        count = labels.count(interp_label)
        stats_text += f"{interp_label}: {count}\n"
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Візуалізацію збережено: {save_path}")
    
    plt.show()

def analyze_clusters(embeddings, labels, method_name):
    """Аналіз кластерів в латентному просторі."""
    print(f"\nАналіз кластерів ({method_name}):")
    print("-" * 50)
    
    unique_labels = list(set(labels))
    
    for label in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == label]
        label_embeddings = embeddings[idx]
        
        if len(label_embeddings) > 0:
            centroid = np.mean(label_embeddings, axis=0)
            spread = np.std(label_embeddings, axis=0)
            
            print(f"\n{label}:")
            print(f"  Кількість точок: {len(label_embeddings)}")
            print(f"  Центроїд: [{centroid[0]:.3f}, {centroid[1]:.3f}]")
            print(f"  Стандартне відхилення: [{spread[0]:.3f}, {spread[1]:.3f}]")
            
            # Обчислення відстаней між центроїдами
            for other_label in unique_labels:
                if other_label != label:
                    other_idx = [i for i, l in enumerate(labels) if l == other_label]
                    if len(other_idx) > 0:
                        other_centroid = np.mean(embeddings[other_idx], axis=0)
                        distance = np.linalg.norm(centroid - other_centroid)
                        print(f"  Відстань до {other_label}: {distance:.3f}")

def main():
    """Основна функція для виконання візуалізації."""
    print("=" * 60)
    print("ВІЗУАЛІЗАЦІЯ ЛАТЕНТНОГО ПРОСТОРУ VAE")
    print("=" * 60)
    
    # Завантаження латентних векторів
    data = load_latent_vectors()
    
    if not data:
        print("\nПОМИЛКА: Не знайдено жодного файлу з латентними векторами!")
        print("Спочатку запустіть generate_latent_vectors.py")
        return
    
    # t-SNE візуалізація
    tsne_results, labels, colors = perform_tsne(data)
    if tsne_results is not None:
        create_visualization(tsne_results, labels, colors, 't-SNE', 'latent_space_tsne.png')
        analyze_clusters(tsne_results, labels, 't-SNE')
    
    # UMAP візуалізація
    try:
        umap_results, labels, colors = perform_umap(data)
        if umap_results is not None:
            create_visualization(umap_results, labels, colors, 'UMAP', 'latent_space_umap.png')
            analyze_clusters(umap_results, labels, 'UMAP')
    except Exception as e:
        print(f"\nПОПЕРЕДЖЕННЯ: Не вдалося виконати UMAP: {e}")
        print("Можливо, потрібно встановити: pip install umap-learn")
    
    print("\n" + "=" * 60)
    print("Візуалізацію завершено!")
    print("=" * 60)

if __name__ == '__main__':
    main()
