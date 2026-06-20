import sys
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import subprocess
import os

def check_dependencies():
    """Перевірка та встановлення необхідних залежностей."""
    print("Перевірка залежностей...")
    
    required_packages = {
        'torch': 'torch',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'sklearn': 'scikit-learn',
        'umap': 'umap-learn'
    }
    
    missing_packages = []
    
    for import_name, pip_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {import_name} встановлено")
        except ImportError:
            print(f"✗ {import_name} не встановлено")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\nВстановлення відсутніх пакетів: {', '.join(missing_packages)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("Пакети успішно встановлено!")
    
    print("-" * 50)

def run_script(script_name):
    """Запуск Python скрипту."""
    print(f"\nЗапуск {script_name}...")
    print("=" * 60)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("Попередження:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"ПОМИЛКА при виконанні {script_name}:")
        print(e.stdout)
        print(e.stderr)
        return False
    
    return True

def check_output_files():
    """Перевірка наявності вихідних файлів."""
    print("\nПеревірка вихідних файлів:")
    print("-" * 50)
    
    expected_files = [
        ('z_train_ar.npy', 'Латентні вектори тренувальних AR моделей'),
        ('z_train_garch.npy', 'Латентні вектори тренувальних GARCH моделей'),
        ('z_generated.npy', 'Латентні вектори згенерованих ARMA-GARCH моделей'),
        ('latent_space_tsne.png', 't-SNE візуалізація'),
        ('latent_space_umap.png', 'UMAP візуалізація')
    ]
    
    all_present = True
    for filename, description in expected_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"✓ {filename} ({size} байт) - {description}")
        else:
            print(f"✗ {filename} - {description}")
            all_present = False
    
    return all_present

def main():
    """Основна функція для запуску повного пайплайну."""
    print("=" * 60)
    print("ЗАПУСК ПОВНОГО ПАЙПЛАЙНУ ВІЗУАЛІЗАЦІЇ ЛАТЕНТНОГО ПРОСТОРУ")
    print("=" * 60)
    
    # Перевірка залежностей
    check_dependencies()
    
    # Крок 1: Генерація латентних векторів
    print("\nКРОК 1: ГЕНЕРАЦІЯ ЛАТЕНТНИХ ВЕКТОРІВ")
    if not run_script("generate_latent_vectors.py"):
        print("\nПомилка на етапі генерації латентних векторів!")
        return
    
    # Крок 2: Візуалізація латентного простору
    print("\nКРОК 2: ВІЗУАЛІЗАЦІЯ ЛАТЕНТНОГО ПРОСТОРУ")
    if not run_script("latent_space_visualization.py"):
        print("\nПомилка на етапі візуалізації!")
        return
    
    # Перевірка результатів
    print("\nПЕРЕВІРКА РЕЗУЛЬТАТІВ")
    if check_output_files():
        print("\n✓ Всі файли успішно створено!")
        print("\nРезультати візуалізації збережено в:")
        print("  - latent_space_tsne.png")
        print("  - latent_space_umap.png")
    else:
        print("\n✗ Деякі файли відсутні. Перевірте виконання скриптів.")
    
    print("\n" + "=" * 60)
    print("ПАЙПЛАЙН ЗАВЕРШЕНО!")
    print("=" * 60)

if __name__ == '__main__':
    main()
