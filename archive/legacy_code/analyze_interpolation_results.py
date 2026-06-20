import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Налаштування для української мови
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Завантаження всіх файлів
ANALYSIS_DIR = 'results/generation_1/analysis'
dfs = []
for i in range(1, 6):
    filename = os.path.join(ANALYSIS_DIR, f'interpolation_detailed_{i}.csv')
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        dfs.append((i, df))

if not dfs:
    print('Попередження: Не знайдено жодного файлу interpolation_detailed_*.csv у results/generation_1/analysis/. Завершення роботи.')
    exit(0)

# 1. Графік валідності
plt.figure(figsize=(10, 6))
for i, df in dfs:
    # Конвертуємо is_param_valid в числові значення (1 для True, 0 для False)
    validity = df['is_param_valid'].map({True: 1, False: 0})
    plt.plot(df['lambda'], validity, marker='o', label=f'Траєкторія {i}', alpha=0.7)

plt.xlabel('λ')
plt.ylabel('Валідність параметрів (1=True, 0=False)')
plt.title('Валідність параметрів моделей вздовж траєкторій інтерполяції')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 1.1)
plt.savefig('validity_by_lambda.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Графіки сум модулів коефіцієнтів
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

for i, df in dfs:
    # Сума |φ_k|
    sum_phi = df['phi1'].abs() + df['phi2'].abs()
    ax1.plot(df['lambda'], sum_phi, marker='o', label=f'Траєкторія {i}', alpha=0.7)
    
    # Сума |θ_l|
    sum_theta = df['theta1'].abs() + df['theta2'].abs() + df['theta3'].abs()
    ax2.plot(df['lambda'], sum_theta, marker='o', label=f'Траєкторія {i}', alpha=0.7)

ax1.set_xlabel('λ')
ax1.set_ylabel('Σ|φ_k|')
ax1.set_title('Сума модулів AR коефіцієнтів')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('λ')
ax2.set_ylabel('Σ|θ_l|')
ax2.set_title('Сума модулів MA коефіцієнтів')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('coefficients_sums_by_lambda.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Аналіз норм параметрів
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Збір даних для кореляційного аналізу
correlations_l1 = []
correlations_l2 = []

for i, df in dfs:
    # L1 норма (сума модулів)
    phi_l1 = df['phi1'].abs() + df['phi2'].abs()
    theta_l1 = df['theta1'].abs() + df['theta2'].abs() + df['theta3'].abs()
    
    # L2 норма (евклідова)
    phi_l2 = np.sqrt(df['phi1']**2 + df['phi2']**2)
    theta_l2 = np.sqrt(df['theta1']**2 + df['theta2']**2 + df['theta3']**2)
    
    # Візуалізація L1 норм
    ax1.plot(df['lambda'], phi_l1, marker='o', linestyle='-', label=f'Траєкторія {i} (φ)', alpha=0.7)
    ax1.plot(df['lambda'], theta_l1, marker='s', linestyle='--', label=f'Траєкторія {i} (θ)', alpha=0.7)
    
    # Обчислення кореляцій
    corr_phi_l1 = df['lambda'].corr(phi_l1)
    corr_theta_l1 = df['lambda'].corr(theta_l1)
    corr_phi_l2 = df['lambda'].corr(phi_l2)
    corr_theta_l2 = df['lambda'].corr(theta_l2)
    
    correlations_l1.append((i, corr_phi_l1, corr_theta_l1))
    correlations_l2.append((i, corr_phi_l2, corr_theta_l2))

ax1.set_xlabel('λ')
ax1.set_ylabel('L1 норма')
ax1.set_title('L1 норми векторів коефіцієнтів AR (φ) та MA (θ)')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)

# Таблиця кореляцій
ax2.axis('tight')
ax2.axis('off')

# Підготовка даних для таблиці
table_data = []
for (i, corr_phi_l1, corr_theta_l1), (_, corr_phi_l2, corr_theta_l2) in zip(correlations_l1, correlations_l2):
    table_data.append([
        f'Траєкторія {i}',
        f'{corr_phi_l1:.3f}',
        f'{corr_theta_l1:.3f}',
        f'{corr_phi_l2:.3f}',
        f'{corr_theta_l2:.3f}'
    ])

table = ax2.table(cellText=table_data,
                  colLabels=['', 'Corr(λ, L1_φ)', 'Corr(λ, L1_θ)', 'Corr(λ, L2_φ)', 'Corr(λ, L2_θ)'],
                  loc='center',
                  cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

plt.tight_layout()
plt.savefig('norms_and_correlations.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Завантаження даних ---
sum_path = 'results/generation_2/robustness/architecture_geometric_summary.csv'
align_path = 'results/generation_2/robustness/architecture_alignment_vs_baseline.csv'
df = pd.read_csv(sum_path)
df_align = pd.read_csv(align_path)

archs = df['architecture_id'].values

# --- Візуалізація геометричних метрик ---
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.bar(archs, df['avg_sigma1'], label='σ₁')
plt.bar(archs, df['avg_sigma2'], label='σ₂', alpha=0.7)
plt.ylabel('Середнє сингулярне значення')
plt.title('Сингулярні значення manifold')
plt.legend()

plt.subplot(1,2,2)
plt.bar(archs, df['avg_kappa_eff'])
plt.ylabel('κ (анізотропія)')
plt.title('Анізотропія manifold')
plt.tight_layout()
plt.savefig('results/generation_2/robustness/summary_singular_kappa.png')

# --- Візуалізація alignment ---
plt.figure(figsize=(8,5))
plt.bar(df_align['architecture_id_tested'], df_align['avg_angle_u1_vs_baseline_deg'], label='u₁')
plt.bar(df_align['architecture_id_tested'], df_align['avg_angle_u2_vs_baseline_deg'], label='u₂', alpha=0.7)
plt.ylabel('Середній кут (градуси)')
plt.title('Кут між напрямками manifold (u₁, u₂) та baseline')
plt.legend()
plt.tight_layout()
plt.savefig('results/generation_2/robustness/summary_alignment_angles.png')

print('Візуалізації збережено у results/generation_2/robustness/')

# Вивід підсумкової статистики
print("=== АНАЛІЗ ЗАВЕРШЕНО ===")
print("\n1. Валідність:")
for i, df in dfs:
    valid_count = df['is_param_valid'].sum()
    total_count = len(df)
    print(f"   Траєкторія {i}: {valid_count}/{total_count} ({100*valid_count/total_count:.1f}%) валідних моделей")

print("\n2. Кореляції з λ:")
print("   L1 норми:")
for i, corr_phi, corr_theta in correlations_l1:
    print(f"     Траєкторія {i}: φ={corr_phi:+.3f}, θ={corr_theta:+.3f}")

print("\nГрафіки збережено:")
print("  - validity_by_lambda.png")
print("  - coefficients_sums_by_lambda.png")
print("  - norms_and_correlations.png")