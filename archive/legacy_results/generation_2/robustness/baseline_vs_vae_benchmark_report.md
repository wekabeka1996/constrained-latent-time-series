# Baseline Random Search Benchmark Report

**Дата:** 2025-06-11

## Мета
Порівняти diversity та ефективність генерації ARMA-GARCH моделей між VAE (β=5.0) та baseline (random search + rejection sampling).

---

## Результати baseline_random_search.py

- **Валідних ARMA-GARCH моделей (baseline):** 1000
- **Кількість спроб:** 4705
- **Ефективність (успіх/спроба):** 0.213
- **Shannon entropy (порядки):** 6.327
- **Top-5 порядків:**
  - (1, 4, 1, 2): 6
  - (5, 2, 1, 4): 5
  - (1, 2, 1, 1): 5
  - (3, 4, 1, 1): 5
  - (1, 1, 2, 4): 5
- **Частка ARMA-only:** 0.73%
- **Частка GARCH-only:** 0.93%
- **Частка ARMA-GARCH:** 98.34%

### Таргетована генерація ARMA(2,2)
- **100 валідних ARMA(2,2) моделей**
- **Кількість спроб:** 11308
- **Ефективність:** 0.009

---

## Порівняння з VAE (β=5.0)

- **VAE (β=5.0):**
  - Ентропія: (див. arma_garch_beta_summary.csv, β=5.0)
  - Ефективність: ~1.0 (майже всі згенеровані валідні)
  - Діапазон порядків ширший, частка ARMA-only ≈ 0

---

## Висновки
- **Baseline**: Дуже низька ефективність для таргетованих порядків, нижча ентропія, більшість моделей — ARMA-GARCH.
- **VAE**: Вища ентропія, майже ідеальна ефективність, кращий контроль над порядками.

---

## Файли
- results/generation_2/robustness/baseline_valid_thetas.npy
- results/generation_2/robustness/baseline_order_stats.csv
- arma_garch_beta_summary.csv (VAE)
- entropy_vs_beta.png

---

**Оновлено:** 2025-06-11
