import torch
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import sys
import os
import math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Симуляція_2 import decode_discrete_structure

# --- Конфігурація ---
BETA_LIST = [0.1, 0.5, 1.0, 2.0, 5.0]
MODEL_PATHS = [
    'models/vae_beta01.pth',
    'models/vae_beta05.pth',
    'models/vae_beta10.pth',
    'models/vae_beta20.pth',
    'models/vae_beta50.pth',
]
N_SAMPLES = 2000
D_LATENT = 8
D_INPUT = 40

# --- VAE (базова архітектура) ---
class VAE(torch.nn.Module):
    def __init__(self, d_input=D_INPUT, d_latent=D_LATENT):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(d_input, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 32), torch.nn.ReLU()
        )
        self.fc_mu = torch.nn.Linear(32, d_latent)
        self.fc_logvar = torch.nn.Linear(32, d_latent)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(d_latent, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, d_input), torch.nn.Tanh()
        )
    def decode(self, z):
        return self.decoder(z)

# --- Валідація ARMA-GARCH ---
def validate_arma_garch(x, epsilon=0.05):
    """
    Робастна валідація: декодує структуру та перевіряє валідність як у Фазі I.
    """
    struct_vec = x[:8]  # Адаптуйте при потребі до вашого формату
    decoded_struct = decode_discrete_structure(struct_vec, epsilon=epsilon)
    model_type = decoded_struct.get("type", "UNKNOWN")
    p = decoded_struct.get('p_order', 0)
    q = decoded_struct.get('q_order', 0)
    r = decoded_struct.get('r_order', 0)
    s = decoded_struct.get('s_order', 0)
    is_arma_struct_active = (p > 0 or q > 0)
    is_garch_struct_active = (r > 0 or s > 0)
    if model_type == "ARMA" and is_arma_struct_active:
        return True, (p, q, r, s)
    if model_type == "GARCH" and is_garch_struct_active:
        return True, (p, q, r, s)
    if model_type == "ARMA-GARCH" and is_arma_struct_active and is_garch_struct_active:
        return True, (p, q, r, s)
    return False, (p, q, r, s)

# --- Основний цикл ---
results = []
for beta, model_path in zip(BETA_LIST, MODEL_PATHS):
    print(f'\n=== β={beta} ===')
    model = VAE()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    z_samples = np.random.randn(N_SAMPLES, D_LATENT).astype(np.float32)
    z_tensor = torch.tensor(z_samples)
    with torch.no_grad():
        x_hat = model.decode(z_tensor).numpy()
    valid = []
    orders = []
    for x in x_hat:
        is_valid, pqrs = validate_arma_garch(x)
        if is_valid:
            valid.append(True)
            orders.append(pqrs)
        else:
            valid.append(False)
    n_valid = sum(valid)
    unique_orders = Counter(orders)
    # Shannon entropy calculation
    total_valid = n_valid if n_valid > 0 else 1
    order_counts = np.array(list(unique_orders.values()))
    order_probs = order_counts / order_counts.sum() if order_counts.sum() > 0 else np.ones_like(order_counts)
    entropy = -np.sum(order_probs * np.log(order_probs + 1e-12))
    # Top-3 most frequent orders and their shares
    top3 = unique_orders.most_common(3)
    top3_orders = [str(k) for k, v in top3]
    top3_shares = [v / total_valid for k, v in top3]
    print(f'Валідних ARMA-GARCH: {n_valid} / {N_SAMPLES}')
    print('Топ-5 порядків:')
    for order, count in unique_orders.most_common(5):
        print(f'  {order}: {count}')
    results.append({
        'beta': beta,
        'n_valid': n_valid,
        'unique_orders': unique_orders,
        'total': N_SAMPLES,
        'entropy': entropy,
        'top3_orders': top3_orders,
        'top3_shares': top3_shares
    })
    # Зберігаємо гістограму
    if n_valid > 0:
        top_orders = unique_orders.most_common(10)
        labels = [str(k) for k, v in top_orders]
        counts = [v for k, v in top_orders]
        plt.figure(figsize=(8,4))
        plt.bar(labels, counts)
        plt.title(f'β={beta}: Топ-10 порядків ARMA-GARCH')
        plt.ylabel('Кількість')
        plt.xlabel('(p,q,r,s)')
        plt.tight_layout()
        plt.savefig(f'results/generation_2/robustness/arma_garch_hist_beta{str(beta).replace(".","")}.png', dpi=150)
        plt.close()

# Зведена таблиця
summary = pd.DataFrame({
    'beta': [r['beta'] for r in results],
    'n_valid': [r['n_valid'] for r in results],
    'n_unique_orders': [len(r['unique_orders']) for r in results],
    'entropy': [r['entropy'] for r in results],
    'top3_orders': [r['top3_orders'] for r in results],
    'top3_shares': [r['top3_shares'] for r in results],
    'total': [r['total'] for r in results],
    'valid_frac': [r['n_valid']/r['total'] for r in results]
})
summary.to_csv('results/generation_2/robustness/arma_garch_beta_summary.csv', index=False)
# Entropy vs beta plot
plt.figure(figsize=(6,4))
plt.plot(summary['beta'], summary['entropy'], marker='o')
plt.xlabel('β')
plt.ylabel('Shannon entropy H')
plt.title('Shannon entropy of ARMA-GARCH order distribution vs β')
plt.grid(True)
plt.tight_layout()
plt.savefig('results/generation_2/robustness/entropy_vs_beta.png', dpi=150)
plt.close()
print('\n=== Зведена таблиця ===')
print(summary)
print('Гістограми, таблиця та графік ентропії збережені у results/generation_2/robustness/')
