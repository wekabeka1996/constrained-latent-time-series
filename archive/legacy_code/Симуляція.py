# Прототип модуля генерації нових гіпотез \theta' за допомогою VAE (модифікована версія з multi-head rule-based декодуванням)
# Архітектура підтримує комбіновані ARMA-GARCH моделі без явного argmax-декодування

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# -----------------------------
# 1. Вектор ознак z_theta
# -----------------------------

D_STRUCT = 10  # one-hot типу моделі, порядки lagів, режимність тощо
D_PARAM = 20   # уніфікована кількість параметрів (наприклад, AR/GARCH/SDE)
D_STAT = 10    # ознаки, такі як автокореляція, куртозис тощо
D_LATENT = 8   # розмірність латентного простору VAE
D_INPUT = D_STRUCT + D_PARAM + D_STAT
MODEL_TYPES = ["ARMA", "GARCH", "ARMA-GARCH"]
MAX_LAG_ORDER = 5

# -----------------------------
# 2. Архітектура VAE
# -----------------------------

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(D_INPUT, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, D_LATENT)
        self.fc_logvar = nn.Linear(32, D_LATENT)

        self.decoder = nn.Sequential(
            nn.Linear(D_LATENT, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, D_INPUT),
            nn.Tanh()  # [-1,1] нормалізований output
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# -----------------------------
# 3. Loss-функція VAE
# -----------------------------

def vae_loss(x, x_hat, mu, logvar):
    recon = nn.functional.mse_loss(x_hat, x, reduction='mean')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + kl

# -----------------------------
# 4. Генерація нової гіпотези
# -----------------------------

def sample_new_hypothesis(model: VAE, n=1):
    with torch.no_grad():
        z = torch.randn(n, D_LATENT)
        z_theta = model.decode(z).cpu().numpy()
        return z_theta

# -----------------------------
# 5. Rule-based декодування структури
# -----------------------------

# Оновлений код: VAE-генератор і коректне розпізнання структур ARMA, GARCH та ARMA-GARCH
# Ключова частина: покращена логіка decode_discrete_structure з виходами arma/garch з буфером epsilon


def decode_discrete_structure(v_struct_decoded_normalized, epsilon=0.05):
    f_prime = {}
    current_idx = 0

    type_logits = (v_struct_decoded_normalized[current_idx : current_idx + 2] + 1) / 2
    arma_score, garch_score = type_logits[0], type_logits[1]
    current_idx += 2

    if arma_score > 0.5 + epsilon and garch_score > 0.5 + epsilon:
        f_prime['type'] = "ARMA-GARCH"
    elif arma_score > 0.5 + epsilon and garch_score <= 0.5:
        f_prime['type'] = "ARMA"
    elif garch_score > 0.5 + epsilon and arma_score <= 0.5:
        f_prime['type'] = "GARCH"
    else:
        # Якщо обидва слабкі або неоднозначні, вибираємо домінуючий
        if arma_score >= garch_score:
            f_prime['type'] = "ARMA"
        else:
            f_prime['type'] = "GARCH"

    if f_prime['type'] in ["ARMA", "ARMA-GARCH"]:
        p_logit = v_struct_decoded_normalized[current_idx]
        f_prime['p_order'] = round(((p_logit + 1) / 2) * MAX_LAG_ORDER)
        current_idx += 1
        q_logit = v_struct_decoded_normalized[current_idx]
        f_prime['q_order'] = round(((q_logit + 1) / 2) * MAX_LAG_ORDER)
        current_idx += 1

    if f_prime['type'] in ["GARCH", "ARMA-GARCH"]:
        r_logit = v_struct_decoded_normalized[current_idx]
        f_prime['r_order'] = round(((r_logit + 1) / 2) * MAX_LAG_ORDER)
        current_idx += 1
        s_logit = v_struct_decoded_normalized[current_idx]
        f_prime['s_order'] = round(((s_logit + 1) / 2) * MAX_LAG_ORDER)
        current_idx += 1

    return f_prime


# -----------------------------
# 6. Розширена валідація
# -----------------------------

def validate_theta_advanced(f_prime_struct, decoded_params, decoded_stats):
    is_valid = True

    if f_prime_struct['type'] in ["ARMA", "ARMA-GARCH"]:
        p = f_prime_struct.get('p_order', 0)
        q = f_prime_struct.get('q_order', 0)
        if p == 0 and q == 0:
            return False

    if f_prime_struct['type'] in ["GARCH", "ARMA-GARCH"]:
        r = f_prime_struct.get('r_order', 0)
        s = f_prime_struct.get('s_order', 0)
        if r == 0 and s == 0:
            return False

        omega = decoded_params[0]
        alphas = decoded_params[1:1 + r]
        betas = decoded_params[1 + r : 1 + r + s]

        if omega <= 0 or any(a < 0 for a in alphas) or any(b < 0 for b in betas):
            return False
        if sum(alphas) + sum(betas) >= 1:
            return False

    return is_valid

# -----------------------------
# 7. Обгортка генерації
# -----------------------------

def validate_theta(z_theta: np.ndarray):
    valid = []
    for z in z_theta:
        param_section = z[D_STRUCT:D_STRUCT + D_PARAM]
        struct_section = z[:D_STRUCT]
        stat_section = z[-D_STAT:]
        f_struct = decode_discrete_structure(struct_section)
        if validate_theta_advanced(f_struct, param_section, stat_section):
            valid.append(z)
    return np.array(valid)

# -----------------------------
# 8. Повна процедура
# -----------------------------

def generate_valid_thetas(model, n_trials=100):
    candidates = sample_new_hypothesis(model, n=n_trials)
    return validate_theta(candidates)

# -----------------------------
# 9. Dataset
# -----------------------------

class HypothesisDataset(Dataset):
    def __init__(self, Z):
        self.Z = torch.tensor(Z, dtype=torch.float32)

    def __len__(self):
        return self.Z.shape[0]

    def __getitem__(self, idx):
        return self.Z[idx]

# -----------------------------
# 10. Навчання VAE
# -----------------------------

def train_vae(model, dataloader, epochs=100, lr=1e-3):
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for x in dataloader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            loss = vae_loss(x, x_hat, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# -----------------------------
# 11. Повний запуск на CPU
# -----------------------------

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    def generate_structured_theta(model_type: str, n: int):
        Z = []
        for _ in range(n):
            z = np.zeros(D_INPUT)
            struct = np.zeros(D_STRUCT)
            params = np.zeros(D_PARAM)
            stats = np.random.uniform(-1, 1, size=D_STAT)

            if model_type == "ARMA":
                struct[0] = 1  # ARMA indicator
                struct[2] = np.random.uniform(-1, 1)  # p_order
                struct[3] = np.random.uniform(-1, 1)  # q_order
                p = round(((struct[2] + 1)/2)*MAX_LAG_ORDER)
                phi = np.random.uniform(-0.8, 0.8, size=p)
                params[:p] = phi

            elif model_type == "GARCH":
                struct[1] = 1  # GARCH indicator
                struct[4] = np.random.uniform(-1, 1)  # r_order
                struct[5] = np.random.uniform(-1, 1)  # s_order
                r = round(((struct[4] + 1)/2)*MAX_LAG_ORDER)
                s = round(((struct[5] + 1)/2)*MAX_LAG_ORDER)
                omega = np.random.uniform(0.1, 0.5)
                alpha = np.random.uniform(0, 0.3, size=r)
                beta = np.random.uniform(0, 0.4 - sum(alpha), size=s)
                params[0] = omega
                params[1:1+r] = alpha
                params[1+r:1+r+s] = beta

            z[:D_STRUCT] = struct
            z[D_STRUCT:D_STRUCT+D_PARAM] = params
            z[-D_STAT:] = stats
            Z.append(z)
        return np.array(Z)

    Z_ar = generate_structured_theta("ARMA", 250)
    Z_garch = generate_structured_theta("GARCH", 250)
    Z_train = np.concatenate([Z_ar, Z_garch], axis=0)

    dataset = HypothesisDataset(Z_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = VAE()
    train_vae(model, dataloader, epochs=100)

    # Генерація нових гіпотез та перевірка
    generated = generate_valid_thetas(model, n_trials=1000)
    print(f"Згенеровано валідних гіпотез: {len(generated)} з 1000")

    # Збереження результатів
    np.save("generated_valid_thetas.npy", generated)
