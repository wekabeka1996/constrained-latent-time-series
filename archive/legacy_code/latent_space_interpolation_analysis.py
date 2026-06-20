import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# --- Константи та клас VAE ---
D_STRUCT = 10
D_PARAM = 20
D_STAT = 10
D_LATENT = 8
D_INPUT = D_STRUCT + D_PARAM + D_STAT
MAX_LAG_ORDER = 5
MODEL_TYPES_LIST = ["ARMA", "GARCH", "ARMA-GARCH", "UNKNOWN"]

class VAE(nn.Module):
    def __init__(self, d_input=D_INPUT, d_latent=D_LATENT):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_input, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, d_latent)
        self.fc_logvar = nn.Linear(32, d_latent)
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, d_input),
            nn.Tanh()
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

def decode_discrete_structure(v_struct_decoded_normalized, epsilon=0.05):
    f_prime = {'type': "UNKNOWN", 'p_order': 0, 'q_order': 0, 'r_order': 0, 's_order': 0}
    current_idx = 0
    if len(v_struct_decoded_normalized) < 2:
        return f_prime
    score_components = (v_struct_decoded_normalized[current_idx : min(current_idx + 2, len(v_struct_decoded_normalized))] + 1) / 2
    arma_score = score_components[0].item() if hasattr(score_components[0], 'item') else score_components[0]
    garch_score = 0.0
    if len(score_components) > 1:
        garch_score = score_components[1].item() if hasattr(score_components[1], 'item') else score_components[1]
    current_idx += 2
    is_arma_strong = arma_score > 0.5 + epsilon
    is_garch_strong = garch_score > 0.5 + epsilon
    is_arma_weak = arma_score > 0.5
    is_garch_weak = garch_score > 0.5
    if is_arma_strong and is_garch_strong:
        f_prime['type'] = "ARMA-GARCH"
    elif is_arma_strong and not is_garch_weak:
        f_prime['type'] = "ARMA"
    elif is_garch_strong and not is_arma_weak:
        f_prime['type'] = "GARCH"
    elif is_arma_weak and is_garch_weak:
        f_prime['type'] = "ARMA-GARCH"
    elif is_arma_weak:
        f_prime['type'] = "ARMA"
    elif is_garch_weak:
        f_prime['type'] = "GARCH"
    else:
        f_prime['type'] = "UNKNOWN"
    order_indices = [current_idx, current_idx + 1, current_idx + 2, current_idx + 3]
    if f_prime['type'] in ["ARMA", "ARMA-GARCH"]:
        if order_indices[1] < len(v_struct_decoded_normalized):
            p_logit = v_struct_decoded_normalized[order_indices[0]]
            q_logit = v_struct_decoded_normalized[order_indices[1]]
            f_prime['p_order'] = round(((p_logit.item() + 1) / 2) * MAX_LAG_ORDER) if hasattr(p_logit, 'item') else round(((p_logit + 1) / 2) * MAX_LAG_ORDER)
            f_prime['q_order'] = round(((q_logit.item() + 1) / 2) * MAX_LAG_ORDER) if hasattr(q_logit, 'item') else round(((q_logit + 1) / 2) * MAX_LAG_ORDER)
    if f_prime['type'] in ["GARCH", "ARMA-GARCH"]:
        if order_indices[3] < len(v_struct_decoded_normalized):
            r_logit = v_struct_decoded_normalized[order_indices[2]]
            s_logit = v_struct_decoded_normalized[order_indices[3]]
            f_prime['r_order'] = round(((r_logit.item() + 1) / 2) * MAX_LAG_ORDER) if hasattr(r_logit, 'item') else round(((r_logit + 1) / 2) * MAX_LAG_ORDER)
            f_prime['s_order'] = round(((s_logit.item() + 1) / 2) * MAX_LAG_ORDER) if hasattr(s_logit, 'item') else round(((s_logit + 1) / 2) * MAX_LAG_ORDER)
    return f_prime

def validate_model_params(decoded_struct_info, params_vector):
    model_type = decoded_struct_info.get("type", "UNKNOWN")
    p = decoded_struct_info.get('p_order', 0)
    q = decoded_struct_info.get('q_order', 0)
    r = decoded_struct_info.get('r_order', 0)
    s = decoded_struct_info.get('s_order', 0)
    if model_type == "UNKNOWN": return False
    is_arma_struct_active = (p > 0 or q > 0)
    is_garch_struct_active = (r > 0 or s > 0)
    if model_type == "ARMA" and not is_arma_struct_active: return False
    if model_type == "GARCH" and not is_garch_struct_active: return False
    if model_type == "ARMA-GARCH" and not (is_arma_struct_active and is_garch_struct_active): return False
    return True

def main_interpolation_analysis(num_pairs=5, random_seed=42):
    device = torch.device('cpu')
    vae_model = VAE(d_input=D_INPUT, d_latent=D_LATENT).to(device)
    model_weights_path = 'vae_model_weights.pth'
    if os.path.exists(model_weights_path):
        try:
            vae_model.load_state_dict(torch.load(model_weights_path, map_location=device))
            print(f"Навчені ваги для VAE успішно завантажені з '{model_weights_path}'.")
        except Exception as e:
            print(f"ПОПЕРЕДЖЕННЯ: Помилка при завантаженні вагів з '{model_weights_path}': {e}. Модель буде використовувати випадкову ініціалізацію.")
    else:
        print(f"ПОПЕРЕДЖЕННЯ: Файл з вагами '{model_weights_path}' не знайдено. Модель ініціалізована випадковими вагами, що зробить аналіз нерепрезентативним.")
    vae_model.eval()
    np.random.seed(random_seed)
    from src.io_utils import require_file
    try:
        ar_path = require_file("z_train_ar.npy", "Latent vectors for AR models needed for interpolation. Provide the real artifact or generate it through the documented canonical pipeline. Do not use dummy fallback arrays.")
        garch_path = require_file("z_train_garch.npy", "Latent vectors for GARCH models needed for interpolation. Provide the real artifact or generate it through the documented canonical pipeline. Do not use dummy fallback arrays.")
        
        z_train_ar_all = np.load(ar_path)
        z_train_garch_all = np.load(garch_path)
        
        if z_train_ar_all.ndim == 1: 
            z_train_ar_all = z_train_ar_all.reshape(1, -1)
        if z_train_garch_all.ndim == 1: 
            z_train_garch_all = z_train_garch_all.reshape(1, -1)
            
        if z_train_ar_all.size == 0 or z_train_garch_all.size == 0:
            raise ValueError("z_train_ar.npy or z_train_garch.npy is empty. Cannot continue interpolation.")
        if z_train_ar_all.shape[1] != D_LATENT or z_train_garch_all.shape[1] != D_LATENT:
            raise ValueError(f"Wrong latent space dimension. Expected {D_LATENT}, got {z_train_ar_all.shape[1]} and {z_train_garch_all.shape[1]}. Cannot continue interpolation.")
            
    except (FileNotFoundError, ValueError) as e:
        print(f"ПОМИЛКА: {e}")
        raise

    lambdas = np.linspace(0, 1, 21)
    print(f"\nПочаток аналізу інтерполяції для {num_pairs} випадкових пар...")
    for pair_idx in range(num_pairs):
        idx_ar = np.random.choice(z_train_ar_all.shape[0])
        idx_garch = np.random.choice(z_train_garch_all.shape[0])
        z_AR_repr = torch.tensor(z_train_ar_all[idx_ar:idx_ar+1], dtype=torch.float32).to(device)
        z_GARCH_repr = torch.tensor(z_train_garch_all[idx_garch:idx_garch+1], dtype=torch.float32).to(device)
        interpolation_results = []
        csv_rows = []
        z_interp_list = []
        print(f"\nПара {pair_idx+1}: AR idx={idx_ar}, GARCH idx={idx_garch}")
        for lmbda_val in lambdas:
            z_interpolated = (1 - lmbda_val) * z_AR_repr + lmbda_val * z_GARCH_repr
            z_interp_list.append(z_interpolated.cpu().numpy().flatten())
            with torch.no_grad():
                theta_prime_full_vector = vae_model.decode(z_interpolated).cpu().numpy().flatten()
            struct_part_vae_output = theta_prime_full_vector[:D_STRUCT]
            params_part_vae_output = theta_prime_full_vector[D_STRUCT : D_STRUCT + D_PARAM]
            decoded_structure = decode_discrete_structure(struct_part_vae_output, epsilon=0.05)
            is_valid_params = validate_model_params(decoded_structure, params_part_vae_output)
            phi1 = params_part_vae_output[0] if len(params_part_vae_output) > 0 else None
            phi2 = params_part_vae_output[1] if len(params_part_vae_output) > 1 else None
            theta1 = params_part_vae_output[2] if len(params_part_vae_output) > 2 else None
            theta2 = params_part_vae_output[3] if len(params_part_vae_output) > 3 else None
            theta3 = params_part_vae_output[4] if len(params_part_vae_output) > 4 else None
            current_result = {
                'lambda': lmbda_val,
                'type': decoded_structure.get('type','UNKNOWN'),
                'p': decoded_structure.get('p_order',0),
                'q': decoded_structure.get('q_order',0),
                'r': decoded_structure.get('r_order',0),
                's': decoded_structure.get('s_order',0),
                'is_param_valid': is_valid_params,
                'phi1': phi1,
                'phi2': phi2,
                'theta1': theta1,
                'theta2': theta2,
                'theta3': theta3
            }
            interpolation_results.append(current_result)
            csv_rows.append([lmbda_val, current_result['type'], current_result['p'], current_result['q'], current_result['r'], current_result['s'], is_valid_params, phi1, phi2, theta1, theta2, theta3])
        # Збереження інтерполяційних латентних векторів
        z_interp_array = np.stack(z_interp_list, axis=0)
        np.save(f"z_interp_{pair_idx+1}.npy", z_interp_array)
        print(f"Інтерполяційні латентні вектори збережено у 'z_interp_{pair_idx+1}.npy'")
        # Збереження у CSV
        with open(f"interpolation_detailed_{pair_idx+1}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["lambda", "type", "p", "q", "r", "s", "is_param_valid", "phi1", "phi2", "theta1", "theta2", "theta3"])
            writer.writerows(csv_rows)
        print(f"Детальні результати інтерполяції збережено у 'interpolation_detailed_{pair_idx+1}.csv'")
    print(f"\nАналіз для {num_pairs} пар завершено. Всі траєкторії та таблиці збережено.")

if __name__ == '__main__':
    main_interpolation_analysis(num_pairs=5)
