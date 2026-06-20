import sys
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import numpy as np
import os

# -----------------------------
# 1. Визначення архітектури VAE
# -----------------------------
D_STRUCT = 10
D_PARAM = 20
D_STAT = 10
D_LATENT = 8
D_INPUT = D_STRUCT + D_PARAM + D_STAT

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

from src.io_utils import require_file

# -----------------------------
# 2. Основна логіка скрипту
# -----------------------------
def main():
    # Параметри моделі
    d_input_for_vae = D_INPUT 
    d_latent_for_vae = D_LATENT

    # Налаштування пристрою
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Використовується пристрій: {device}")

    # Ініціалізація моделі VAE
    print("\nІніціалізація моделі VAE...")
    vae_model = VAE(d_input=d_input_for_vae, d_latent=d_latent_for_vae).to(device)
    
    # Завантаження навчених вагів
    model_weights_path = 'vae_model_weights.pth'
    if os.path.exists(model_weights_path):
        try:
            vae_model.load_state_dict(torch.load(model_weights_path, map_location=device))
            print(f"Навчені ваги для VAE успішно завантажені з '{model_weights_path}'.")
        except Exception as e:
            print(f"ПОПЕРЕДЖЕННЯ: Помилка при завантаженні вагів з '{model_weights_path}': {e}.")
            print("Модель VAE буде використовувати випадкову ініціалізацію.")
    else:
        print(f"ПОПЕРЕДЖЕННЯ: Файл з вагами '{model_weights_path}' не знайдено.")
        print("Модель VAE ініціалізована випадковими вагами.")
        print("Для значущих результатів візуалізації завантажте навчені ваги.")

    vae_model.eval()  # Режим оцінки

    # Завантаження вхідних даних
    print("\nЗавантаження вхідних даних (векторів θ)...")
    try:
        z_train_path = require_file("Z_train.npy", "Training parameter vectors needed for encoding. Provide the real artifact or generate it through the documented canonical pipeline. Do not use dummy fallback arrays.")
        train_labels_path = require_file("train_labels.npy", "Training class labels needed to group latent vectors. Provide the real artifact or generate it through the documented canonical pipeline. Do not use dummy fallback arrays.")
        generated_thetas_path = require_file("generated_valid_thetas.npy", "Generated valid theta vectors from VAE simulation to encode. Provide the real artifact or generate it through the documented canonical pipeline. Do not use dummy fallback arrays.")

        Z_train_all = np.load(z_train_path)
        train_labels_all = np.load(train_labels_path)
        generated_thetas_all = np.load(generated_thetas_path)
        print("Вхідні дані успішно завантажено.")
        
        # Перевірка розмірності
        if Z_train_all.shape[1] != d_input_for_vae:
            raise ValueError(f"Z_train.npy має {Z_train_all.shape[1]} колонок, очікується {d_input_for_vae}")
        if generated_thetas_all.shape[1] != d_input_for_vae:
            raise ValueError(f"generated_valid_thetas.npy має {generated_thetas_all.shape[1]} колонок, очікується {d_input_for_vae}")

    except FileNotFoundError as e:
        print(f"ПОМИЛКА: Файл не знайдено: {e}")
        raise
    except ValueError as ve:
        print(f"ПОМИЛКА формату даних: {ve}")
        raise
    except Exception as e:
        print(f"Невідома помилка завантаження вхідних файлів: {e}")
        raise

    # Перетворення в тензори PyTorch
    Z_train_tensor = torch.tensor(Z_train_all, dtype=torch.float32).to(device)
    generated_thetas_tensor = torch.tensor(generated_thetas_all, dtype=torch.float32).to(device)

    # Обробка через енкодер VAE
    print("\nОбробка даних через енкодер VAE для отримання векторів μ...")
    
    with torch.no_grad():
        # Обробка тренувальних даних
        if Z_train_tensor.nelement() > 0:
            mu_train, _ = vae_model.encode(Z_train_tensor)
            latent_vectors_train_mu = mu_train.cpu().numpy()
        else:
            latent_vectors_train_mu = np.empty((0, d_latent_for_vae))
        
        # Обробка згенерованих даних
        if generated_thetas_tensor.nelement() > 0:
            mu_generated, _ = vae_model.encode(generated_thetas_tensor)
            latent_vectors_generated_mu = mu_generated.cpu().numpy()
        else:
            latent_vectors_generated_mu = np.empty((0, d_latent_for_vae))
    
    print("Дані успішно оброблені енкодером.")

    # Розділення тренувальних латентних векторів
    print("\nРозділення тренувальних латентних векторів за мітками...")
    z_train_ar_list = []
    z_train_garch_list = []

    if latent_vectors_train_mu.shape[0] > 0 and latent_vectors_train_mu.shape[0] == len(train_labels_all):
        for i, label in enumerate(train_labels_all):
            if label.upper() == 'AR':
                z_train_ar_list.append(latent_vectors_train_mu[i])
            elif label.upper() == 'GARCH':
                z_train_garch_list.append(latent_vectors_train_mu[i])
        print("Тренувальні латентні вектори розділено.")
    else:
        print(f"ПОПЕРЕДЖЕННЯ: Кількість латентних векторів ({latent_vectors_train_mu.shape[0]}) "
              f"не відповідає кількості міток ({len(train_labels_all)}).")

    z_train_ar = np.array(z_train_ar_list) if z_train_ar_list else np.empty((0, d_latent_for_vae))
    z_train_garch = np.array(z_train_garch_list) if z_train_garch_list else np.empty((0, d_latent_for_vae))

    # Збереження латентних векторів
    print("\nЗбереження латентних векторів...")
    
    if z_train_ar.size > 0:
        np.save("z_train_ar.npy", z_train_ar)
        print(f"Файл z_train_ar.npy ({z_train_ar.shape}) збережено.")
        # Збереження у CSV
        np.savetxt("z_train_ar.csv", z_train_ar, delimiter=",", fmt="%.8f")
        print(f"Файл z_train_ar.csv ({z_train_ar.shape}) збережено.")
    else:
        print("Масив z_train_ar порожній, файл не створено.")

    if z_train_garch.size > 0:
        np.save("z_train_garch.npy", z_train_garch)
        print(f"Файл z_train_garch.npy ({z_train_garch.shape}) збережено.")
        # Збереження у CSV
        np.savetxt("z_train_garch.csv", z_train_garch, delimiter=",", fmt="%.8f")
        print(f"Файл z_train_garch.csv ({z_train_garch.shape}) збережено.")
    else:
        print("Масив z_train_garch порожній, файл не створено.")

    if latent_vectors_generated_mu.size > 0:
        np.save("z_generated.npy", latent_vectors_generated_mu)
        print(f"Файл z_generated.npy ({latent_vectors_generated_mu.shape}) збережено.")
        # Збереження у CSV
        np.savetxt("z_generated.csv", latent_vectors_generated_mu, delimiter=",", fmt="%.8f")
        print(f"Файл z_generated.csv ({latent_vectors_generated_mu.shape}) збережено.")
    else:
        print("Масив z_generated порожній, файл не створено.")

    # Збереження міток у CSV
    if len(train_labels_all) > 0:
        np.savetxt("train_labels.csv", train_labels_all, fmt="%s")
        print(f"Файл train_labels.csv ({len(train_labels_all)}) збережено.")
    else:
        print("Масив train_labels порожній, файл не створено.")
        
    print("\nРоботу скрипту generate_latent_vectors.py завершено.")
    print("=" * 50)

if __name__ == '__main__':
    main()
