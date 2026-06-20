import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset

# --- Базові константи ---
D_STRUCT = 10
D_PARAM = 20
D_STAT = 10
D_INPUT = D_STRUCT + D_PARAM + D_STAT

# --- VAE архітектури ---
class VAE_Deeper(nn.Module):
    def __init__(self, d_input=D_INPUT, d_latent=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_input, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, d_latent)
        self.fc_logvar = nn.Linear(32, d_latent)
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, d_input), nn.Tanh()
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

class VAE_Wider(nn.Module):
    def __init__(self, d_input=D_INPUT, d_latent=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_input, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, d_latent)
        self.fc_logvar = nn.Linear(64, d_latent)
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, d_input), nn.Tanh()
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

class VAE_LatentDim(nn.Module):
    def __init__(self, d_input=D_INPUT, d_latent=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_input, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, d_latent)
        self.fc_logvar = nn.Linear(32, d_latent)
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, d_input), nn.Tanh()
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

def train_vae(model_class, d_latent=8, seed=42, epochs=100, save_path=None):
    # Завантаження даних
    Z_train = np.load('Z_train.npy')
    dataset = torch.tensor(Z_train, dtype=torch.float32)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = model_class(d_latent=d_latent)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    def vae_loss(x, x_hat, mu, logvar):
        recon = nn.functional.mse_loss(x_hat, x, reduction='mean')
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon + kl
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x in dataloader:
            x = x.to(device)
            opt.zero_grad()
            x_hat, mu, logvar = model(x)
            loss = vae_loss(x, x_hat, mu, logvar)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(dataset)
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Ваги моделі збережено у {save_path}")
    return model

# Приклад запуску для трьох архітектур:
if __name__ == '__main__':
    print("=== Тренування VAE_Deeper ===")
    train_vae(VAE_Deeper, d_latent=8, seed=42, epochs=100, save_path='models/vae_deeper.pth')
    print("=== Тренування VAE_Wider ===")
    train_vae(VAE_Wider, d_latent=8, seed=42, epochs=100, save_path='models/vae_wider.pth')
    print("=== Тренування VAE_LatentDim (d_latent=16) ===")
    train_vae(VAE_LatentDim, d_latent=16, seed=42, epochs=100, save_path='models/vae_latent16.pth')