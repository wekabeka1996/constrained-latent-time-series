import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

# --- Константи ---
D_STRUCT = 10
D_PARAM = 20
D_STAT = 10
D_LATENT = 8
D_INPUT = D_STRUCT + D_PARAM + D_STAT

# --- Клас VAE ---
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

def vae_loss(x, x_hat, mu, logvar, beta=1.0):
    recon = nn.functional.mse_loss(x_hat, x, reduction='mean')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + beta * kl

class HypothesisDataset(Dataset):
    def __init__(self, Z):
        self.Z = torch.tensor(Z, dtype=torch.float32)
    def __len__(self):
        return self.Z.shape[0]
    def __getitem__(self, idx):
        return self.Z[idx]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='Z_train.npy')
    parser.add_argument('--labels_path', type=str, default='train_labels.npy')
    parser.add_argument('--save_path', type=str, default='vae_model_weights.pth')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--beta', type=float, default=1.0)
    args = parser.parse_args()
    # Завантаження даних
    Z_train = np.load(args.train_path)
    dataset = HypothesisDataset(Z_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = VAE()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for x in dataloader:
            x = x.to(device)
            opt.zero_grad()
            x_hat, mu, logvar = model(x)
            loss = vae_loss(x, x_hat, mu, logvar, beta=args.beta)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(dataset)
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f}")
    torch.save(model.state_dict(), args.save_path)
    print(f"Ваги навченої моделі VAE збережено у '{args.save_path}'")

if __name__ == '__main__':
    main()
