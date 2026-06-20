import numpy as np
import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from generate_latent_vectors import VAE, D_INPUT, D_LATENT
from Симуляція_2 import decode_discrete_structure

def slerp(val, low, high):
    """Spherical linear interpolation."""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    if omega < 1e-6:
        return (1.0 - val) * low + val * high
    so = np.sin(omega)
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high

def validate_arma_garch(x, epsilon=0.05):
    """
    Робастна валідація: декодує структуру та перевіряє валідність як у Фазі I.
    """
    # x: сирий вектор (np.ndarray), не dict
    struct_vec = x[:8] if isinstance(x, (np.ndarray, list)) else x
    decoded_struct = decode_discrete_structure(struct_vec, epsilon=epsilon)
    model_type = decoded_struct.get("type", "UNKNOWN")
    p = decoded_struct.get('p_order', 0)
    q = decoded_struct.get('q_order', 0)
    r = decoded_struct.get('r_order', 0)
    s = decoded_struct.get('s_order', 0)
    is_arma_struct_active = (p > 0 or q > 0)
    is_garch_struct_active = (r > 0 or s > 0)
    if model_type == "ARMA" and is_arma_struct_active:
        return True
    if model_type == "GARCH" and is_garch_struct_active:
        return True
    if model_type == "ARMA-GARCH" and is_arma_struct_active and is_garch_struct_active:
        return True
    return False

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 1. z_start: mean AR latent
    z_train_ar = np.load('z_train_ar.npy')
    z_start = np.mean(z_train_ar, axis=0)
    # 2. Load VAE (beta=5.0)
    vae = VAE(d_input=D_INPUT, d_latent=D_LATENT).to(device)
    vae.load_state_dict(torch.load('models/vae_beta50.pth', map_location=device))
    vae.eval()
    # 3. Generate valid ARMA-GARCH, encode to get z_end
    found = False
    for _ in range(1000):
        z = torch.randn(D_LATENT).to(device)
        with torch.no_grad():
            x_gen = vae.decode(z).cpu().numpy()
        if validate_arma_garch(x_gen):
            z_end = vae.encode(torch.tensor(x_gen, dtype=torch.float32).to(device).unsqueeze(0))[0][0].detach().cpu().numpy()
            found = True
            break
    if not found:
        print('Не вдалося знайти валідну ARMA-GARCH модель для z_end')
        return
    # 4. SLERP trajectory
    n_points = 21
    slerp_points = np.array([slerp(lam, z_start, z_end) for lam in np.linspace(0, 1, n_points)])
    # 5. Decode and analyze
    results = []
    for i, z in enumerate(slerp_points):
        z_torch = torch.tensor(z, dtype=torch.float32).to(device)
        with torch.no_grad():
            x_gen = vae.decode(z_torch).cpu().numpy()
        valid = validate_arma_garch(x_gen)
        decoded = decode_discrete_structure(x_gen)
        model_type = decoded.get('model_type', 'unknown')
        pqrs = tuple(decoded.get(k, 0) for k in ['p_order','q_order','r_order','s_order'])
        results.append({'lambda': i/(n_points-1), 'valid': valid, 'model_type': model_type, 'pqrs': pqrs})
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('results/generation_2/robustness/slerp_trajectory_analysis.csv', index=False)
    print(df)
    print('Результати збережено у results/generation_2/robustness/slerp_trajectory_analysis.csv')

if __name__ == '__main__':
    main()
