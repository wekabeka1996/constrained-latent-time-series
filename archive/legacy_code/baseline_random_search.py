import numpy as np
from src.Симуляція_2 import decode_discrete_structure
from collections import Counter
import random

D_STRUCT = 10
MAX_LAG_ORDER = 5

# Робастна валідація (копія з analyze_beta_diversity.py)
def validate_arma_garch(x, epsilon=0.05):
    struct_vec = x[:8] if isinstance(x, (np.ndarray, list)) else x
    decoded_struct = decode_discrete_structure(struct_vec, epsilon=epsilon)
    model_type = decoded_struct.get("type", "UNKNOWN")
    p = decoded_struct.get('p_order', 0)
    q = decoded_struct.get('q_order', 0)
    r = decoded_struct.get('r_order', 0)
    s = decoded_struct.get('s_order', 0)
    is_arma_struct_active = (p > 0 or q > 0)
    is_garch_struct_active = (r > 0 or s > 0)
    if model_type == "ARMA-GARCH" and is_arma_struct_active and is_garch_struct_active:
        return True, (p, q, r, s)
    return False, (p, q, r, s)

def random_struct_vector():
    # Перші 2: активація ARMA/GARCH [-1,1]
    arma_garch = np.random.uniform(-1, 1, 2)
    # Далі 4: порядки p,q,r,s (логіти) [-1,1]
    pqrs_logits = np.random.uniform(-1, 1, 4)
    # Решта - шум
    rest = np.random.uniform(-1, 1, D_STRUCT - 6)
    return np.concatenate([arma_garch, pqrs_logits, rest])

def random_theta_vector():
    struct = random_struct_vector()
    # Параметрична та статистична частини (заповнюємо шумом)
    param = np.random.uniform(-1, 1, 20)
    stat = np.random.uniform(-1, 1, 10)
    return np.concatenate([struct, param, stat])

def generate_valid_arma_garch(n_valid=1000, target_order=None):
    valid_thetas = []
    orders = []
    attempts = 0
    while len(valid_thetas) < n_valid:
        x = random_theta_vector()
        is_valid, pqrs = validate_arma_garch(x)
        attempts += 1
        if is_valid:
            if target_order is not None and pqrs[:2] != target_order:
                continue
            valid_thetas.append(x)
            orders.append(pqrs)
    return np.array(valid_thetas), orders, attempts

def main():
    # 1. Benchmark diversity
    n = 1000
    thetas, orders, attempts = generate_valid_arma_garch(n_valid=n)
    print(f"Baseline: {n} valid ARMA-GARCH, attempts: {attempts}, efficiency: {n/attempts:.3f}")
    order_counts = Counter(orders)
    probs = np.array(list(order_counts.values())) / n
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    print(f"Shannon entropy: {entropy:.3f}")
    print(f"Top-5 orders: {order_counts.most_common(5)}")
    np.save("results/generation_2/robustness/baseline_valid_thetas.npy", thetas)
    # 2. Targeted ARMA(2,2)
    target = (2,2)
    n_target = 100
    _, _, attempts_target = generate_valid_arma_garch(n_valid=n_target, target_order=target)
    print(f"Baseline: {n_target} valid ARMA({target[0]},{target[1]}) models, attempts: {attempts_target}, efficiency: {n_target/attempts_target:.3f}")

if __name__ == '__main__':
    main()
