import numpy as np
from collections import Counter

# --- Константи, що визначають структуру вектора z_theta ---
D_STRUCT = 10  # Розмірність структурної частини вектора z_theta
MAX_LAG_ORDER = 5  # Максимальний порядок для p, q, r, s

# --- Функція декодування дискретної структури моделі ---
def decode_discrete_structure(v_struct_decoded_normalized, epsilon=0.05):
    """
    Декодує структурну частину вектора z_theta.
    v_struct_decoded_normalized: частина вектора z_theta, що відповідає за структуру,
                                 очікуються значення в діапазоні [-1, 1] (вихід Tanh VAE).
    epsilon: буфер для порогів активації.
    """
    f_prime = {}
    current_idx = 0

    # Перші два елементи v_struct_decoded_normalized визначають активацію ARMA та GARCH
    # Перетворюємо з [-1, 1] в [0, 1] для score
    score_components = (v_struct_decoded_normalized[current_idx : current_idx + 2] + 1) / 2
    arma_score = score_components[0].item() if hasattr(score_components[0], 'item') else score_components[0]
    garch_score = score_components[1].item() if hasattr(score_components[1], 'item') else score_components[1]
    current_idx += 2

    # Визначення типу моделі на основі score та epsilon
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
    elif arma_score > garch_score:
        f_prime['type'] = "ARMA"
    else:
        f_prime['type'] = "GARCH"

    pqrs_indices = [current_idx, current_idx + 1, current_idx + 2, current_idx + 3]

    # Декодування порядків p, q (для ARMA частини)
    if pqrs_indices[1] < len(v_struct_decoded_normalized):
        p_logit = v_struct_decoded_normalized[pqrs_indices[0]]
        q_logit = v_struct_decoded_normalized[pqrs_indices[1]]
        f_prime['p_order'] = round(((p_logit.item() + 1) / 2) * MAX_LAG_ORDER) if hasattr(p_logit, 'item') else round(((p_logit + 1) / 2) * MAX_LAG_ORDER)
        f_prime['q_order'] = round(((q_logit.item() + 1) / 2) * MAX_LAG_ORDER) if hasattr(q_logit, 'item') else round(((q_logit + 1) / 2) * MAX_LAG_ORDER)
    else:
        f_prime['p_order'] = 0
        f_prime['q_order'] = 0

    # Декодування порядків r, s (для GARCH частини)
    if pqrs_indices[3] < len(v_struct_decoded_normalized):
        r_logit = v_struct_decoded_normalized[pqrs_indices[2]]
        s_logit = v_struct_decoded_normalized[pqrs_indices[3]]
        f_prime['r_order'] = round(((r_logit.item() + 1) / 2) * MAX_LAG_ORDER) if hasattr(r_logit, 'item') else round(((r_logit + 1) / 2) * MAX_LAG_ORDER)
        f_prime['s_order'] = round(((s_logit.item() + 1) / 2) * MAX_LAG_ORDER) if hasattr(s_logit, 'item') else round(((s_logit + 1) / 2) * MAX_LAG_ORDER)
    else:
        f_prime['r_order'] = 0
        f_prime['s_order'] = 0
        
    return f_prime


def main():
    from src.io_utils import require_file
    
    # --- Завантаження згенерованих валідних гіпотез (очікується .npy файл) ---
    try:
        thetas_path = require_file(
            "generated_valid_thetas.npy", 
            "Generated valid theta vectors from VAE simulation. Provide the real artifact or generate it through the documented canonical pipeline. Do not use dummy fallback arrays."
        )
        Z_all_features = np.load(thetas_path)
        if Z_all_features.ndim == 1:
            if D_STRUCT <= len(Z_all_features):
                Z_struct_features = Z_all_features[:D_STRUCT].reshape(1, D_STRUCT)
            else:
                raise ValueError(f"Розмірність D_STRUCT ({D_STRUCT}) більша за довжину завантаженого масиву ({len(Z_all_features)}).")
        elif Z_all_features.ndim == 2:
            Z_struct_features = Z_all_features[:, :D_STRUCT]
        else:
            raise ValueError(f"Завантажений масив має неочікувану кількість вимірів: {Z_all_features.ndim}.")
        
        if Z_struct_features.shape[0] == 0:
            raise ValueError("Завантажений файл 'generated_valid_thetas.npy' не містить даних.")

    except FileNotFoundError as e:
        print(f"ПОМИЛКА: Файл не знайдено: {e}")
        raise
    except ValueError as ve:
        print(f"ПОМИЛКА формату даних: {ve}")
        raise
    except Exception as e:
        print(f"Помилка при завантаженні або обробці файлу 'generated_valid_thetas.npy': {e}")
        raise

    # --- Аналіз згенерованих гіпотез ---
    orders_list = []
    arma_count = 0
    garch_count = 0
    combined_count = 0
    other_count = 0

    for i in range(Z_struct_features.shape[0]):
        raw_struct_vector = Z_struct_features[i, :]
        decoded_struct_info = decode_discrete_structure(raw_struct_vector, epsilon=0.05)
        
        p = decoded_struct_info.get('p_order', 0)
        q = decoded_struct_info.get('q_order', 0)
        r = decoded_struct_info.get('r_order', 0)
        s = decoded_struct_info.get('s_order', 0)
        
        orders_list.append((p, q, r, s))

        is_arma_active = p > 0 or q > 0
        is_garch_active = r > 0 or s > 0

        if is_arma_active and not is_garch_active:
            arma_count += 1
        elif is_garch_active and not is_arma_active:
            garch_count += 1
        elif is_arma_active and is_garch_active:
            combined_count += 1
        else:
            other_count += 1

    # --- Збереження масиву порядків у файл ---
    if orders_list:
        orders_array = np.array(orders_list)
        np.save("orders_generated.npy", orders_array)
        print(f"\nМасив порядків (p,q,r,s) для {len(orders_list)} гіпотез збережено у 'orders_generated.npy'.")
    else:
        print("\nНемає даних про порядки для збереження (список orders_list порожній).")

    # --- Вивід результатів підрахунку ---
    total_processed = len(orders_list)
    print(f"\nЗагальна кількість оброблених (і параметрично валідних) гіпотез: {total_processed}")

    if total_processed > 0:
        print("\nРозподіл типів моделей (на основі аналізу порядків p,q,r,s):")
        print(f"Чисті ARMA (p>0 or q>0; r=0,s=0):          {arma_count} ({arma_count/total_processed:.2%})")
        print(f"Чисті GARCH (r>0 or s>0; p=0,q=0):         {garch_count} ({garch_count/total_processed:.2%})")
        print(f"Комбіновані ARMA-GARCH (p>0 or q>0; r>0,s>0): {combined_count} ({combined_count/total_processed:.2%})")
        if other_count > 0:
            print(f"Інші (p=0,q=0,r=0,s=0 - модель без динаміки): {other_count} ({other_count/total_processed:.2%})")
    else:
        print("Немає згенерованих гіпотез для аналізу типів.")


if __name__ == '__main__':
    main()