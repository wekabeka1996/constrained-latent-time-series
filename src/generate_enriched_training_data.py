import numpy as np
import os

def generate_polynomial_coeffs(order):
    """
    Генерує дійсні коефіцієнти поліному з коренями всередині одиничного кола.
    Повертає масив коефіцієнтів (від старшого до молодшого).
    """
    if order == 0:
        return np.array([])
    roots = []
    i = 0
    while i < order:
        # Випадково вирішуємо: дійсний чи комплексний корінь
        if order - i >= 2 and np.random.rand() < 0.5:
            # Комплексно-спряжена пара
            R = np.random.uniform(0, 1)
            alpha = np.random.uniform(0, 2*np.pi)
            r = np.sqrt(R) * np.exp(1j * alpha)
            roots.append(r)
            roots.append(np.conj(r))
            i += 2
        else:
            # Дійсний корінь
            r = np.random.uniform(-1, 1)
            roots.append(r)
            i += 1
    coeffs = np.poly(roots)
    return coeffs.real

def generate_valid_arma_via_roots(p, q):
    # AR: 1 - phi1*z - ... - phip*z^p = 0
    # MA: 1 + theta1*z + ... + thetaq*z^q = 0
    ar_coeffs = generate_polynomial_coeffs(p)
    ma_coeffs = generate_polynomial_coeffs(q)
    # Витягуємо phi, theta зі стандартної форми
    # np.poly повертає [1, a1, a2, ...], а нам треба [phi1, phi2, ...]
    phi = -ar_coeffs[1:] if p > 0 else np.array([])
    theta = ma_coeffs[1:] if q > 0 else np.array([])
    return phi, theta

def main():
    # Завантажити оригінальні дані
    Z_train = np.load('Z_train.npy')
    train_labels = np.load('train_labels.npy')
    # Додати ARMA моделі
    n_arma = 100
    arma_vecs = []
    arma_labels = []
    attempts = 0
    while len(arma_vecs) < n_arma and attempts < n_arma * 20:
        p, q = np.random.choice([1,2,3]), np.random.choice([1,2,3])
        phi, theta = generate_valid_arma_via_roots(p, q)
        attempts += 1
        if phi is None:
            continue
        # Формуємо вектор так само, як у оригінальному датасеті
        x = np.zeros(Z_train.shape[1])
        # ...структурні індикатори: ARMA
        x[0] = 1  # AR
        x[1] = 1  # MA
        x[2] = 0  # GARCH
        # ...порядки
        x[3] = p
        x[4] = q
        # ...коефіцієнти AR
        x[10:10+p] = phi
        # ...коефіцієнти MA
        x[20:20+q] = theta
        arma_vecs.append(x)
        arma_labels.append(2)  # label 2 = ARMA
    if len(arma_vecs) == 0:
        print('Не вдалося згенерувати жодної валідної ARMA моделі!')
        return
    Z_enriched = np.vstack([Z_train, np.array(arma_vecs)])
    labels_enriched = np.concatenate([train_labels, np.array(arma_labels)])
    np.save('Z_train_enriched.npy', Z_enriched)
    np.save('train_labels_enriched.npy', labels_enriched)
    print(f'Збагачений датасет збережено: Z_train_enriched.npy ({Z_enriched.shape}), train_labels_enriched.npy ({labels_enriched.shape})')

if __name__ == '__main__':
    main()
