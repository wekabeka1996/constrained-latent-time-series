import numpy as np
arr = np.load("orders_generated.npy", allow_pickle=True)
print(f"Тип: {type(arr)}; Довжина: {len(arr)}")
print("Перші 3 елементи:")
for i in range(min(3, len(arr))):
    print(arr[i])
