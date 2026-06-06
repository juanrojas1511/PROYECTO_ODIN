import os


import numpy as np
from sklearn.model_selection import train_test_split
print(os.getcwd())

# ═══════════════════════════════════════
# DATASET SIMULADO
# 0 = Gato
# 1 = Perro
# ═══════════════════════════════════════

dataset = np.array([

    # GATOS
    [172,148,132,38,35,30,0.98,151,0],
    [165,142,125,42,39,33,1.02,144,0],
    [180,155,138,35,32,28,0.95,158,0],
    [145,128,115,48,44,38,1.05,129,0],

    # PERROS
    [140,118,98,62,57,50,1.25,119,1],
    [155,130,108,70,65,56,1.30,131,1],
    [128,108,90,75,69,60,1.20,109,1],
    [170,142,118,58,54,47,1.35,143,1]

])

# FEATURES y LABELS
X = dataset[:, :-1]
y = dataset[:, -1]

# DIVIDIR DATASET
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=7
)

# ═══════════════════════════════════════
# GUARDAR EN DISCO
# ═══════════════════════════════════════

np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)

np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("✅ Dataset guardado correctamente en tu disco")