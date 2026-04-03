import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import pandas as pd

data = fetch_california_housing(as_frame=True)
df = data.frame
df.to_csv('california_housing.csv', index=False)

np.random.seed(42)

# 1. LOAD DATA
def load_csv(path):
    rows = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append([float(x) for x in row])
    return header, np.array(rows)

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(script_dir, 'california_housing.csv')

header, data = load_csv(data_path)

print(f"\nDataset shape  : {data.shape}")
print(f"Features       : {header[:-1]}")
print(f"Target         : {header[-1]}")

# 2. SPLIT FEATURES AND TARGET
X = data[:, :-1]          # shape (20640, 8)
y = data[:, -1:].copy()   # shape (20640, 1)  — MedHouseVal

feat_names = header[:-1]

# 3. TRAIN / VALIDATION / TEST SPLIT  (80/10/10)
def train_val_test_split(X, y, val_ratio=0.10, test_ratio=0.10, seed=42):
    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(len(X))
    n    = len(X)
    n_te = int(n * test_ratio)
    n_va = int(n * val_ratio)
    te   = idx[:n_te]
    va   = idx[n_te:n_te + n_va]
    tr   = idx[n_te + n_va:]
    return X[tr], X[va], X[te], y[tr], y[va], y[te]

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

print(f"\nSplit — Train: {X_train.shape[0]:,} | "
      f"Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")

# 4. STANDARDSCALER  — Z-score normalization (mean=0, std=1)
class StandardScaler:
    def fit(self, arr):
        self.mean_ = arr.mean(axis=0)
        self.std_  = arr.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, arr):
        return (arr - self.mean_) / self.std_

    def inverse_transform(self, arr):
        return arr * self.std_ + self.mean_

scaler_X = StandardScaler().fit(X_train)
scaler_y = StandardScaler().fit(y_train)

X_train_s = scaler_X.transform(X_train)
X_val_s   = scaler_X.transform(X_val)
X_test_s  = scaler_X.transform(X_test)

y_train_s = scaler_y.transform(y_train)
y_val_s   = scaler_y.transform(y_val)
y_test_s  = scaler_y.transform(y_test)

# 5. BATCH NORMALIZATION  (numpy only)
#
#   Forward (training):
#       mu  = mean(x)
#       var = variance(x)
#       x_hat = (x - mu) / sqrt(var + eps)
#       out   = gamma * x_hat + beta
#
#   running_mean / running_var — inference-ի համար
#   gamma, beta — learnable parameters
class BatchNorm1d:
    def __init__(self, size, eps=1e-5, momentum=0.1):
        self.gamma        = np.ones((1, size))   # scale  (learnable)
        self.beta         = np.zeros((1, size))  # shift  (learnable)
        self.eps          = eps
        self.momentum     = momentum
        self.running_mean = np.zeros((1, size))
        self.running_var  = np.ones((1, size))
        self._cache       = None                 # backward-ի համար

    def forward(self, x, training=True):
        if training:
            mu      = x.mean(axis=0, keepdims=True)
            var     = x.var(axis=0,  keepdims=True)
            x_hat   = (x - mu) / np.sqrt(var + self.eps)
            # running statistics update
            self.running_mean = ((1 - self.momentum) * self.running_mean
                                 + self.momentum * mu)
            self.running_var  = ((1 - self.momentum) * self.running_var
                                 + self.momentum * var)
            self._cache = (x, x_hat, mu, var)
        else:
            x_hat = ((x - self.running_mean)
                     / np.sqrt(self.running_var + self.eps))

        return self.gamma * x_hat + self.beta

    def backward(self, dout):
        x, x_hat, mu, var = self._cache
        N = x.shape[0]

        dgamma = (dout * x_hat).sum(axis=0, keepdims=True)
        dbeta  = dout.sum(axis=0, keepdims=True)

        dx_hat = dout * self.gamma
        dvar   = (-0.5 * dx_hat * (x - mu)
                  * (var + self.eps) ** (-1.5)).sum(axis=0, keepdims=True)
        dmu    = ((-dx_hat / np.sqrt(var + self.eps)).sum(axis=0, keepdims=True)
                  + dvar * (-2 * (x - mu)).mean(axis=0, keepdims=True))
        dx     = (dx_hat / np.sqrt(var + self.eps)
                  + dvar * 2 * (x - mu) / N
                  + dmu / N)
        return dx, dgamma, dbeta

# 6. NEURAL NETWORK  (pure numpy)
#
#   Architecture:
#     input(8)
#       → Linear(H1) → BatchNorm → ReLU → Dropout
#       → Linear(H2) → BatchNorm → ReLU → Dropout
#       → Linear(1)
#
#   Init      : He (Kaiming) — best for ReLU
#   Reg       : Dropout + L2 weight decay
#   Optimizer : Full-batch gradient descent + step LR decay
def relu(z):       
    return np.maximum(0.0, z)
def relu_grad(z):  
    return (z > 0).astype(float)

def he_init(fan_in, fan_out, rng):
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, fan_out))
class CaliforniaModel:
    """
    Fully-connected 2-hidden-layer network with BatchNorm for regression.
    Forward:
      Z1 = X  @ W1 + b1
      N1 = BatchNorm(Z1)          ← NEW
      A1 = ReLU(N1)  [+ Dropout]
      Z2 = A1 @ W2 + b2
      N2 = BatchNorm(Z2)          ← NEW
      A2 = ReLU(N2)  [+ Dropout]
      Z3 = A2 @ W3 + b3           (linear output)
    Loss: MSE(y_true, Z3) + L2 weight penalty
    """
    def __init__(self, input_dim, h1=128, h2=64, seed=0):
        rng     = np.random.default_rng(seed)
        self.W1 = he_init(input_dim, h1, rng); self.b1 = np.zeros((1, h1))
        self.bn1 = BatchNorm1d(h1)

        self.W2 = he_init(h1, h2, rng);        self.b2 = np.zeros((1, h2))
        self.bn2 = BatchNorm1d(h2)

        self.W3 = he_init(h2, 1, rng);         self.b3 = np.zeros((1, 1))

        self._c = {}   # backprop cache

    def forward(self, X, dropout=0.0, training=False):
        # Layer 1
        Z1 = X @ self.W1 + self.b1
        N1 = self.bn1.forward(Z1, training=training)   # BatchNorm
        A1 = relu(N1)
        if training and dropout > 0:
            mask1 = (np.random.rand(*A1.shape) > dropout) / (1.0 - dropout)
            A1   *= mask1
        else:
            mask1 = None

        # Layer 2
        Z2 = A1 @ self.W2 + self.b2
        N2 = self.bn2.forward(Z2, training=training)   # BatchNorm
        A2 = relu(N2)
        if training and dropout > 0:
            mask2 = (np.random.rand(*A2.shape) > dropout) / (1.0 - dropout)
            A2   *= mask2
        else:
            mask2 = None

        # Output layer
        Z3 = A2 @ self.W3 + self.b3

        self._c = dict(X=X,
                       Z1=Z1, N1=N1, A1=A1, mask1=mask1,
                       Z2=Z2, N2=N2, A2=A2, mask2=mask2)
        return Z3   # shape (N, 1)

    def backward(self, y_true, y_pred, lr, wd):
        """
        Gradients of:
          Loss = (1/N) * ||y_pred - y_true||²  +  wd * (||W1||² + ||W2||² + ||W3||²)
        """
        N = y_true.shape[0]
        c = self._c

        # Output layer
        dZ3 = (2.0 / N) * (y_pred - y_true)       # (N, 1)
        dW3 = c['A2'].T @ dZ3 + wd * self.W3
        db3 = dZ3.sum(axis=0, keepdims=True)
        dA2 = dZ3 @ self.W3.T

        # Dropout 2
        if c['mask2'] is not None:
            dA2 *= c['mask2']

        # ReLU 2
        dN2 = dA2 * relu_grad(c['N2'])

        # BatchNorm 2 backward
        dZ2, dgamma2, dbeta2 = self.bn2.backward(dN2)
        self.bn2.gamma -= lr * dgamma2
        self.bn2.beta  -= lr * dbeta2

        # Layer 2
        dW2 = c['A1'].T @ dZ2 + wd * self.W2
        db2 = dZ2.sum(axis=0, keepdims=True)
        dA1 = dZ2 @ self.W2.T

        # Dropout 1
        if c['mask1'] is not None:
            dA1 *= c['mask1']

        # ReLU 1
        dN1 = dA1 * relu_grad(c['N1'])

        # BatchNorm 1 backward
        dZ1, dgamma1, dbeta1 = self.bn1.backward(dN1)
        self.bn1.gamma -= lr * dgamma1
        self.bn1.beta  -= lr * dbeta1

        # Layer 1
        dW1 = c['X'].T @ dZ1 + wd * self.W1
        db1 = dZ1.sum(axis=0, keepdims=True)

        # Gradient descent update
        self.W1 -= lr * dW1;  self.b1 -= lr * db1
        self.W2 -= lr * dW2;  self.b2 -= lr * db2
        self.W3 -= lr * dW3;  self.b3 -= lr * db3

    def predict(self, X):
        return self.forward(X, training=False)

    # Persist
    def save(self, path):
        np.savez(path,
                 W1=self.W1, b1=self.b1,
                 bn1_gamma=self.bn1.gamma, bn1_beta=self.bn1.beta,
                 bn1_rm=self.bn1.running_mean, bn1_rv=self.bn1.running_var,
                 W2=self.W2, b2=self.b2,
                 bn2_gamma=self.bn2.gamma, bn2_beta=self.bn2.beta,
                 bn2_rm=self.bn2.running_mean, bn2_rv=self.bn2.running_var,
                 W3=self.W3, b3=self.b3)

    @classmethod
    def load(cls, path, input_dim, h1, h2):
        m   = cls(input_dim, h1, h2)
        npz = np.load(path)
        m.W1, m.b1 = npz['W1'], npz['b1']
        m.bn1.gamma        = npz['bn1_gamma']
        m.bn1.beta         = npz['bn1_beta']
        m.bn1.running_mean = npz['bn1_rm']
        m.bn1.running_var  = npz['bn1_rv']
        m.W2, m.b2 = npz['W2'], npz['b2']
        m.bn2.gamma        = npz['bn2_gamma']
        m.bn2.beta         = npz['bn2_beta']
        m.bn2.running_mean = npz['bn2_rm']
        m.bn2.running_var  = npz['bn2_rv']
        m.W3, m.b3 = npz['W3'], npz['b3']
        return m

# 7. METRICS
def mse_loss(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)

# 8. TRAINING FUNCTION
def train_model(h1, h2, lr_init, dropout, weight_decay,
                epochs=200, verbose=False, seed=0):

    model = CaliforniaModel(X_train_s.shape[1], h1=h1, h2=h2, seed=seed)
    lr    = lr_init
    train_losses, val_losses = [], []

    for ep in range(epochs):
        if ep > 0 and ep % 60 == 0:
            lr *= 0.5

        # Training step (BatchNorm training=True)
        y_pred_tr = model.forward(X_train_s, dropout=dropout, training=True)
        model.backward(y_train_s, y_pred_tr, lr=lr, wd=weight_decay)
        train_losses.append(mse_loss(y_train_s, y_pred_tr))

        # Validation (BatchNorm training=False → running stats)
        y_pred_va = model.predict(X_val_s)
        val_losses.append(mse_loss(y_val_s, y_pred_va))

        if verbose and ep % 30 == 0:
            print(f"  Epoch {ep:4d} | Train MSE={train_losses[-1]:.4f}"
                  f" | Val MSE={val_losses[-1]:.4f} | lr={lr:.5f}")

    return model, train_losses, val_losses

# 9. HYPERPARAMETER GRID SEARCH
param_grid = [
    {'h1': 64,  'h2': 32, 'lr_init': 0.010, 'dropout': 0.2, 'weight_decay': 1e-4},
    {'h1': 128, 'h2': 64, 'lr_init': 0.010, 'dropout': 0.2, 'weight_decay': 1e-4},
    {'h1': 128, 'h2': 64, 'lr_init': 0.005, 'dropout': 0.3, 'weight_decay': 1e-4},
    {'h1': 256, 'h2': 64, 'lr_init': 0.005, 'dropout': 0.2, 'weight_decay': 1e-3},
    {'h1': 64,  'h2': 16, 'lr_init': 0.020, 'dropout': 0.1, 'weight_decay': 1e-4},
]

best_val_rmse = float('inf')
best_params   = None
best_model    = None

for p in param_grid:
    model, _, _ = train_model(**p, epochs=200, verbose=False, seed=42)
    preds  = scaler_y.inverse_transform(model.predict(X_val_s))
    trues  = scaler_y.inverse_transform(y_val_s)
    v_rmse = rmse(trues, preds)
    print(f"  h1={p['h1']:3d} h2={p['h2']:2d}"
          f" lr={p['lr_init']:.4f} drop={p['dropout']:.1f}"
          f"  →  Val RMSE: {v_rmse:.4f} ($100k)")

    if v_rmse < best_val_rmse:
        best_val_rmse = v_rmse
        best_params   = p
        best_model    = model

print(f"\nBest params   : {best_params}")
print(f"Best Val RMSE : {best_val_rmse:.4f} ($100k)")

# 10. RETRAIN BEST MODEL — 300 epochs
print("\nRETRAINING BEST MODEL — 300 epochs")

best_model, train_losses, val_losses = train_model(**best_params, epochs=300, verbose=True, seed=42)

# 11. PLOTS
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(train_losses, label='Train MSE', linewidth=1.8, color='royalblue')
ax.plot(val_losses,   label='Validation MSE', linewidth=1.8, linestyle='--', color='orangered')
ax.set_title("Loss Curves — Best Model (300 epochs)", fontsize=13, fontweight='bold')
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE (scaled)")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'loss_curves.png'), dpi=150)
plt.show()

y_pred_test_s = best_model.predict(X_test_s)
y_pred_test   = scaler_y.inverse_transform(y_pred_test_s)
y_true_test   = scaler_y.inverse_transform(y_test_s)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_true_test, y_pred_test, alpha=0.4, edgecolors='none', color='steelblue', s=15)
lo = min(y_true_test.min(), y_pred_test.min())
hi = max(y_true_test.max(), y_pred_test.max())
ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect prediction')
ax.set_xlabel("Actual Median House Value ($100k)", fontsize=11)
ax.set_ylabel("Predicted Median House Value ($100k)", fontsize=11)
ax.set_title("Test Set — Predicted vs Actual", fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'pred_vs_actual.png'), dpi=150)
plt.show()

def permutation_importance(model, X_s, y_s, scaler_y, names, n_rep=5):
    base_pred  = scaler_y.inverse_transform(model.predict(X_s))
    base_true  = scaler_y.inverse_transform(y_s)
    base_rmse_ = rmse(base_true, base_pred)
    scores = []
    for i in range(X_s.shape[1]):
        rep = []
        for _ in range(n_rep):
            Xp       = X_s.copy()
            Xp[:, i] = np.random.permutation(Xp[:, i])
            pp = scaler_y.inverse_transform(model.predict(Xp))
            rep.append(rmse(base_true, pp) - base_rmse_)
        scores.append(np.mean(rep))
    return np.array(scores)

importance = permutation_importance(best_model, X_val_s, y_val_s, scaler_y, feat_names)
sorted_idx = np.argsort(importance)[::-1]
colors     = ['#e74c3c' if importance[i] > 0.05 else '#3498db' for i in sorted_idx]

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(range(len(feat_names)), importance[sorted_idx], color=colors, edgecolor='white', width=0.7)
ax.set_xticks(range(len(feat_names)))
ax.set_xticklabels([feat_names[i] for i in sorted_idx], rotation=25, ha='right', fontsize=11)
ax.set_ylabel("Mean RMSE increase ($100k)")
ax.set_title("Permutation Feature Importance (Validation Set)", fontsize=12, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'feature_importance.png'), dpi=150)
plt.show()

y_pred_val = scaler_y.inverse_transform(best_model.predict(X_val_s))
y_true_val = scaler_y.inverse_transform(y_val_s)

val_rmse_  = rmse(y_true_val, y_pred_val)
val_r2_    = r2_score(y_true_val, y_pred_val)
test_rmse_ = rmse(y_true_test, y_pred_test)
test_r2_   = r2_score(y_true_test, y_pred_test)

print("\nFINAL EVALUATION")
print(f"\nValidation RMSE : {val_rmse_:.4f}  ($100k units)")
print(f"Validation R²   : {val_r2_:.4f}")
print(f"\nTest RMSE       : {test_rmse_:.4f}  ($100k units)")
print(f"Test R²         : {test_r2_:.4f}")

# 12. SAVE MODEL  +  LOAD  +  INFERENCE
save_path = os.path.join(script_dir, 'california_model')
best_model.save(save_path)

loaded = CaliforniaModel.load(
    save_path + '.npz',
    input_dim = X_train_s.shape[1],
    h1 = best_params['h1'],
    h2 = best_params['h2']
)

sample_s = X_test_s[0:1]
pred_s   = loaded.predict(sample_s)
pred_val = scaler_y.inverse_transform(pred_s)[0, 0]
true_val = scaler_y.inverse_transform(y_test_s[0:1])[0, 0]

print("\nSAMPLE INFERENCE  (1st test block group)")
print(f"Predicted median house value : {pred_val*100_000:>10,.0f}")
print(f"Actual   median house value  : {true_val*100_000:>10,.0f}")
