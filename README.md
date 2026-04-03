**Key Techniques Used:**
- **He (Kaiming) initialization** → optimal for ReLU activations
- **Batch Normalization** → custom implementation with full forward & backward pass
- **Dropout** → reduces overfitting
- **L2 weight decay** → additional regularization
- **Manual backpropagation** → all gradients computed by hand

---

## 🔍 Hyperparameter Tuning

Tested multiple combinations of hidden layer sizes, learning rate, dropout rate, and weight decay.

**Tuning Results:**

| h1  | h2 | lr     | dropout | Val RMSE ($100k) |
|-----|----|--------|---------|------------------|
| 64  | 32 | 0.0100 | 0.2     | 0.8237 |
| 128 | 64 | 0.0100 | 0.2     | 0.7746 |
| 128 | 64 | 0.0050 | 0.3     | 0.8302 |
| 256 | 64 | 0.0050 | 0.2     | 0.7834 |
| **64**  | **16** | **0.0200** | **0.1** | **0.7689 ✅** |

**Best Parameters:**
```python
{'h1': 64, 'h2': 16, 'lr_init': 0.02, 'dropout': 0.1, 'weight_decay': 1e-4}
```

---

## 📉 Training Process

- **Loss function:** MSE (implemented in pure NumPy)
- **Optimizer:** Full-batch gradient descent with manual backpropagation
- **LR scheduler:** Step decay — halved every 60 epochs
- **Epochs:** 200 (tuning) → 300 (final training)

---

## 📊 Results & Evaluation

> Results are expressed in $100k units (e.g. RMSE of 0.76 = $76,000 error).

### ✅ Validation Performance
| Metric | Value |
|--------|-------|
| RMSE | 0.7639 ($100k) ≈ **$76,390** avg error |
| R² Score | 0.5571 |

👉 The model explains **~55.7%** of the variance in house prices.

### 🧪 Test Performance
| Metric | Value |
|--------|-------|
| RMSE | 0.7385 ($100k) ≈ **$73,850** avg error |
| R² Score | 0.5735 |

👉 The model explains **~57.4%** of the variance on unseen data.

### 🔎 Sample Inference (1st test block group)
| | Value |
|-|-------|
| Predicted | $305,495 |
| Actual | $267,600 |

---

## 📈 Visualizations

### 1. Loss Curves
Training and validation MSE decrease over epochs. Model converges and stabilizes. Slight gap indicates mild overfitting.

![Loss Curves](images/loss_curves.png)

### 2. Predicted vs Actual
Points follow the diagonal trend. Some deviation at high-value properties.

![Predicted vs Actual](images/pred_vs_actual.png)

### 3. Feature Importance
`MedInc` dominates with ~0.52 importance. Geographic features (Latitude, Longitude) follow.

![Feature Importance](images/feature_importance.png)

---

## 🧠 Interpretation of Results

- The model successfully learns patterns in the data
- Performance drop from validation → test is expected
- All components (BatchNorm, Dropout, backpropagation) implemented **manually** — no external ML libraries used

---

## 💾 Model Saving & Inference

Model is saved as `california_model.npz` and can be loaded for inference:
```python
loaded = CaliforniaModel.load('california_model.npz', input_dim=8, h1=64, h2=16)
pred = loaded.predict(X_test_s[0:1])
```

---

## 🏁 Conclusion

- ✅ Built and trained a neural network for house price prediction using **pure NumPy**
- ✅ Applied proper ML pipeline: Preprocessing → Train/Val/Test split → Hyperparameter tuning → Evaluation
- ✅ Implemented BatchNorm, Dropout, He initialization, and backpropagation **entirely from scratch**
- ✅ Achieved reasonable accuracy on a large real-world dataset (20,640 samples)