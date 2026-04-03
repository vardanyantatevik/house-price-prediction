🏠 House Price Prediction

📌 Project Overview

This project implements a house price prediction model using pure NumPy. The goal is to build and train a neural network completely from scratch to predict median house values based on geographic and demographic features.

🎯 Task Objectives

1. Train a model for house price prediction
2. Preprocess and normalize the dataset
3. Design a neural network architecture from scratch using only NumPy
4. Find the best combination of hyperparameters
5. Evaluate the model using:
     RMSE (Root Mean Squared Error)
     R² Score
6. Visualize training/validation loss, predicted vs actual, and feature importance
7. Save the trained model and perform inference

📊 Dataset

Source: California Housing Dataset (sklearn.datasets.fetch_california_housing)
Total samples: 20,640 rows
Features (8):
   MedInc — Median income in block group
   HouseAge — Median house age
   AveRooms — Average rooms per household
   AveBedrms — Average bedrooms per household
   Population — Block group population
   AveOccup — Average household members
   Latitude / Longitude
Target: MedHouseVal — Median house value (in $100k units)

🔧 Preprocessing Steps

Loaded CSV using Python's built-in csv module
Split into Train / Validation / Test (80 / 10 / 10)
Applied custom StandardScaler (NumPy-only) — fit on training set only
Both features (X) and target (y) are normalized
No missing values in the dataset

⚙️ Technologies Used
🧠 Machine Learning

Pure NumPy → building and training neural network from scratch

📊 Data Handling

pandas → export to CSV only
csv → loading data
numpy → all numerical operations

📈 Visualization

matplotlib → plotting loss curves, predicted vs actual, feature importance

🧠 Model Architecture

A fully connected neural network built in pure NumPy:
Input (8 features)
  → Linear(H1) → BatchNorm1d → ReLU → Dropout
  → Linear(H2) → BatchNorm1d → ReLU → Dropout
  → Linear(1)   [output: predicted price]

Key Techniques Used:
   He (Kaiming) initialization → optimal for ReLU activations
   Batch Normalization → custom implementation with full forward & backward pass
   Dropout → reduces overfitting
   L2 weight decay → additional regularization
   Manual backpropagation → all gradients computed by hand

🔍 Hyperparameter Tuning

Tested multiple combinations of:
   Hidden layer sizes (h1, h2)
   Learning rate (lr_init)
   Dropout rate (dropout)
   Weight decay (weight_decay)

Best Parameters Found:

python{'h1': 64,  'h2': 16, 'lr_init': 0.02,  'dropout': 0.1, 'weight_decay': 1e-4}
These parameters produced the lowest validation RMSE.

📉 Training Process

Loss function: MSE (implemented in pure NumPy)
Optimizer: Full-batch gradient descent with manual backpropagation
Learning rate scheduler: step decay — halved every 60 epochs
Model trained for:
   200 epochs (during tuning)
   300 epochs (final training)

📊 Results & Evaluation

Results are expressed in $100k units (e.g. RMSE of 0.76 = $76,000 error).

🔍 Hyperparameter Tuning Results:
| h1  | h2 | lr     | dropout | Val RMSE ($100k) |
|-----|----|--------|---------|------------------|
| 64  | 32 | 0.0100 | 0.2     | 0.8237           |
| 128 | 64 | 0.0100 | 0.2     | 0.7746           |
| 128 | 64 | 0.0050 | 0.3     | 0.8302           |
| 256 | 64 | 0.0050 | 0.2     | 0.7834           |
| 64  | 16 | 0.0200 | 0.1     | 0.7689           |

Best Parameters Found:
{'h1': 64, 'h2': 16, 'lr_init': 0.02, 'dropout': 0.1, 'weight_decay': 1e-4}

✅ Validation Performance:
RMSE: 0.7639 ($100k units) ≈ $76,390 average error
R² Score: 0.5571
👉 The model explains ~55.7% of the variance in house prices.

🧪 Test Performance:
RMSE: 0.7385 ($100k units) ≈ $73,850 average error
R² Score: 0.5735
👉 The model explains ~57.4% of the variance on unseen data.

🔎 Sample Inference (1st test block group):
Predicted: $305,495
Actual:    $267,600

📈 Visualizations

1. Loss Curves
   Training and validation MSE decrease over epochs
   Model converges and stabilizes
   Slight gap between train/val curves indicates mild overfitting
![Loss Curves](images/loss_curves.png)

2. Predicted vs Actual Plot
   Points follow the general trend of real prices
   Some deviation exists, especially for high-value properties
   Indicates reasonable but not perfect predictions
![Predicted vs Actual](images/pred_vs_actual.png)

3. Feature Importance
   Permutation-based importance computed on the validation set
   Shows which features most influence predictions
![Feature Importance](images/feature_importance.png)

🧠 Interpretation of Results
The model successfully learns patterns in the data
Performance drop from validation → test is expected
All components (BatchNorm, Dropout, backpropagation) implemented manually — no external ML libraries used

💾 Model Saving & Inference

Model is saved as:

california_model.npz

Can be loaded later to make predictions on new data:

pythonloaded = CaliforniaModel.load('california_model.npz', input_dim=8, h1=..., h2=...)
pred = loaded.predict(X_test_s[0:1])

🏁 Conclusion

Successfully built and trained a neural network for house price prediction using pure NumPy
Applied proper ML pipeline:
   Preprocessing
   Train / Validation / Test split
   Hyperparameter tuning
   Evaluation
Implemented BatchNorm, Dropout, He initialization, and backpropagation entirely from scratch
Achieved reasonable accuracy on a large real-world dataset (20,640 samples)