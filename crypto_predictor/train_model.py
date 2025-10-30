import numpy as np
from sklearn.linear_model import LinearRegression
from joblib import dump

# Giả lập dữ liệu (open, high, low, close, volume) -> predicted_close
np.random.seed(42)
X = np.random.rand(500, 5) * 100  # input features
y = X[:, 3] * 1.01 + np.random.randn(500) * 0.5  # target = close * 1.01 + noise

# Huấn luyện mô hình
model = LinearRegression()
model.fit(X, y)

# Lưu mô hình
dump(model, "app/models/price_model.joblib")
print("✅ Model saved to app/models/price_model.joblib")
