import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Rastgele veri oluşturma
np.random.seed(42)
X = np.random.rand(100, 5)  # 100 örnek, 5 öznitelik
y = 2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2] + np.random.randn(100)

# Veriyi eğitim ve test kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge Regresyonu modelini oluşturma
alpha = 1.0  # Düzenleme parametresi (alpha=0 ise standart lineer regresyon olur)
ridge_model = Ridge(alpha=alpha)

# Modeli eğitme
ridge_model.fit(X_train, y_train)

# Eğitim ve test kümesi üzerinde tahmin yapma
y_train_pred = ridge_model.predict(X_train)
y_test_pred = ridge_model.predict(X_test)

# Hata hesaplama
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"Eğitim kümesi RMSE: {train_rmse:.2f}")
print(f"Test kümesi RMSE: {test_rmse:.2f}")
