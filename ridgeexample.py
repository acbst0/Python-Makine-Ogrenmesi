from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np

# Örnek veri oluşturma
np.random.seed(42)
X = np.random.rand(100, 5)  # 100 örnek, her biri 5 özellik içeriyor
y = 2 * X[:, 0] + 3 * X[:, 1] + 0.5 * X[:, 2] + np.random.normal(0, 0.1, 100)  # Gerçek ilişkiyi yansıtan gürültülü hedef

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge regresyon modelini oluşturma
alpha = 1.0  # Ridge regresyonun düzenlileştirme parametresi
ridge_model = Ridge(alpha=alpha)

# Modeli eğitme
ridge_model.fit(X_train, y_train)

# Eğitilmiş modeli kullanarak tahmin yapma
y_pred = ridge_model.predict(X_test)
print(X)
# Tahminin performansını değerlendirme
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
