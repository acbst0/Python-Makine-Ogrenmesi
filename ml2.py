import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Numpy kütüphanesini içe aktar
# Veri manipülasyonu için kullanılacak

# Scikit-learn kütüphanesinden train_test_split ve Ridge sınıflarını içe aktar
# Veriyi eğitim ve test kümelerine ayırmak, Ridge Regresyonu modeli oluşturmak için

# Skor hesaplamak için ortalama karesel hata metriği (mean_squared_error) fonksiyonunu içe aktar

# Rastgele veri oluşturma
np.random.seed(42)  # Rastgele verinin tekrarlanabilirliğini sağlar
X = np.random.rand(100, 5)  # 100 örnek ve 5 öznitelik içeren rastgele bir veri oluşturulur
y = 2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2] + np.random.randn(100)
# y, X özniteliklerine dayalı olarak gerçek değerlere biraz rastgele gürültü eklenerek oluşturulur

# Veriyi eğitim ve test kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Veri, eğitim ve test kümelerine ayrılır
# test_size, verinin ne kadarının test verisi olarak kullanılacağını belirler
# random_state, rastgele veri ayrımının tekrarlanabilirliğini sağlar

# Ridge Regresyonu modelini oluşturma
alpha = 1.0  # Düzenleme parametresi (alpha=0 ise standart lineer regresyon olur)
ridge_model = Ridge(alpha=alpha)
# Ridge regresyonu modeli oluşturulur
# alpha, düzenleme katsayısını belirler (düzenleme yoksa alpha=0)

# Modeli eğitme
ridge_model.fit(X_train, y_train)
# Oluşturulan Ridge modeli, eğitim verileri üzerinde eğitilir

# Eğitim ve test kümesi üzerinde tahmin yapma
y_train_pred = ridge_model.predict(X_train)
y_test_pred = ridge_model.predict(X_test)
# Eğitilmiş model, hem eğitim hem de test verileri üzerinde tahminler yapar

# Hata hesaplama
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
# Eğitim ve test kümesi üzerindeki tahminlerin gerçek değerlerle ne kadar uyumlu olduğunu
# ortalama karesel hata (RMSE) ile değerlendirir

# Sonuçları yazdırma
print(f"Eğitim kümesi RMSE: {train_rmse:.2f}")
print(f"Test kümesi RMSE: {test_rmse:.2f}")
# Eğitim ve test kümesi üzerindeki tahminlerin hata metriklerini yazdırır
