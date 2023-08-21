import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Örnek veri kümesini oluşturalım (örneğin, otomobil yakıt tüketimi)
# Özellikler: Motor gücü (HP) ve araba ağırlığı (kg)
X = np.array([[120, 1000],
              [150, 1200],
              [100, 800],
              [180, 1500],
              [200, 1800]])

# Etiketler: Yakıt tüketimi (lt/100km)
y = np.array([15])

# Veriyi eğitim ve test kümelerine bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lineer regresyon modelini oluşturalım ve eğitelim
model = LinearRegression()
model.fit(X_train, y_train)

# Test kümesi üzerinde tahmin yapalım
y_pred = model.predict(X_test)

# Hata metriklerini hesaplayalım (örneğin, ortalama karesel hata)
mse = mean_squared_error(y_test, y_pred)
print("Ortalama Karesel Hata:", mse)

# Yeni bir otomobil örneğinin yakıt tüketimini tahmin edelim
new_car = np.array([[150, 2000]])
predicted_fuel_consumption = model.predict(new_car)
print("Tahmin Edilen Yakit Tüketimi: ", predicted_fuel_consumption[0])
