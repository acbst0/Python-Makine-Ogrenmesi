import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Beygir gücü ve yakıt tüketimi verilerini okuma
with open("veriler.txt", "r") as file:
    veri_lines = file.readlines()
    
with open("yakit_tuketimi.txt", "r") as file:
    yakit_lines = file.readlines()

# Her satırı bir liste elemanı olarak almak ve boşlukları temizlemek
veri_data = [line.strip().split() for line in veri_lines]
yakit_data = [float(line.strip()) for line in yakit_lines]

# Verileri özellik matrisi (X) ve hedef vektörü (y) olarak ayırma
X = np.array([[float(line[0]), float(line[1]), float(line[2])] for line in veri_data])  # Beygir gücü ve ağırlık
y = np.array(yakit_data)                              # Yakıt tüketimi

# Veriyi eğitim ve test kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lineer Regresyon modelini oluşturma
model = LinearRegression()

# Modeli eğitme
model.fit(X_train, y_train)

# Test kümesi üzerinde tahmin yapma
y_pred = model.predict(X_test)

# Hata hesaplaması (örneğin, RMSE)
mse = mean_squared_error(y_test, y_pred)
print("Ortalama Karesel Hata:", mse)

# Sonuçları yazdırma
hp = float(input("Yeni aracın beygir gücünü girin: "))
kg = float(input("Yeni aracın ağırlığını girin: "))
cc = float(input("Yeni aracın motor hacmini girin: "))
girdiaraba = np.array([[hp, kg, cc]])  # Tek bir satır içeren bir dizi oluşturuldu

predicted_fuel_consumption = model.predict(girdiaraba)
print("Tahmin Edilen Yakit Tüketimi: ", predicted_fuel_consumption[0])