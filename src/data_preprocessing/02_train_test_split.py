import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #test ve train ayrımı için gerekli
from sklearn.experimental import enable_iterative_imputer # IterativeImputer deneysel olduğu için önce bunu çağırmalıyız
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler #ölçeklendirme için gerekli

#TEMİZ VERİ SETİ İLE DEVAM EDİLİR

df = pd.read_csv('SONbrfss2024_temiz.csv')
#GERİYE KALAN İŞLEMLER
#1) Train/test split
#2) KNN imputasyonu. gereksiz değerleri NaN yapmıştık. Şimdi o değerleri elimizdeki verilere dayanarak ortalama bir şekilde dolduracağız.(KNN imputation)
#3) Ölçeklendirme: Veriler arasında orantısızlık varsa model yanlı olur. Bunu engellemek için ölçekleme yaparız.


BAGIMSIZ = [
    'CHCSCNC1', 'CHECKUP1', '_AGEG5YR', 'SMOKE100',
    'DIABETE4', 'EXERANY2', 'ASTHMA3', 'CHCKDNY2',
    'HAVARTH4', 'CVDINFR4', 'PERSDOC3', '_RFHLTH',
    'INCOME3', 'EDUCA', '_BMI5CAT',
]
#şuan ağırlıklarla yani hipotez fonksiyonundaki X'in önündeki teta'lar ile işimiz yok. Sadece veriyi ikiye böleceğiz o kadar.
X = df[BAGIMSIZ] #bağımsız değişkenler = X
y = df['CHCOCNC1'] #Hedef değişken = y = tahmin edilmeye çalışılan sonuç. "Bu kişi kanser midir?" Sorusuna yanıt verecek!!!!

X_train, X_test, y_train, y_test = train_test_split( #Burada ikisi de train ve test olarak ayrıldı.
    X, y,
    test_size=0.20, # %20 test %80 eğitim verisi olarak ayarlandı.
    random_state=42,
    stratify=y
)

print(f"Eğitim seti: {len(X_train):,} satır")
print(f"Test seti  : {len(X_test):,} satır")
print(f"Eğitimde kanser oranı: %{y_train.mean()*100:.1f}")
print(f"Testte kanser oranı  : %{y_test.mean()*100:.1f}")
# Ayırma işlemi burada biter.

#Sıra NaN'ları doldurmak için Iterative Imputer'da.

# KNN aşırı uzun sürdüğü için yerine çok daha hızlı çalışan makine öğrenmesi tabanlı IterativeImputer eklendi.
imputer = IterativeImputer(random_state=42, max_iter=10)

# Sadece eğitim verisine fit edilecek.
X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=BAGIMSIZ
)

# Test verisine sadece transform uygulanacak.
X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=BAGIMSIZ
)

print("Eksik veri doldurma (IterativeImputer) tamamlandı.")
print(f"Train'de kalan NaN: {X_train.isna().sum().sum()}")
print(f"Test'te kalan NaN : {X_test.isna().sum().sum()}")

# ÖLÇEKLENDİRME 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Scaler'ı sadece Eğitim (Train) verisine fit ediyoruz. (Veri sızıntısını önlemek için)
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=BAGIMSIZ
)

# Test verisine sadece transform yapıyoruz
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=BAGIMSIZ
)
print("Ölçeklendirme tamamlandı.")

# --- VERİLERİ KAYDETME (KNN VE ÖLÇEKLEME SONRASI) ---
# İşlem çok uzun sürdüğü için, bir sonraki modelleme aşamasında doğrudan bu temiz/ölçeklenmiş verileri okumak için kaydediyoruz.
X_train_scaled.to_csv("X_train_hazir.csv", index=False)
X_test_scaled.to_csv("X_test_hazir.csv", index=False)
y_train.to_csv("y_train_hazir.csv", index=False)
y_test.to_csv("y_test_hazir.csv", index=False)
print("Tüm veriler ayrı dosyalar olarak başarıyla kaydedildi!")
scaler = StandardScaler()

# test verisine bilgi sızmaması için scaler'ı sadece eğitim verisine fit ediyoruz.
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=BAGIMSIZ
)

# Test verisine sadece transform yapıyoruz
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=BAGIMSIZ
)

print("Veri Ölçekleme (Scaling) tamamlandı.")
