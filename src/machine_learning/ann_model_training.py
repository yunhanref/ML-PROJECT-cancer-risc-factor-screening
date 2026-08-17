import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 0. Tekrarlanabilirlik (Reproducibility) İçin Rastgelelik Sabitleme
np.random.seed(42)
tf.random.set_seed(42)

# 1. Hazır Veri Setlerini Yükleme
print("Veri setleri yükleniyor...")
X_train = pd.read_csv('X_train_hazir.csv')
X_test = pd.read_csv('X_test_hazir.csv')
y_train = pd.read_csv('y_train_hazir.csv')
y_test = pd.read_csv('y_test_hazir.csv')

y_train = y_train.squeeze()
y_test = y_test.squeeze()

# 2. Ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- DÜZELTME: Sınıf Ağırlıklarını Hesaplama ---
print("Dengesiz sınıflar için 'balanced' ağırlıklar hesaplanıyor...")
sinif_agirliklari = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
sinif_agirliklari_dict = dict(enumerate(sinif_agirliklari))

# 3. ANN Modelini İnşa Etme
print("Model inşa ediliyor...")
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Modeli Eğitme (Ağırlıklar ile)
print("Model eğitimi başlıyor...")
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train_scaled, y_train, 
                    epochs=100, 
                    batch_size=256, 
                    validation_split=0.2, 
                    class_weight=sinif_agirliklari_dict, 
                    callbacks=[early_stop],
                    verbose=1)

# 5. Tahminler ve Değerlendirme
y_pred_probs = model.predict(X_test_scaled)
y_pred_classes = (y_pred_probs > 0.5).astype(int)

# --- ÇIKTILARIN ÜRETİLMESİ ---
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('ANN Konfüzyon Matrisi (Balanced)')
plt.ylabel('Gerçek Değerler')
plt.xlabel('Tahmin Edilen Değerler')
plt.savefig('ann_konfuzyon_matrisi.png', bbox_inches='tight')
plt.close()

roc_auc = roc_auc_score(y_test, y_pred_probs)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ANN (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Eğrisi (Balanced)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.savefig('ann_roc_egrisi.png', bbox_inches='tight')
plt.close()

rapor = classification_report(y_test, y_pred_classes)
with open('siniflandirma_raporu.txt', 'w') as f:
    f.write("ANN Sınıflandırma Raporu (Balanced Ağırlıklar ile)\n")
    f.write("="*50 + "\n")
    f.write(rapor)

model.save('kanser_taramasi_ann_model.keras')
print("Tüm işlemler tamamlandı ve dosyalar kaydedildi!")
