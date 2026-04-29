# Fetal Health Classification with XGBoost & SMOTE

Bu proje, Kardiyotokogram (CTG) verilerini kullanarak fetüs sağlık durumunu üç farklı sınıfa (Normal, Şüpheli, Patolojik) ayıran bir makine öğrenmesi modelidir.

## Projenin Öne Çıkan Özellikleri
* **Sınıf Dengesizliği Çözümü:** Tıbbi verilerdeki aşırı sınıf dengesizliği problemi SMOTE algoritması ile sentetik veri üretilerek dengelenmiştir.
* **Veri Ölçeklendirme:** Sensör verilerindeki farklılıklar StandardScaler ile normalize edilmiştir.
* **Sıfır Tip-II Hata Hedefi:** Modeller arası yarışmada XGBoost şampiyon olmuş ve GridSearchCV ile hiperparametre optimizasyonu yapılmıştır. En kritik tıbbi metrik olan *False Negative* (Hasta bebeğe sağlıklı deme) oranı %0'a indirilmiştir.

## Kullanılan Teknolojiler
* **Dil:** Python 3.13
* **Kütüphaneler:** Scikit-learn, XGBoost, Imbalanced-learn (SMOTE), Pandas, Seaborn
* **Model:** XGBoost Classifier (Accuracy: 97.68%)

## Modeli Kendi Bilgisayarında Çalıştır
Depo içerisinde model dosyaları yer almamaktadır. Programın derlenebilir ve çalıştırılabilir sürümü için "Releases" sekmesindeki app.exe dosyasını indirmeniz yeterlidir.

Demo Ortamı: Kullanıcı Deneyimini bozmamak adına, doktorlardan 21 farklı değeri manuel girmeleri beklenmemiştir. XGBoost'un "Feature Importance" analizine göre karar mekanizmasını %70 oranında etkileyen en kritik 5 değer manuel girişe açılmıştır. Geri kalan 16 parametre, "sağlıklı bir fetüsün ortalama değerleri" ile doldurularak modele iletilmektedir.


<img width="931" height="646" alt="fotoAdsız" src="https://github.com/user-attachments/assets/9608a846-6538-45fa-a31e-5f2f6eba94bb" />

