import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
class StresAnalizi:
 def __init__(self, dosya_yolu):
 self.df = pd.read_csv(dosya_yolu)
 self.df_temiz = None
 self.X_egitim, self.X_test, self.y_egitim, self.y_test = None, None, None, None
 self.X_pca_egitim, self.X_pca_test = None, None
 self.pca = None
 self.y_tahmin_doğrusal, self.y_tahmin_lojistik, self.y_tahmin_lojistik_pca = None, None, None
 self.basari_doğrusal, self.basari_lojistik, self.basari_lojistik_pca = None, None, None
 self.karısıklık_matrisi_doğrusal, self.karısıklık_matrisi_lojistik, self.karısıklık_matrisi_lojistik_pca = None, None, None
 def ozellikleri_analiz_et(self):
 print("I. Her Özelliği Analiz Edin ve İstatistiklerini Verin")
 print("Toplam özellik sayısı:", len(self.df.columns))
 print("Toplam gözlem sayısı:", len(self.df))
 print(self.df.describe().T)
 print(self.df.info())
 print()
 def verileri_temizle(self):
 print("II. Verilerin Temizlenmesi ve Eğitim İçin Hazırlanması:")
 print("Eksik Veri Sayısı:")
 print(self.df.isnull().sum())
 print()
 self.df_temiz = self.df.fillna(self.df.mean())
 X_doğrusal = self.df_temiz.drop(columns=['stres_seviyesi'])
 y_doğrusal = self.df_temiz['stres_seviyesi']
 X_lojistik = self.df_temiz[['uyku_kalitesi']]
 y_lojistik = self.df_temiz['stres_seviyesi']
 self.X_egitim, self.X_test, self.y_egitim, self.y_test = train_test_split(X_doğrusal, y_doğrusal, test_size=0.2, random_state=42)
 self.X_pca_egitim, self.X_pca_test, _, _ = train_test_split(X_lojistik, y_lojistik, test_size=0.2, random_state=42)
 print("Temizlenmiş Veri Seti:")
 print(self.df_temiz.head())
 print()
 def boyut_azaltma(self):
 print("III. Veri Kümesi Boyut Azaltma:")
 print("III. I. PCA Boyut Küçültme Yöntemini Uygulayın")
 self.pca = PCA(n_components=2)
 X_pca = self.pca.fit_transform(self.X_egitim)
 df_pca = pd.DataFrame(data=X_pca, columns=['Birincil_Bileşen', 'İkincil_Bileşen'])
 df_pca['stres_seviyesi'] = self.y_egitim
 print("PCA Uygulanan Veri Seti:")
 print(df_pca.head())
 print()
 print("II. Soruyu Yeni Azaltılmış Özelliklerle Tekrarlayın:")
 self.X_pca_egitim, self.X_pca_test, _, _ = train_test_split(X_pca, self.y_egitim, test_size=0.2, random_state=42)
 def doğrusal_regresyon(self):
 print("IV. Doğrusal Regresyon ile Stres Seviyesi Tahmini:")
 model_doğrusal = LinearRegression()
 model_doğrusal.fit(self.X_egitim, self.y_egitim)
 self.y_tahmin_doğrusal = model_doğrusal.predict(self.X_test)
 print("PCA'sız Doğrusal Regresyon Modeli MSE:", mean_squared_error(self.y_test, self.y_tahmin_doğrusal))
 def lojistik_regresyon(self):
 print("V. Lojistik Regresyon ile Stres ve Uyku Kalitesi İlişkisi (PCA'sız):")
 model_lojistik = LogisticRegression()
 model_lojistik.fit(self.X_egitim, self.y_egitim)
 self.y_tahmin_lojistik = model_lojistik.predict(self.X_test)
 self.basari_lojistik = accuracy_score(self.y_test, self.y_tahmin_lojistik)
 self.karısıklık_matrisi_lojistik = confusion_matrix(self.y_test, self.y_tahmin_lojistik)
 print("Lojistik Regresyon Accuracy (PCA'sız):", self.basari_lojistik)
 print("Lojistik Regresyon Confusion Matrix (PCA'sız):")
 print(self.karısıklık_matrisi_lojistik)
 def lojistik_regresyon_pca(self):
 print("V. Lojistik Regresyon ile Stres ve Uyku Kalitesi İlişkisi (PCA):")
 model_lojistik_pca = LogisticRegression()
 model_lojistik_pca.fit(self.X_pca_egitim, self.y_egitim)
 self.y_tahmin_lojistik_pca = model_lojistik_pca.predict(self.X_pca_test)
 self.basari_lojistik_pca = accuracy_score(self.y_test, self.y_tahmin_lojistik_pca)
 self.karısıklık_matrisi_lojistik_pca = confusion_matrix(self.y_test, self.y_tahmin_lojistik_pca)
 print("Lojistik Regresyon Accuracy (PCA):", self.basari_lojistik_pca)
 print("Lojistik Regresyon Confusion Matrix (PCA):")
 print(self.karısıklık_matrisi_lojistik_pca)
 def sonuçları_görselleştir(self):
 print("VI. Sonuçları Görselleştirme:")
 print("IV. I. Doğrusal Regresyon Performans Sonuçları:")
 df_sonuçlar_doğrusal = pd.DataFrame({'Gerçek': self.y_test, 'Tahmin Edilen': self.y_tahmin_doğrusal})
 sns.scatterplot(x='Gerçek', y='Tahmin Edilen', data=df_sonuçlar_doğrusal)
 plt.xlabel('Gerçek Değerler')
 plt.ylabel('Tahmin Edilen Değerler')
 plt.title('Doğrusal Regresyon: Gerçek vs Tahmin Edilen Değerler')
 plt.show()
 print("V. I. Lojistik Regresyon Performans Sonuçları:")
 doğruluk_lojistik = model_lojistik.score(self.X_test, self.y_test)
 karısıklık_matrisi_lojistik = confusion_matrix(self.y_test, self.y_tahmin_lojistik)
 print("Lojistik Regresyon Accuracy:", doğruluk_lojistik)
 print("Lojistik Regresyon Confusion Matrix:")
 print(karısıklık_matrisi_lojistik)
 print("V. II. Lojistik Regresyon Performans Sonuçları (PCA uygulandı):")
 doğruluk_lojistik_pca = accuracy_score(self.y_test, self.y_tahmin_lojistik_pca)
 karısıklık_matrisi_lojistik_pca = confusion_matrix(self.y_test, self.y_tahmin_lojistik_pca)
 print("Lojistik Regresyon Accuracy (PCA):", doğruluk_lojistik_pca)
 print("Lojistik Regresyon Confusion Matrix (PCA):")
 print(karısıklık_matrisi_lojistik_pca)
stres_analizi = StresAnalizi("StresSeviyesiVeriSeti.csv")
stres_analizi.ozellikleri_analiz_et()
stres_analizi.verileri_temizle()
stres_analizi.boyut_azaltma()
stres_analizi.doğrusal_regresyon()
stres_analizi.lojistik_regresyon()
stres_analizi.lojistik_regresyon_pca()
stres_analizi.sonuçları_görselleştir()