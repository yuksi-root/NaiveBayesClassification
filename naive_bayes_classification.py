import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB # Başarı Oranı =  0.90
from matplotlib.colors import ListedColormap

dataset = pd.read_csv('SosyalMedyaReklamKampanyasi.csv')

# Veri Setini Bağımlı ve Bağımsız Niteliklere Ayrılması
# Niteliklerden bağımsız değişken olarak Eğitim için sadece Yaş ve Tahmini Maaş kullanılacak.
X = dataset.iloc[:, [2,3]].values #yaş ve Tahmini maaş bağımsız değişkeni
y = dataset.iloc[:, 4].values     #Satın alma bağımlı değişkeni 


#Veriyi Eğitim ve Test Olarak Ayrılması
#Veri setinde 400 kayıt var bunun 300’ünü eğitim, 100’ünü test için ayrıldı.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
print("X_test=",len(X_test))
print("y_test=",len(y_test))
print(dataset.iloc[:,[2]].max())#max yaş 60
print(dataset.iloc[:,[2]].min())#min yaş 18
print(dataset.iloc[:,[3]].max())#tahmini max maaş 150000
print(dataset.iloc[:,[3]].min())#tahmini min maaş 15000

#Normalizasyon – Feature Scaling
#Bağımsız değişkenlerden yaş ile tahmini gelir aynı birimde olmadığı için feature scaling uygulanacak
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Naive Bayes Modeli Oluşturmak ve Eğitilmesi
#scikit-learn kütüphanesini kullanarak
#Tahmin etmek için kullandığımız veri sürekli (reel, ondalıklı vs.) oldugundan 
#GaussianNB sınıfından yararlanacagız.
classifier = GaussianNB()
classifier.fit(X_train,y_train)

#Test Seti ile Tahmin Yapılması
#Ayırdığımız test setimizi (X_test) kullanarak oluşturduğumuz model ile tahmin yapalım  
y_tahmin = classifier.predict(X_test)

#Hata Matrisini Oluşturulması
sns.set()
cm = confusion_matrix(y_test, y_tahmin)
mat = confusion_matrix(y_tahmin, y_test)
sns.heatmap(mat.T, square=True, annot=True, fmt='.2g', cbar=False,
            xticklabels='auto', yticklabels='auto')
plt.title("Doğru Değer")
plt.xlabel('Karmaşıklık Matrisi')
plt.ylabel('Tahmini Değer');
plt.show();
print(cm)#cm = karmaşıklık matrisi = confusion_matrix

#Başarı Oranı
accuracy = accuracy_score(y_test, y_tahmin)
print("Başarı Oranı = ",accuracy)

#Eğitim Seti İçin Grafikte Gösterilmesi
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('green', 'red')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())#almayanlar kırmızı bölge alanlar yeşil bölge
colors = np.array(["blue", "black"])#alan kişiler maviler almayanlar siyah
for i, j in enumerate(np.unique(y_set)):
       plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=colors[i] , label = j)
plt.title('Naive Bayes (Eğitim Seti)')
plt.xlabel('Yaş')
plt.ylabel('Maaş')
plt.legend()
plt.show()

#Test Seti İçin Grafikte Gösterilmesi
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('green', 'red')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
                               
for i, j in enumerate(np.unique(y_set)):
       plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                   c = colors[i] , label = j)
plt.title('Naive Bayes (Test Seti)')
plt.xlabel('Yaş')
plt.ylabel('Maaş')
plt.legend()
plt.show()
#65 tane mavi nokta satın alan kişiler
#25 tane siyah nokta satın almayan kişiler
#3 mavi nokta kırmızı bölgede yanlıs sınıflandırma hatası
#7 siyah nokta yeşil bölgede yanlış sınıflandırma hatası
                                
