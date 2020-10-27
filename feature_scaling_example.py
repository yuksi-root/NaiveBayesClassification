import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
#os.chdir('C:/Users/yuksel/Desktop/CAN/1)Programming/NaiveBayesClassification')
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB # Başarı Oranı =  0.90
from matplotlib.colors import ListedColormap

dataset = pd.read_csv('SosyalMedyaReklamKampanyasi.csv')

# Veri Setini Bağımlı ve Bağımsız Niteliklere Ayırmak 
# Niteliklerden bağımsız değişken olarak Eğitim için sadece Yaş ve Tahmini Maaş kullanılacak.
X = dataset.iloc[:, [2]].values #yaş ve Tahmini maaş bağımsız değişkeni
y = dataset.iloc[:, 4].values     #Satın alma bağımlı değişkeni 


#Veriyi Eğitim ve Test Olarak Ayırmak
#Veri setinde 400 kayıt var bunun 300’ünü eğitim, 100’ünü test için ayrıldı.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#print("X_test=",len(X_test))
#print(X_test)
#print("X_train",len(X_train))
print(X_test)
sapma = X_train.std()
ort = X_train.mean() 
ornekTest =( 30 - ort) / (sapma) #formül (X - ort) / sapma
print("X_Test_Normalizasyonu",ornekTest) 
print("ort",ort)   
print("sapma",sapma)
print("enkücük",X_test.min())
print("enbüyük",X_test.max())
ornekTrain = (44 - ort) / sapma
print("X_train_Normu",ornekTrain)
#print("y_test=",len(y_test))#orjinal test verisi
#print("y_train",len(y_train))#orjinal train verisi
#print(dataset.iloc[:,[2]].max())#max yaş 60
#print(dataset.iloc[:,[2]])
#print(dataset.iloc[:,[2]].min())#min yaş 18
#print(dataset.iloc[:,[3]].max())#tahmini max maaş 150000
#print(dataset.iloc[:,[3]].min())#tahmini min maaş 15000

#Normalizasyon – Feature Scaling
#Bağımsız değişkenlerden yaş ile tahmini gelir aynı birimde olmadığı için feature scaling uygulanacak
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#print(X_test)