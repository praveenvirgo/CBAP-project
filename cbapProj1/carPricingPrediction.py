import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

cars = pd.read_csv("/home/praveen/learnPython/Finalprojects_DS-master/Car_pricing_prediction/CarPrice_Assignment.csv")
#print(cars.info())
#print(type( cars ) )
#print(cars.head())
#print(cars['symboling'].astype('category').value_counts())
riskData = cars['symboling'].astype('category').value_counts()
#s = riskData.iloc[:, 0] 
#print(type( riskData ) )
#index = np.arange(len(riskData[:, 0]))
#print(index)
#print( cars['aspiration'].astype('category').value_counts())
#print(cars['aspiration'].head())
#print(cars['drivewheel'].astype('category').value_counts())
#print(cars['drivewheel'].head())
#print(cars['wheelbase'].head())
#sns.distplot(cars['symboling'])
#plt.savefig("riskData.png")
#sns.distplot(cars['wheelbase'])
#plt.savefig("wheelData.png")
#sns.distplot(cars['curbweight'])
#plt.savefig("curbData.png")
cars_numeric = cars.select_dtypes(include=['float64', 'int'])
#print(carsNumeric.head())
cars_numeric = cars_numeric.drop(['symboling', 'car_ID'], axis=1)
#print(carsNumeric.head())
plt.figure(figsize=(20, 10))
sns.pairplot(cars_numeric)
plt.savefig("mult.png")
#print(carsNumeric.shape[1])
plt.show()
cor = cars_numeric.corr()
#print(cor)
plt.figure(figsize=(16,8))
sns.heatmap(cor, cmap="YlGnBu", annot=True)
#plt.show()
plt.savefig("heatdis.png")
cars['symboling'] = cars['symboling'].astype('object')
#print( cars['CarName'][:30] )
carnames = cars['CarName'].apply(lambda x: x.split(" ")[0])
#print(carnames[:30])
import re
p = re.compile(r'\w+-?\w+')
carnames = cars['CarName'].apply(lambda x: re.findall(p, x)[0])
#print(carnames)
cars['car_company'] = cars['CarName'].apply(lambda x: re.findall(p, x)[0])
#print(cars['car_company'].astype('category').value_counts())
#print(cars.columns)
cars.loc[(cars['car_company'] == "vw") | (cars['car_company'] == "vokswagen"), 'car_company'] = 'volkswagen'
cars.loc[cars['car_company']=="porcshce", 'car_company'] = 'porsche'
cars.loc[ cars['car_company'] == "toyouta", 'car_company'] = 'toyota'
cars.loc[ cars['car_company'] == "Nissan", 'car_company'] = 'nissan'
cars.loc[cars['car_company'] == "maxda", 'car_company'] = 'mazda'
cars = cars.drop('CarName', axis=1)
#print(cars.columns)
X = cars.loc[:, ['symboling', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'fuelsystem', 'boreratio', 'stroke', 'cpmpressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'car_company']]
y = cars['price']
cars_categorical = X.select_dtypes(include=['object'])
#print(cars_categorical.head())
cars_dummies = pd.get_dummies(cars_categorical, drop_first=True)
X = X.drop(list(cars_categorical.columns), axis=1)
X = pd.concat([X, cars_dummies], axis=1)
from sklearn.preprocessing import scale
cols = X.columns
#print(X.head())
X = pd.DataFrame(scale(X))
X.columns = cols
#print(X.head())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
lm = LinearRegression()
#lm.fit(X_train, y_train)
