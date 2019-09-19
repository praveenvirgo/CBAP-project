import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
cars = pd.read_csv("~/learnPython/Finalprojects_DS-master/Car_pricing_prediction/CarPrice_Assignment.csv")              
cars_numeric = cars.select_dtypes(include=['float64', 'int'])
cars_numeric = cars_numeric.drop(['symboling', 'car_ID'], axis=1)
cor = cars_numeric.corr()
cars['symboling'] = cars['symboling'].astype('object')
carnames = cars['CarName'].apply(lambda x: x.split(" ")[0])
import re
p = re.compile(r'\w+-?\w+')
carnames = cars['CarName'].apply(lambda x: re.findall(p, x)[0])
cars['car_company'] = cars['CarName'].apply(lambda x: re.findall(p, x)[0])
cars.loc[(cars['car_company'] == "vw") | 
                         (cars['car_company'] == "vokswagen")
                                  , 'car_company'] = 'volkswagen'

# porsche
cars.loc[cars['car_company'] == "porcshce", 'car_company'] = 'porsche'

# toyota
cars.loc[cars['car_company'] == "toyouta", 'car_company'] = 'toyota'

# nissan
cars.loc[cars['car_company'] == "Nissan", 'car_company'] = 'nissan'

# mazda
cars.loc[cars['car_company'] == "maxda", 'car_company'] = 'mazda'
cars = cars.drop('CarName', axis=1)
X = cars.loc[:, ['symboling', 'fueltype', 'aspiration', 'doornumber',
               'carbody', 'drivewheel', 'enginelocation', 'wheelbase', 'carlength',
                      'carwidth', 'carheight', 'curbweight', 'enginetype', 'cylindernumber',
                             'enginesize', 'fuelsystem', 'boreratio', 'stroke', 'compressionratio',
                                    'horsepower', 'peakrpm', 'citympg', 'highwaympg',
                                           'car_company']]

y = cars['price']
cars_categorical = X.select_dtypes(include=['object'])
cars_dummies = pd.get_dummies(cars_categorical, drop_first=True)
X = X.drop(list(cars_categorical.columns), axis=1)
X = pd.concat([X, cars_dummies], axis=1)
from sklearn.preprocessing import scale
cols = X.columns
X = pd.DataFrame(scale(X))
X.columns = cols
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                                    train_size=0.7,
                                                                                                                        test_size = 0.3, random_state=100)
lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)
from sklearn.metrics import r2_score
#print(r2_score(y_test, y_pred))
from sklearn.feature_selection import RFE
lm = LinearRegression()
rfe_15 = RFE(lm, 15)
rfe_15.fit(X_train, y_train)
y_pred = rfe_15.predict(X_test)

from sklearn.feature_selection import RFE
lm = LinearRegression()
rfe_6 = RFE(lm, 6)

# fit with 6 features
rfe_6.fit(X_train, y_train)

# predict
y_pred = rfe_6.predict(X_test)

import statsmodels.api as sm
col_15 = X_train.columns[rfe_15.support_]
X_train_rfe_15 = X_train[col_15]
X_train_rfe_15 = sm.add_constant(X_train_rfe_15)
print(X_train_rfe_15.head())
lm_15 = sm.OLS(y_train, X_train_rfe_15).fit()
#print(lm_15.summary())
X_test_rfe_15 = X_test[col_15]


# # Adding a constant variable 
X_test_rfe_15 = sm.add_constant(X_test_rfe_15, has_constant='add')
#X_test_rfe_15.info()
#
#
## # Making predictions
y_pred = lm_15.predict(X_test_rfe_15)
r2_score(y_test, y_pred)
## subset the features selected by rfe_6
col_6 = X_train.columns[rfe_6.support_]
#
# subsetting training data for 6 selected columns
X_train_rfe_6 = X_train[col_6]

# add a constant to the model
X_train_rfe_6 = sm.add_constant(X_train_rfe_6)


# fitting the model with 6 variables
lm_6 = sm.OLS(y_train, X_train_rfe_6).fit()   
#print(lm_6.summary())


# making predictions using rfe_6 sm model
X_test_rfe_6 = X_test[col_6]


# Adding a constant  
X_test_rfe_6 = sm.add_constant(X_test_rfe_6, has_constant='add')
X_test_rfe_6.info()


# # Making predictions
y_pred = lm_6.predict(X_test_rfe_6)


n_features_list = list(range(4, 20))
adjusted_r2 = []
r2 = []
test_r2 = []

for n_features in range(4, 20):

            # RFE with n features
                lm = LinearRegression()
                rfe_n = RFE(lm, n_features)

                # specify number of features
                rfe_n = RFE(lm, n_features)

                     # fit with n features
                rfe_n.fit(X_train, y_train)

#                     # subset the features selected by rfe_6
                col_n = X_train.columns[rfe_n.support_]
#
              # subsetting training data for 6 selected columns
                X_train_rfe_n = X_train[col_n]
#
#                     # add a constant to the model
                X_train_rfe_n = sm.add_constant(X_train_rfe_n)
                lm_n = sm.OLS(y_train, X_train_rfe_n).fit()
                adjusted_r2.append(lm_n.rsquared_adj)
                r2.append(lm_n.rsquared)
#
#
               # fitting the model with 6 variables
                r2.append(lm_n.rsquared)
#                     
#                         
#                  # making predictions using rfe_15 sm model
                X_test_rfe_n = X_test[col_n]
#
#
#                 # # Adding a constant variable 
                X_test_rfe_n = sm.add_constant(X_test_rfe_n, has_constant='add')
#
#
#
#                # # Making predictions
                y_pred = lm_n.predict(X_test_rfe_n)
#
                test_r2.append(r2_score(y_test, y_pred))
#
#plt.figure(figsize=(10, 8))
#plt.plot(n_features_list, adjusted_r2, label="adjusted_r2")
##plt.plot(n_features_list, r2, label="train_r2")
#plt.plot(n_features_list, test_r2, label="test_r2")
#plt.legend(loc='upper left')
#plt.savefig("pred.png")
##plt.show()
##c = [i for i in range(len(y_pred))]
##fig = plt.figure()
##plt.plot(c,y_test-y_pred, color="blue", linewidth=2.5, linestyle="-")
##fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
##plt.xlabel('Index', fontsize=18)                      # X-label
##plt.ylabel('ytest-ypred', fontsize=16)                # Y-label
##plt.show()
#fig = plt.figure()
#sns.distplot((y_test-y_pred),bins=50)
#fig.suptitle('Error Terms', fontsize=20)                  # Plot heading 
#plt.xlabel('y_test-y_pred', fontsize=18)                  # X-label
#plt.ylabel('Index', fontsize=16)                          # Y-label
#plt.show()
#plt.savefig("error.png")
#sns.distplot(cars['price'],bins=50)
#plt.savefig("error.png")
predictors = ['carwidth', 'curbweight', 'enginesize', 
                             'enginelocation_rear', 'car_company_bmw', 'car_company_porsche']

cors = X.loc[:, list(predictors)].corr()
sns.heatmap(cors, annot=True)
plt.savefig("fin.png")
plt.show()
