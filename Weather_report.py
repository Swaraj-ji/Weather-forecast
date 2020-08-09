from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

#Real dataset is very large ,so extracting 150 data points as our use dataset
data_set = pd.read_csv("weatherHistory.csv")
use, not_use = train_test_split(data_set,train_size=200, random_state=2)   #I used random_state because to get the same dataset every time.

#Eleminating some non-useful features
new_use = use.drop(["Formatted Date","Loud Cover","Wind Bearing (degrees)","Daily Summary","Pressure (millibars)","Apparent Temperature (C)"], axis=1)
new_use = new_use.dropna()    #dropping those rows which have same null data.
print("Shape of our total dataset:",new_use.shape)          #After dropping checking, how much rows are dropped.

#since we are using our own data set so we cannot use .data or .target methods instead we have to do explicitly
new_use_1 = pd.DataFrame(new_use[["Temperature (C)","Humidity","Wind Speed (km/h)","Visibility (km)"]])
new_use_2 = pd.DataFrame(new_use[["Precip Type","Summary"]])
X = new_use_1.to_numpy()    #The input data
y = new_use_2.to_numpy()    #The output data
print("Shape of our X and y:",X.shape,y.shape)           #shape of X and Y

#splitting into train and test subset
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=1)
print("Shape of X_train and y_train:",X_train.shape,y_train.shape)
print("Shape of X_test and y_test:",X_test.shape,y_test.shape)

#Applying algo. and fitting our training set into the algo.
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

#Prediction of output
a = np.array([(40.4722,0.89,7.1197,15.8263)])    #User Input 1.Temperature(C) 2.Humidity 3.Wind speed(km\hr) 4.Visibility(km)
y_pred = knn.predict(a)

#User friendly printed statment
b = ["Precip Type","Summary"]
c = []
c.append(y_pred[0][0])
c.append(y_pred[0][1])
zipped = dict(zip(b,c))
print(zipped)