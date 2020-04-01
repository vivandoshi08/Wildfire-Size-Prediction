import pandas as pd

df =pd.read_csv('forestfires.csv', na_values=['NA'])

import numpy as np
import matplotlib.pyplot as plt
df = df[(df['area'] >0)]
df.area=np.log(df.area+1)
X = df.iloc[:, 8]
y = df.iloc[:, 12]
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X.values.reshape(-1,1))
y = sc_y.fit_transform(y.values.reshape(-1,1))
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', context='notebook')
cols = ['temp', 'RH', 'wind', 'rain','area','FFMC','DMC']
sns.pairplot(df[cols], size=2.5);
plt.show()

import numpy as np
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
cbar=True,
annot=True,
square=True,
fmt='.2f',
annot_kws={'size': 11},
yticklabels=cols,
xticklabels=cols)
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
slr=LinearRegression()
slr.fit(X_train,y_train)
y_train_pred=slr.predict(X_train)
y_test_pred=slr.predict(X_test)
mean_squared_error(y_test,y_test_pred)

plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_test_pred, color='blue', linewidth=3)
plt.show()