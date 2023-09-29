import pandas as pd
import numpy as np

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')



train_data = pd.read_csv('E:\credit card fraud prediction/fraudTrain.csv')
test_data = pd.read_csv('E:\credit card fraud prediction/fraudTest.csv')


train_data.head()

plt.figure(figsize=(8, 6))
sns.countplot(x='is_fraud', data=pd.concat([train_data, test_data], ignore_index=True))
plt.title('Data Distribution')
plt.xlabel(' (0:Not Fraud  | 1:Fraud) ')
plt.ylabel('Count')
#plt.show()

train_data.info()

test_data.info()

train_data.isnull().sum(),test_data.isnull().sum()

cols_to_drop = ['Unnamed: 0','cc_num','merchant','first','last','trans_num','unix_time','street','category']
train_data.drop(columns=cols_to_drop,inplace = True)
test_data.drop(columns=cols_to_drop,inplace = True)
#print(train_data.shape)
#print(test_data.shape)

train_data['lat_dist'] = abs(round(train_data['merch_lat']-train_data['lat'],2))
train_data['long_dist'] = abs(round(train_data['merch_long']-train_data['long'],2))

test_data['lat_dist'] = abs(round(test_data['merch_lat']-test_data['lat'],2))
test_data['long_dist'] = abs(round(test_data['merch_long']-test_data['long'],2))
cols_to_drop = ['trans_date_trans_time','city','lat','long','job','dob','merch_lat','merch_long','state']
train_data.drop(columns=cols_to_drop,inplace = True)
test_data.drop(columns=cols_to_drop,inplace = True)
#train_data.head()


train_data.gender =[ 1 if value == "M" else 0 for value in train_data.gender]
test_data.gender =[ 1 if value == "M" else 0 for value in test_data.gender]
#train_data.head()

X_train = train_data.drop('is_fraud',axis=1)
X_test = test_data.drop('is_fraud',axis=1)
y_train = train_data['is_fraud']
y_test = test_data['is_fraud']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_trian = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy using Logistic Regression: {accuracy:.2f}')

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state = 45)
dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy using Decision Tree: {accuracy:.2f}')

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy using RandomForest: {accuracy:.2f}')

import xgboost as xgb
xgbclf = xgb.XGBClassifier(learning_rate=0.1,n_estimators=100,max_depth=3,objective='binary:logistic',random_state=45)
xgbclf.fit(X_train,y_train)
y_pred = xgbclf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy using xgboost: {accuracy:.2f}')



