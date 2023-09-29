
import numpy as np 
import tensorflow as  tf 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

data=pd.read_csv('E:\Churn_prediction/Churn_Modelling.csv')
data.head()

data=data.drop(columns=['RowNumber','CustomerId','Surname'])
data.head()


data['Gender']=data['Gender'].apply(lambda x :  0 if x=='Female' else 1)
data['Gender']=data['Gender'].astype(int)
data.head()



from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
data['Geography']=label_encoder.fit_transform(data['Geography'])
data.head()


value_counts=data['Exited'].value_counts()
plt.pie(value_counts, labels=['Not Exited', 'Exited'], autopct='%1.1f%%', colors=sns.color_palette('Set3'))


value_counts

X=data.drop('Exited',axis=1)
y=data['Exited']

import numpy as np
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

#print("Class distribution before oversampling:", Counter(y))

ros = RandomOverSampler(random_state=42)

X, y = ros.fit_resample(X, y)

#print("Class distribution after oversampling:", Counter(y))

X=np.array(X)
X=(X-X.mean())/X.std()


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


#Random Forests
random_forest_model = RandomForestClassifier()
random_forest_model.fit(x_train, y_train)
random_forest_pred = random_forest_model.predict(x_test)

print("Random Forests:")
print("Accuracy:", accuracy_score(y_test, random_forest_pred))
print("Classification Report:\n", classification_report(y_test, random_forest_pred))

