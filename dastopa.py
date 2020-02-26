import pandas as pd  
import matplotlib.pyplot as plt

df1 = pd.read_csv('click_train.csv')

#df2 = pd.read_csv('ad.csv')
#df3 = pd.merge(df1, df2, on='adId')

df4 = pd.read_csv('ad_title.csv')
df5 = pd.merge(df1, df4, on='adId')

df = df5.drop(['clicked'], axis=1)
#df = df.drop(['adId'], axis=1)
#df = df.drop(['displayId'], axis=1)
target = df5['clicked']

from sklearn.model_selection import train_test_split
X_train,x_test,Y_train,y_test = train_test_split(df,target,test_size = 0.6,random_state = 0)


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,Y_train)
pr = dtc.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
#print(confusion_matrix(y_test,rfc.predict(x_test)))
#print(classification_report(y_test,rfc.predict(x_test)))
print(accuracy_score(y_test,pr))
