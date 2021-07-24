import csv

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from Pre import *
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm, datasets
from sklearn.multiclass import OneVsRestClassifier
import  time
from sklearn import tree
import pickle

data = pd.read_csv('spotify_training_classification.csv')

data2=data.iloc[:,0:19]
data2=data2.drop(['artists', 'id',  'name','liveness','mode'], axis=1)
duration_scaled = featureScaling(np.array(data2['duration_ms']), 0, 1)
tempo_scaled = featureScaling(np.array(data2['tempo']), 0, 1)
loudness_scaled = featureScaling(np.array(data2['loudness']), 0, -1)
instrumentalness_scaled = featureScaling(np.array(data2['instrumentalness']), 0, 1)
year_scaled = featureScaling(np.array(data2['year']), 0, 1)
###################################################################################
data5=data2.iloc[:,:]
df=pd.DataFrame(data2)
df.year=year_scaled
df.tempo=tempo_scaled
df.loudness=loudness_scaled
df.instrumentalness=instrumentalness_scaled
df.duration_ms=duration_scaled



X=data5.iloc[:,0:14].values

y=data['popularity_level']
ydata=[]
plt.subplots(figsize=(12, 8))
top_corr = data5.corr()
sns.heatmap(top_corr, annot=True)
plt.show()
im=[]
for i in y:
    if(i=='Intermediate'):
        ydata.append(1)
    elif (i=='Low'):
          ydata.append(0)
    else :
        ydata.append(2);
data5['popularity_level']=ydata

high=data5[data5['popularity_level']==2]


im=data5[data5['popularity_level']==1]
print (len(im))
low=data5[data5['popularity_level']==0]
print (len(low))


data3 = high.iloc[:,:]
data3 = data3.drop(['release_date'], axis=1)

Xhigh=data3.iloc[:,0:13].values

miss_mean_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
miss_mean_imputer.fit(Xhigh[:,0:13])
Xhigh[:,0:13]=miss_mean_imputer.transform(Xhigh[:,0:13])


data4 = low.iloc[:,:]
data4 = data4.drop(['release_date'], axis=1)
Xlow=data4.iloc[:,0:13].values

miss_mean_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
miss_mean_imputer.fit(Xlow[:,0:13])
Xlow[:,0:13]=miss_mean_imputer.transform(Xlow[:,0:13])

data5 = im.iloc[:,:]
data5 = data5.drop(['release_date'], axis=1)
Xim=data5.iloc[:,0:13].values

miss_mean_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
miss_mean_imputer.fit(Xim[:,0:13])
Xim[:,0:13]=miss_mean_imputer.transform(Xim[:,0:13])

data2['popularity_level']=ydata
plt.subplots(figsize=(12, 8))
top_corr = data2.corr()
sns.heatmap(top_corr, annot=True)
plt.show()

Xim=np.array(Xim)
Xhigh=np.array(Xhigh)
Xlow=np.array(Xlow)
X2=np.concatenate((Xlow,Xim,Xhigh))
print(X2)
ydata.sort()
print("here")
print(np.count_nonzero(ydata[0:52977]))
plt.subplots(figsize=(12, 8))
top_corr = data5.corr()
sns.heatmap(top_corr, annot=True)
plt.show()
X21=X2[:,0:4]
X22=X2[:,5:7]
X23=X2[:,9:10]

Xdata=np.concatenate((X21,X22,X23),axis=1)
X_train, X_test, y_train, y_test = train_test_split(Xdata, ydata, test_size=0.2, random_state=0, shuffle=True)

"""""
X_train, X_test, y_train, y_test = split(Xdata,ydata)

testeddata=np.concatenate((X_test,y_test),axis=1)
with open('X_test.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerows(testeddata)
"""""


C = 0.01

###################################
starttime1=time.time()
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
Logistic= 'LogisticRegression.sav'
pickle.dump(model, open(Logistic, 'wb'))
EndTime1 =time.time()
print('Training Time of LogisticRegression  = ',  EndTime1- starttime1)
print("finsih log")
#################################
starttimetree=time.time()
clf=tree.DecisionTreeClassifier(max_depth=6)
clf=clf.fit(X_train,y_train)
DecisionTree = 'DecisionTreeClassifier.sav'
pickle.dump(clf, open(DecisionTree, 'wb'))
end = time.time()
print('Training Time of Tree  = ',  end- starttimetree)

print("finsih tree")
########KNN######333
starttime5=time.time()
modelKNN = KNeighborsClassifier(n_neighbors=50)
modelKNN.fit(X_train, y_train)
KNNLoader = 'KNeighborsClassifier.sav'
pickle.dump(modelKNN, open(KNNLoader, 'wb'))
Endtime5=time.time()
print('Training Time of KNN  = ',  Endtime5- starttime5)

print("finsih KNN")
################################
############One vs All
starttime3=time.time()
lin_svc = svm.LinearSVC(C=C).fit(X_train, y_train)
lin_svcloader = 'SVM One To ALL.sav'
pickle.dump(lin_svc, open(lin_svcloader, 'wb'))
Endtime3=time.time()
print('Training Time of SVM ONE TO All  = ',  Endtime3- starttime3)

print("finsih LL")
##################################
starttime2=time.time()
svc = svm.SVC(kernel='rbf').fit(X_train, y_train)
SVConetoone = 'SVMOneToone.sav'
pickle.dump(svc, open(SVConetoone, 'wb'))
Endtime2=time.time()
print('Training Time of SVM ONE TO one  = ',  Endtime2- starttime2)

print("finsih one vs one")

########poly with degree 3#####
starttime4=time.time()
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, y_train)
POLy_Svm = 'SVM.sav'
pickle.dump(poly_svc, open(POLy_Svm, 'wb'))

endtime4=time.time()
print('Training Time of SVM poly  = ',  endtime4- starttime4)

print("finsih poly")
########################

fig = plt.figure(figsize=(10, 5))

plt.bar(['Logistic Regression', 'Decision Tree', 'KNN', 'OnevOne_SVM', 'One Vs All', 'SVM POly'], [ EndTime1- starttime1,  end- starttimetree, Endtime5- starttime5 ,  Endtime2- starttime2,  Endtime3- starttime3,endtime4- starttime4], color='maroon',
        width=0.4)

plt.xlabel("Classification")
plt.ylabel("Training Time")
plt.title("Training Time for each model")
plt.show()


