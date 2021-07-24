import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Pre import *
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm, datasets
from sklearn.neighbors import KNeighborsClassifier

import  time
from sklearn import tree
import pickle



data = pd.read_csv('data_testing_classification.csv')

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
low=data5[data5['popularity_level']==0]


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


Xim=np.array(Xim)
Xhigh=np.array(Xhigh)
Xlow=np.array(Xlow)
X2=np.concatenate((Xlow,Xim,Xhigh))
ydata.sort()
X21=X2[:,0:4]
X22=X2[:,5:7]
X23=X2[:,9:10]

Xdata=np.concatenate((X21,X22,X23),axis=1)

#X_train, X_test, y_train, y_test = train_test_split(Xdata, ydata, test_size=0.2, random_state=0, shuffle=True)

###################################
loaded_model = pickle.load(open('LogisticRegression.sav', 'rb'))
start_testing_logistic_regression = time.time()
Y_prediction_logistic_regression = loaded_model.predict(Xdata)
stop_testing_logistic_regression = time.time()
print("1- Testing Time of logistic_regression = ", stop_testing_logistic_regression - start_testing_logistic_regression)
#print('2- Mean Square Error of logistic_regression ',
    #  metrics.mean_squared_error(np.asarray(y_test), Y_prediction_logistic_regression))
#print('3- R2 Score of logistic_regression = ', loaded_model.score(y_test, Y_prediction_logistic_regression))
accuracy_logistic_regression = loaded_model.score(Xdata, ydata) * 100
print("4- Accuracy of logistic_regression = " + str(accuracy_logistic_regression))



#################################
#svm_model_linear_ovr = OneVsRestClassifier(SVC(kernel='linear', C=C)).fit(X_train, y_train)
#svm_predictions = svm_model_linear_ovr.predict(X_test)

# model accuracy for X_test
#accuracy = svm_model_linear_ovr.score(X_test, y_test)
#print('One VS Rest SVM accuracy: ' + str(accuracy))

loaded_model = pickle.load(open('DecisionTreeClassifier.sav', 'rb'))
start_testing_decision_tree = time.time()
Y_prediction_decision_tree = loaded_model.predict(Xdata)
stop_testing_decision_tree = time.time()
print("1- Testing Time of decision_tree = ", stop_testing_decision_tree - start_testing_decision_tree)
accuracy_decision_tree = loaded_model.score(Xdata, ydata) * 100
print("4- Accuracy of decision_tree = " + str(accuracy_decision_tree))
######################KNN
loaded_model = pickle.load(open('KNeighborsClassifier.sav', 'rb'))
start_testing_knn = time.time()
Y_prediction_knn = loaded_model.predict(Xdata)
stop_testing_knn = time.time()
print("1- Testing Time of knn = ", stop_testing_knn - start_testing_knn)

#print('2- Mean Square Error of knn ', metrics.mean_squared_error(np.asarray(y_test), Y_prediction_knn))
#print('3- R2 Score of knn = ', r2_score(y_test, Y_prediction_knn))
accuracy_knn = loaded_model.score(Xdata, ydata) * 100
print("4- Accuracy of Knn = " + str(accuracy_knn))

##########################
############One vs All
loaded_model = pickle.load(open('SVM One To ALL.sav', 'rb'))
result = loaded_model.score(Xdata, ydata)
Y_prediction_logistic_regression = loaded_model.predict(Xdata)
start_testing_svm_all = time.time()
Y_prediction_svm_ovall= loaded_model.predict(Xdata)
stop_testing_svm_all = time.time()
print("1- Testing Time of One.v.all SVM = ", stop_testing_svm_all - start_testing_svm_all)

#print('2- Mean Square Error of One.v.all SVM ', metrics.mean_squared_error(np.asarray(y_test), Y_prediction_svm_ovall))
#print('3- R2 Score of One.v.all SVM = ', r2_score(y_test, Y_prediction_svm_all))
accuracy_svm_ovall = loaded_model.score(Xdata, ydata) * 100
print("4- Accuracy of One.v.all SVM = " + str(accuracy_svm_ovall))

Endtime3=time.time()




##################################

loaded_model = pickle.load(open('SVMOneToone.sav', 'rb'))
start_testing_svm_ovo = time.time()
Y_prediction_svm_ovo = loaded_model.predict(Xdata)
stop_testing_svm_ovo = time.time()
print("1- Testing Time of One.v.One SVM = ", stop_testing_svm_ovo - start_testing_svm_ovo)

#print('2- Mean Square Error of One.v.One SVM ', metrics.mean_squared_error(np.asarray(y_test), Y_prediction_svm_ovo))
#print('3- R2 Score of One.v.One SVM = ', r2_score(y_test, Y_prediction_svm_ovo))
accuracy_svm_ovo = loaded_model.score(Xdata, ydata) * 100
print("4- Accuracy of One.v.One SVM = " + str(accuracy_svm_ovo))

########poly with degree 3#####

starttime4=time.time()
loaded_model = pickle.load(open('SVM.sav', 'rb'))
result = loaded_model.score(Xdata, ydata)*100
Y_prediction_logistic_regression = loaded_model.predict(Xdata)
endtime4=time.time()
print("5- Time of poly SVM = " + str(endtime4-starttime4))

print("5- Accuracy of poly SVM = " + str(result))

########################


fig = plt.figure(figsize=(10, 10))

plt.bar(['Logistic Regression', 'Decision Tree', 'KNN', 'OnevOne_SVM', 'One Vs All', 'SVM POly'], [ stop_testing_logistic_regression - start_testing_logistic_regression, stop_testing_decision_tree - start_testing_decision_tree, stop_testing_knn - start_testing_knn,stop_testing_svm_ovo - start_testing_svm_ovo,stop_testing_svm_all - start_testing_svm_all,endtime4-starttime4], color=['red', 'green','maroon','blue','cyan'],
        width=0.3)

plt.xlabel("Classification")
plt.ylabel("Testing Time")
plt.title("Testing Time for each model")
plt.show()

# Bar Plot
fig = plt.figure(figsize=(10, 5))
plt.ylim(70,100)
plt.bar(['Logistic Regression', 'Decision Tree', 'KNN', 'One.v.One SVM', 'Polynomial SVM'],
        [accuracy_logistic_regression, accuracy_decision_tree, accuracy_knn, accuracy_svm_ovo, result], color=  ['red', 'green','maroon','blue','cyan'],
        width=0.3)

plt.xlabel("Classification")
plt.ylabel("Accuracy")
plt.title("Accuracy for each model")
plt.show()

