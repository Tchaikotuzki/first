import pandas as pd
import numpy as np
import random
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier



path='C:\\Users\\94358\\Desktop\\Abalone\\abalone.csv'
data=pd.read_csv(path,encoding='gbk',parse_dates=[0],index_col=0)
data.sort_index(0,ascending=True,inplace=True)

 
d=4177
x=np.zeros((d,8))
y=np.zeros(d)


for i in range(0,d):
      x[i]=np.array(data[i:i+1]\
            [[u'sex',u'length',u'diameter',u'height',u'tweight',u'hweight',u'vheight',u'sweight']]).reshape((1,8))
# print(x.shape)
# print(x)
 
for i in range(0,d):
      y[i]=data.iloc[i][u'cnumber']
# print(np.unique(y))
# print(y)

#解决数据量纲不统一以及分布偏态问题
x=StandardScaler().fit_transform(x)
# datas=pd.DataFrame(x)
# print(datas.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T)


Xtrain,Xtest,Ytrain,Ytest=train_test_split(x,y,test_size=0.3,random_state=420)

# 通过画学习曲线找到最优gamma值
score = []
gamma_range = np.logspace(-10, 1, 50) #返回在对数刻度上均匀间隔的数字
for i in gamma_range:
    clf = SVC(kernel="rbf",gamma = i,cache_size=5000).fit(Xtrain,Ytrain)
    score.append(clf.score(Xtest,Ytest))
    
print(max(score), gamma_range[score.index(max(score))])
plt.plot(gamma_range,score)
plt.show()

# 用学习曲线调参C
score = []
C_range = np.linspace(0.01,30,50)
for i in C_range:
    clf = SVC(kernel="rbf",C=i,gamma = 0.7543120063354607,cache_size=5000).fit(Xtrain,Ytrain)
    score.append(clf.score(Xtest,Ytest))
    
print(max(score), C_range[score.index(max(score))])
plt.plot(C_range,score)
plt.show()