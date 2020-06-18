import random
import csv
import numpy as np 
from matplotlib import pyplot as plt 

file=open("C:\\Users\\94358\\Desktop\\Abalone\\abalone.csv",'r') 
reader=csv.DictReader(file)
datas=[row for row in reader]

random.shuffle(datas)
n=len(datas)*3//10
test_set=datas[0:n]
train_set=datas[n:]
print(len(datas))


def distance(d1,d2):#欧氏距离
    res=0
    #同维度的变量之间计算平方并累加在一起
    for key in ("sex","length","diameter","height","tweight","hweight","vheight","sweight"):
        res+=(float(d1[key])-float(d2[key]))**2
    #开方得出距离
    return res**0.5


K=8
def knn(data):
    #1.距离
    res=[
        #得出result类别，得出一个距离
        {"result":train['cnumber'],"distance":distance(data,train)}
        for train in train_set
    ]
    

    #2.根据距离进行升序排序
    res=sorted(res,key=lambda item:item['distance'])
    
    #3.取前K个
    res2=res[0:K]
    
    #4.加权平均
    result={'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0,'10':0,'11':0,'12':0,'13':0,'14':0,'15':0,'16':0,'17':0,'18':0,'19':0,'20':0,'21':0,'22':0,'23':0,'24':0,'25':0,'26':0,'27':0,'28':0,'29':0}
   
    sum=0#总距离
    for r in res2:
        sum+=r['distance']

    for r in res2:#距离越近，权数越大
        result[r['result']]+=1-r['distance']/sum
    
    a=(max(result, key=result.get))#使用max返回result字典中权数最大的那个类
    return a
    
   
#测试阶段
correct=0
list1=[]#存放实际环数
list2=[]#存放预测环数
for test in test_set:
    result=test['cnumber']
    list1.append(result)

    result2=knn(test)
    list2.append(result2)

    if result==result2:
        correct+=1

s=len(test_set)
print(correct)
print(len(test_set))
print("准确率：{:.2f}%".format(100*correct/len(test_set)))

#画图
x = np.arange(0,50,1) 
y1= list1[0:50]
y2= list2[0:50]
plt.title("k-nearest neighbor classification") 
plt.xlabel("Test_set samples") 
plt.ylabel("Category labels") 
plt.plot(x,y1,"oy",label='Actual rings of train set')
plt.plot(x,y2,"xk",label='Actual rings of train set')
plt.legend
plt.show()