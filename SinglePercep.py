import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier


X = []
y = []
#Import Pulsar dataset from csv
with open("pulsar_stars.csv") as csv_file:
    csv_read = csv.reader(csv_file, delimiter = ',')
    for line in csv_read:
        #print(line)
        feat = []
        binClass = []
        i = 0
        while i<len(line):
            #print(line[i],i)
            if i == 8:
                num = float(line[i])
                #print(num)
                binClass.append(num)
            else:
                num = float(line[i])
                feat.append(num)
            i+=1

        X.append(feat)
        y.append(binClass)

#print(X[1])
#print(y[1])
#print(y)

y = np.reshape(y,len(y))
scores = []
#Define Single Perceptron Model
hyperparam_val = []
for epoch in range(9):
    if not (epoch == 0):
        for learn_rate in range(8):
            #(hidden units, learning rate, c)
            hyperparam_val.append((0.0001*10**learn_rate,100*epoch))
for param in hyperparam_val:
    print(param)
    sp = Perceptron(alpha = param[0], penalty='l2',max_iter=param[1],class_weight='balanced')
    sp.fit(X,y)
    #print(sp.get_params())
    print(sp.score(X,y))
    scores.append((param,sp.score(X,y)))

max = 0
maxInfo = ()
for score in scores:
    if score[1] > max:
        max = score[1]
        maxInfo = score

print(maxInfo)

bestSP = Perceptron(alpha = maxInfo[0][0],penalty = 'l2', max_iter= maxInfo[0][1], class_weight='balanced')

#Apply Adaptive Boosting(AdaBoost)
adaSP = AdaBoostClassifier(base_estimator= bestSP, n_estimators= 50, learning_rate=.1, algorithm='SAMME')

adaSP.fit(X,y)
yP = adaSP.predict(X)

conMat = confusion_matrix(y,yP)
df_cm = pd.DataFrame(conMat, range(2),
                  range(2))

sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size

print('Adaboost Accuracy:',adaSP.score(X,y))

baggedSP = BaggingClassifier(base_estimator= bestSP, n_estimators= 15, max_samples= 500, max_features= 8, bootstrap= True)
baggedSP.fit(X,y)
print('Bagged Accuracy:',baggedSP.score(X,y))

#sequencedSp = BaggingClassifier(base_estimator= adaSP, n_estimators= 15, max_samples= 500, max_features= 8, bootstrap= True)
#sequencedSp.fit(X,y)
#print('Sequenced Accuracy:', sequencedSp.score(X,y))
