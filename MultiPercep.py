import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.neural_network import MLPClassifier
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

mp = MLPClassifier(hidden_layer_sizes=[16,8,6,4,2], activation='tanh', solver = 'adam', alpha= .0001, learning_rate = 'adaptive', learning_rate_init=.001)
mp.fit(X,y)
print(mp.score(X,y))

yP = mp.predict(X)

conMat = confusion_matrix(y,yP)
df_cm = pd.DataFrame(conMat, ['Not Pulsar', 'Pulsar'],['Not Pulsar', 'Pulsar'])

sn.set(font_scale=1)#for label size
ax = sn.heatmap(df_cm, annot=True,fmt= 'd',annot_kws={"size": 16})# font size
ax.set(xlabel = 'Predicted Value', ylabel = 'True Label')
plt.show()