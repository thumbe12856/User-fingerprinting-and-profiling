import pandas as pd 
from IPython.display import display,clear_output
import numpy as np
from sklearn import tree
import sys

trainingFile = 'traing_data2.csv'
testingFile = 'testing_data2.csv'

dataset = pd.read_csv(trainingFile)
dataset['IP label'] = dataset.groupby('IP address').grouper.group_info[0] + 1
dataset['IP address'] = dataset['IP address'].astype('str')
# a = dataset[dataset['IP label'] == 1]
# if(len(a) > 0):
#     ipAddress = a['IP address'].unique()[0]
#     print ipAddress

IP_label = dataset[['IP label', 'IP address']].copy()
IP_label = IP_label.drop_duplicates(['IP label'])
IP_label = IP_label.drop_duplicates(['IP address'])
IP_label = IP_label.reset_index(drop=True)
#print IP_label.iloc[0, 1]

X=dataset.drop(['IP address','flag','date','IP label'], 1)
Y=dataset.drop(['IP address','Domain','flag','date'], 1)

''' training data '''
X=X.values
Y=Y.values

''' testing data '''
test = pd.read_csv(testingFile)
test_X = test.drop(['IP address','flag','date'], 1)
test_Y = test.drop(['Domain','flag','date'], 1)

test_X = test_X.values
test_Y = test_Y.values

print '1. Read file finish.'


from sklearn import neighbors
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

''' Predict algorithm '''
''' 1.K-Nearest-Neighbor '''
K_Nearest_Neighbor = neighbors.KNeighborsClassifier(n_neighbors=590, n_jobs=-1) #for Original Input with label1 0~590
print "2.1.2 KNN training start."
classifier1= K_Nearest_Neighbor.fit(X, Y.ravel())#Original Input with label1 0~9
print "2.1.2 KNN training finish."

'''2.Naive_Bayes'''
Naive_Bayes = GaussianNB() #for Original Input with label1 0~9
print "2.2.1 KNN training start."
classifier2 = Naive_Bayes.fit(X, Y.ravel())#Original Input with label1 0~9
print "2.2.1 Naive Bayes training finish."

"""
''' 3.Random Forest '''
Random_Forest = RandomForestClassifier(max_depth=5) 
classifier3 = Random_Forest.fit(X, Y)

''' 4.Support vector machine(SVC) '''
Support_Vector_Machine= svm.SVC(C=2.0,kernel='rbf') 
classifier4 = Support_Vector_Machine.fit(X, Y)

''' 5.Neural Network '''
Neural_Network = MLPClassifier(solver='adam', hidden_layer_sizes=(10, 3), activation ='relu') 
classifier5 = Neural_Network.fit(X, Y)#Original Input with label1 0~9
"""

print '2. Training finish.'


''' Start predict '''
#test_X = test_X.values
#test_Y = test_Y.values

test_y_predicted1 = classifier1.predict(test_X)
test_y_predicted2 = classifier2.predict(test_X)
# test_y_predicted3 = classifier3.predict(test_X)
# test_y_predicted4 = classifier4.predict(test_X)
# test_y_predicted5 = classifier5.predict(test_X)

print '3. Predicting finish.'


''' compute predicting accuracy '''
correct=0
wrong=0
accuracy=0
new_Ip_address_number=0
new_Ip_address=[]
label = []

for i in range(test_Y.shape[0]):
    print i
    print ('correct %d, i :%d' % (correct, i))
    clear_output()
    a = dataset[dataset['IP label'] == test_y_predicted1[i] ]
    if(len(a) > 0):
        print a
        ipAddress = a['IP address'].unique()[0]
        if(ipAddress == test_Y[i]):
            correct = correct + 1
    else:
        new_Ip_address_number = new_Ip_address_number + 1
        new_Ip_address.append(a)
    
accuracy = correct/test_Y.shape[0]
print "4.1 KNN accuracy:", accuracy

correct=0
wrong=0
accuracy=0
new_Ip_address_number=0
new_Ip_address=[]
label = []

for i in range(test_Y.shape[0]):
    print i
    print ('correct %d, i :%d' % (correct, i))
    clear_output()
    a = dataset[dataset['IP label'] == test_y_predicted2[i] ]
    if(len(a) > 0):
        print a
        ipAddress = a['IP address'].unique()[0]
        if(ipAddress == test_Y[i]):
            correct = correct + 1
    else:
        new_Ip_address_number = new_Ip_address_number + 1
        new_Ip_address.append(a)
    
accuracy = correct/test_Y.shape[0]
print "4.2 Naive Bayes accuracy:", accuracy

print '5. all finish'
