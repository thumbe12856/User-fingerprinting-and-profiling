import numpy as np
import pandas as pd 
from sklearn import tree
from sklearn import metrics
from sklearn import neighbors
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

def readFile(trainingFile, testingFile, domainType = -1):
        """
            dataset:
            +------------------------------------------------------+
            | Index | IP address | Domain | flag | date | IP label |
            |------------------------------------------------------|
            |       |            |        |      |      |          |
            +------------------------------------------------------+

        """
        dataset = pd.read_csv(trainingFile)
        

        """
                domainType:
                true: core domain
                false: support domain
        """
        if(domainType >= 0):
                coreDomain = dataset['flag'] == domainType
                dataset = dataset[coreDomain]


        """

            X:
            +----------------+
            | Index | Domain |
            |----------------|
            |       |        |
            +----------------+

        """
        X=dataset.drop(['IP address','flag','date','IP label'], 1)


        """

            Y:
            +------------------+
            | Index | IP label |
            |------------------|
            |       |          |
            +------------------+

        """
        Y=dataset.drop(['IP address','Domain','flag','date'], 1)

        ''' transform training data to numpy array '''
        X=X.values
        Y=Y.values


        ''' testing data '''
        test = pd.read_csv(testingFile)
        coreDomain = test['flag'] == False
        test = test[coreDomain]

        test_X = test.drop(['IP address', 'flag', 'date', 'label'], 1)
        test_Y = test.drop(['IP address', 'Domain', 'flag', 'date'], 1)

        ''' transform testing data to numpy array '''
        test_X = test_X.values
        test_Y = test_Y.values

        print("1. Read files %s, %s finish." % (trainingFile, testingFile))

        return X, Y, test_X, test_Y


def method(alg, X, Y, test_X, test_Y):
        print '2. Training start.'
        print "2.1 %s training start." % alg

        ''' Predict algorithm '''
        if(alg == "knn"): 
                ''' 1.K-Nearest-Neighbor '''
                K_Nearest_Neighbor = neighbors.KNeighborsClassifier(n_neighbors=590, n_jobs=-1) #for Original Input with label1 0~590
                classifier = K_Nearest_Neighbor.fit(X, Y.ravel())#Original Input with label1 0~9

        elif(alg == "navieBayes"):
                ''' 2.Naive_Bayes '''
                Naive_Bayes = GaussianNB() #for Original Input with label1 0~9
                classifier = Naive_Bayes.fit(X, Y.ravel()) #Original Input with label1 0~9

        elif(alg == "randomForest"):
                ''' 3.Random Forest '''
                Random_Forest = RandomForestClassifier(max_depth=5) 
                classifier = Random_Forest.fit(X, Y)

        elif(alg == "neuralNetwork"):
                ''' 4.Neural Network '''
                Neural_Network = MLPClassifier(solver='adam', hidden_layer_sizes=(10, 3), activation ='relu') 
                classifier = Neural_Network.fit(X, Y) #Original Input with label1 0~9

        print "2.2 %s training finish." % alg

        print '2. Training finish.'


        ''' Start predict '''
        print '3.1 Predicting start.'

        ''' break the data to two datas because it costs lots of memory. '''
        firstHalfTest_X = test_X[:len(test_X) / 2]
        lastHalfTest_X = test_X[-len(test_X) / 2:]
        firstHalfTest_Y = test_Y[:len(test_Y) / 2]
        lastHalfTest_Y = test_Y[-len(test_Y) / 2:]
        
        test_y1_predicted = classifier.predict(firstHalfTest_X)
        test_y2_predicted = classifier.predict(lastHalfTest_X)

        print '3.2 Predicting finish.'

        accuracy1 = metrics.accuracy_score(firstHalfTest_Y, test_y1_predicted)
        accuracy2 = metrics.accuracy_score(lastHalfTest_Y, test_y2_predicted)
        accuracy = (accuracy1 + accuracy2) / 2        

        print "4 accuracy:", accuracy, "\n"
        return accuracy
        