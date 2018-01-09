import pandas as pd 
import numpy as np
import sys
import model

modelTypes = ['knn', 'navieBayes', 'randomForest', 'neuralNetwork']
timeTypes = [1, 6, 24]
domainTypes = {
    'all': -1,
    'core': True,
    'support': False
}

""" main """
if __name__ == "__main__":
    domainType = -1

    if len(sys.argv) < 3:
        print 'Please enter the command with: --model=[model type] --time=[time series type] --domain=[domain type](optional, default = all domain)'
        sys.exit()
    else:
        if sys.argv[1].startswith('--'):
            option = sys.argv[1][2:7]
            if option == 'model':
                modelType = sys.argv[1][8:]
                if modelType not in modelTypes:
                    print modelType
                    print 'only these model type can used.', modelType
                    sys.exit()
            else:
                print 'Please type --model=[model type].'
                sys.exit()

        if sys.argv[2].startswith('--'):
            option = sys.argv[2][2:6]
            if option == 'time':
                timeType = int(sys.argv[2][7:])
                if timeType not in timeTypes:
                    print timeType
                    print 'only these time type can used.', timeTypes
                    sys.exit()
            else:
                print 'Please type --time=[time series type].'
                sys.exit()

        if(len(sys.argv) == 4):
            if sys.argv[3].startswith('--'):
                option = sys.argv[3][2:8]
                if option == 'domain':
                    domainType = sys.argv[3][9:]
                    if domainType not in domainTypes:
                        print domainType
                        print 'only these domain type can used.', domainTypes
                        sys.exit()
                    else:
                        domainType = domainTypes[domainType]
                else:
                    print 'Please type --domain=[domain type].'
                    sys.exit()

    correctMean = 0
    allFiles = 24 / int(timeType)

    for i in range(allFiles):
        if(timeType == 1):
            ''' 1 hour '''
            trainingFile = './data/dataset_with_label/train_1_hour_with_label/training_data2_1hour_' + str(i) + '.csv'
            testingFile = './data/dataset_with_label/test_1_hour_with_label/training_data2_1hour_' + str(i) + '.csv'

        elif(timeType == 6):
            ''' 6 hours '''
            trainingFile = './data/dataset_with_label/train_6_hour_with_label/training_data2_6hour_' + str(i * 6) + '.csv'
            testingFile = './data/dataset_with_label/test_6_hour_with_label/training_data2_6hour_' + str(i * 6) + '.csv'
            
        elif(timeType == 24):    
            ''' 24 hours '''
            trainingFile = './data/training/training_data2_IP_with_label.csv'
            testingFile = './data/testing/testing_data2_with_label.csv'
        else:
            print 'only these time type can used.', timeTypes
            sys.exit()


        ''' step 1 '''
        ''' first of all, read data from files, and then transform the data to the format. '''
        X, Y, test_X, test_Y = model.readFile(trainingFile, testingFile, domainType)

        ''' step 2 to 4 '''
        ''' training and testing '''
        accuracy = model.method(modelType, X, Y, test_X, test_Y)

        ''' record the accuracy '''
        correctMean += accuracy


    print("5. all file finish, mean accuracy:")
    print(correctMean / float(allFiles))
