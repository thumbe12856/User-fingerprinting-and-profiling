import numpy as np
import pandas as pd 
import sys
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from pathlib import Path

def getTTvalue(allTrainingDataset, allTestingDataset):
    returnData = pd.DataFrame()
    for trainDataIP in allTrainingDataset['IP'].index:
        dataset3 = allTestingDataset.copy()
        as_list = dataset3.index.tolist()
        if(trainDataIP in as_list):
            idx = as_list.index(trainDataIP)
            as_list[idx] = 'South Korea'
            dataset3.index = as_list
        allData = allTrainingDataset.append(dataset3).fillna(0)
        returnData = returnData.append(allData[allData.index == trainDataIP])
    return returnData

timeTypes = [1, 6, 24]
domainTypes = {
    'all': -1,
    'core': True,
    'support': False
}

""" main """
if __name__ == "__main__":
    domainType = -1

    if len(sys.argv) < 2:
        print 'Please enter the command with: --time=[time series type] --domain=[domain type](optional, default = all domain)'
        sys.exit()

    else:
        if sys.argv[1].startswith('--'):
            option = sys.argv[1][2:6]
            if option == 'time':
                timeType = int(sys.argv[1][7:])
                if timeType not in timeTypes:
                    print timeType
                    print 'only these time type can used.', timeTypes
            else:
                print 'Please type --time=[time series type].'
                sys.exit()

        if(len(sys.argv) == 3):
            if sys.argv[2].startswith('--'):
                option = sys.argv[2][2:8]
                if option == 'domain':
                    domainType = sys.argv[2][9:]
                    if domainType not in domainTypes:
                        print domainType
                        print 'only these domain type can used.', domainTypes
                    else:
                        domainType = domainTypes[domainType]
                else:
                    print 'Please type --domain=[domain type].'
                    sys.exit()

    correctMean = 0
    allFiles = 24 / int(timeType)
    dataset1Len = 0
    dataset2Len = 0
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

        trainingFileExist = Path(trainingFile)
        testingFileExist = Path(testingFile)
        if(trainingFileExist.exists() & testingFileExist.exists()):

            """ reaed testing  and training file """
            column_names = ['IP', 'Domain', 'flag', 'date', 'IP label']
            dataset1 = pd.read_csv(trainingFile)
            dataset2 = pd.read_csv(testingFile)

            if(domainType >= 0):
                coreDomain = dataset1['flag'] == domainType
                dataset1 = dataset1[coreDomain]
                coreDomain = dataset2['flag'] == domainType
                dataset2 = dataset2[coreDomain]

            dataset1Len += len(dataset1)
            dataset2Len += len(dataset2)
            dataset1.columns = column_names
            dataset2.columns = column_names
            print("1. Read files %s finish." % trainingFile)

            """

                allDataset:
                +-------------------+
                |   training data   |
                +-------------------+
                |   testing  data   |
                +-------------------+

            """
            allDataset = dataset1.append(dataset2)
            allTrainingDataset = pd.DataFrame({'count': dataset1.groupby(['IP', 'Domain']).size()}).unstack().reset_index().fillna(0).set_index('IP')
            allTestingDataset = pd.DataFrame({'count': dataset2.groupby(['IP', 'Domain']).size()}).unstack().reset_index().fillna(0).set_index('IP')


            """

                realTrainX: when in training data time
                +---------------------------------------------------------------------------------+
                |      | visit domain 1 times | visit domain 2 times | ... | visit domain N times |
                |---------------------------------------------------------------------------------|
                | IP 1 |                      |                      |     |                      |
                |---------------------------------------------------------------------------------|
                | IP 2 |                      |                      |     |                      |
                |---------------------------------------------------------------------------------|
                |                                        .                                        | 
                |                                        .                                        |
                |                                        .                                        |
                |---------------------------------------------------------------------------------|
                | IP N |                      |                      |     |                      |
                +---------------------------------------------------------------------------------+

            """
            realTrainX = getTTvalue(allTrainingDataset, allTestingDataset)
            print("2.1 Transform trainging data to X finish.")


            """

                realTestY: when in testing data time
                +---------------------------------------------------------------------------------+
                |      | visit domain 1 times | visit domain 2 times | ... | visit domain N times |
                |---------------------------------------------------------------------------------|
                | IP 1 |                      |                      |     |                      |
                |---------------------------------------------------------------------------------|
                | IP 2 |                      |                      |     |                      |
                |---------------------------------------------------------------------------------|
                |                                        .                                        | 
                |                                        .                                        |
                |                                        .                                        |
                |---------------------------------------------------------------------------------|
                | IP N |                      |                      |     |                      |
                +---------------------------------------------------------------------------------+

            """
            realTestY = getTTvalue(allTestingDataset, allTrainingDataset)
            print("2.2 Transform testing data to Y finish.")


            """

                domainFlag: (type: pandas dataFrame)
                +----------------------------+
                |          | is core domain? |
                |----------------------------|
                | damain 1 |                 |
                |----------------------------|
                | damain 2 |                 |
                |----------------------------|
                |             .              |
                |             .              |
                |             .              |
                |----------------------------|
                | damain N |                 |
                +----------------------------+

            """
            domainFlag = allDataset.drop(['IP', 'date'])
            domainFlag = domainFlag.drop_duplicates(subset=['Domain'], keep='first')
            domainFlag = domainFlag.reset_index(drop=True)
            domainFlag = domainFlag.set_index('Domain')


            """
                domainWeight: (type: numpy array)
                +-------------------------------------------------------+
                |                | domain 1 | domain 2 | ... | domain N |
                |-------------------------------------------------------|
                | is core domain |          |          | ... |          |
                +-------------------------------------------------------+

            """
            domainWeight = np.array([0] * len(realTrainX['count'].columns))
            i = 0
            for domainID in realTrainX['count'].columns:
                flag = domainFlag.loc[domainID]['flag']
                if(flag):
                    weight = 1
                else:
                    weight = 1
                domainWeight[i] =  weight
                i = i + 1
            print("3. Calculate domains' weight finish.")


            ''' for toy test '''
            # trainingX = realTrainX[:5].copy()
            # testingY = realTestY[:5].copy()


            ''' calculating accuracy '''
            correct = 0
            nowIndex = 0

            for testDataIP in realTestY.index:
                # print nowIndex, '->', len(realTestY.index)
                nowIndex = nowIndex + 1
                b = realTestY.loc[testDataIP].values
                maxSum = -1
                maxSumIP = -1
                tempSum = 0
                for trainDataIP in realTrainX.index:
                    a = realTrainX.loc[trainDataIP].values
                    ''' do vector inner product '''
                    tempSum = (a * b * domainWeight).sum()
                    if(tempSum > maxSum):
                        maxSum = tempSum
                        maxSumIP = trainDataIP
                if(testDataIP == maxSumIP):
                    correct = correct + 1

            print("4. Calculate predicting accuracy finish.")
            print "accuracy:", correct / float(len(realTestY.index))
            correctMean += correct / float(len(realTestY.index))
            print(correctMean / float(allFiles)), "\n"

    print("5. all file finish, mean accuracy:")
    print(correctMean / float(allFiles))
    print(dataset1Len / float(allFiles))
    print(dataset2Len / float(allFiles))
