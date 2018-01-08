# User-fingerprinting-and-profiling
### Description
When user visits websites, it must left a record. This project's final goal is to predict users who visits the domains.

For more details, we reference the paper from [ACM SIGCOMM 2017](https://conferences.sigcomm.org/sigcomm/2017/program.html), Workshop 3 [Big-DAMA](https://conferences.sigcomm.org/sigcomm/2017/workshop-big-dama.html), [Users' Fingerprinting Techniques from TCP Traffic](http://delivery.acm.org/10.1145/3100000/3098602/p49-Vassio.pdf?ip=140.113.216.230&id=3098602&acc=OPENTOC&key=AF37130DAFA4998B%2E7DDA227B4DBFAC43%2E4D4702B0C3E38B35%2E9F04A3A78F7D3B8D&CFID=848763280&CFTOKEN=58230019&__acm__=1515312742_9122381161af0c1b04fc1585ff010855). 

### Dataset
The dataset is from the paper supports. [link](https://bigdata.polito.it/content/domains-web-users)

And I transform to my format.


### Environment
* I7 6700
* Ubuntu 16.04
* Python 2.7
* Sklearn 
* Pandas
* Numpy


### Methods
I use some of the predicting models from [sklearn](http://scikit-learn.org/) and cosine similarity to do the predicting. Simply, we want to use domain to predict which user visited by ip address.

##### 1. Cosine similarity
```sh
$ python cosine-similarity.py
```
Step 1. First of all, read the training and testing data file.
<br/>

Step 2. Set the data format to pandas like this:
```
    allDataset:
    +-------------------+
    |	training data 	|
    +-------------------+
    |	testing  data 	|
    +-------------------+
```
<br/>

Step 3. And transform it to training data and testing data like this.
```
	training / testing data:
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

```
<br/>

Step 4. Get the domain's flag, which is core domain or support domain.
```
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
```
<br/>

Step 5. Transform it to numpy array type. Because numpy array doing mutiply is super quick.
```
	domainWeight: (type: numpy array)
    +-------------------------------------------------------+
    |                | domain 1 | domain 2 | ... | domain N |
    |-------------------------------------------------------|
    | is core domain |          |          | ... |          |
    +-------------------------------------------------------+
```
<br/>

Step 6. Calculating accuracy by doing cosine similarity. Choose the biggest similarity value as the predicting answer, and then check it.
<br/>

#### 2. Predicting Models
###### 2.1 K Nearest Neighbor
###### 2.2 Navie Bayes
###### 2.3 Random forest
###### 2.4 MLPClassifier (Neural Network)
```sh
$ python modelPredict.py --model=[model type] --time=[time series type]
```
```sh
modelTypes = ['knn', 'navieBayes', 'randomForest', 'neuralNetwork']
timeTypes = [1, 6, 24]
```

Step 1. read data from csv file. The data format is:
```
	dataset:
    +------------------------------------------------------+
    | Index | IP address | Domain | flag | date | IP label |
    |------------------------------------------------------|
    |      |                      |                        |
    +------------------------------------------------------+
```
<br/>

Step 2. Set training / testing data X and Y.
```
    X:
    +----------------+
    | Index | Domain |
    |----------------|
    |       |        |
    +----------------+

    Y:
    +------------------+
    | Index | IP label |
    |------------------|
    |       |          |
    +------------------+

```
<br/>

Step 3. Train training data to the models.
```python
models.fit(trainingX, trainingY)
```
<br/>

Step 4. Predict testing data to the models.
```python
testingPredicingY = models.predict(testingX)
```
<br/>

Step 5. Calculating accuracy by module ```metrics```.
```python
accuracy = metrics.accuracy_score(testingY, testingPredicingY)
```
