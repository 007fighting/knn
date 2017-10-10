# -*- coding: utf-8 -*-

from com.fighting.util.DataUtil import *
from com.fighting.knn.kNN import *
import matplotlib.pyplot as plt

def knn():
    '''test knn'''
    row, column, classes, k = (100, 5, 3, 10)

    # load data set
    dataUtil = DataUtil()
    dataSet, dataLabel = dataUtil.randomDataSet(row, column, classes)
    print 'dataSet: '
    print dataSet
    print 'dataLabel: '
    print dataLabel

    # normalize
    dataSet = dataUtil.norm(dataSet)
    print 'norm-dataSet:'
    print dataSet

    # plot the data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataSet[:,0], dataSet[:,1], 15*array(dataLabel), 15*array(dataLabel))
    plt.show()

    # random vector X
    vectorX = dataUtil.randomX(dataSet.shape[1])
    print 'vectorX: '
    print vectorX

    # classify
    knn = kNN()
    distance, clz = knn.classify(dataSet, dataLabel, vectorX, k)
    print 'distance: '
    print distance
    print 'clz=%d' % clz

def dating():
    '''test dating classify'''
    # load data set
    dataUtil = DataUtil()
    dataSet, dataLabel = dataUtil.file2DataSet('../../../datasets/knn/datingTestSet.txt')
    dataSet = dataUtil.norm(dataSet)

    # split training set and testing set
    ratio = 0.8
    trainingSet, trainingLabel, testingSet, testingLabel = dataUtil.spitData(dataSet, dataLabel, ratio)
    testingSize = testingSet.shape[0]

    # training and testing
    knn = kNN()
    for k in range(1, 11):
        error = 0
        for i in range(testingSize):
            distance, clz = knn.classify(trainingSet, trainingLabel, testingSet[i,], k)
            if clz != testingLabel[i]:
                error += 1
        print '%d, %.2f' % (k, error*1.0/testingSize)

def f2d():
    '''test file2dataset'''
    dataUtil = DataUtil()
    dataSet, dataLabel = dataUtil.file2DataSet('../../../datasets/knn/datingTestSet.txt')
    print 'dataSet:'
    print dataSet
    print 'dataLabel:'
    print dataLabel

if __name__ == '__main__':
    knn()
    #dating()
    #f2d()