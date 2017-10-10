# -*- coding: utf-8 -*-

import operator
from numpy import *

class kNN:
    def __init__(self):
        pass

    def classify(self, dataSet, dataLabel, vectorX, k):
        # data validate
        (row, column) = dataSet.shape
        if row <= 0 or column <= 0 or row != len(dataLabel) or column != len(vectorX) or k <= 0:
            return None, None

        # calculate distance and sort
        dataX = tile(vectorX, (row, 1))
        distance = (((dataX - dataSet)**2).sum(axis=1))**0.5
        sortedIndice = distance.argsort()

        # classify
        classCount = {}
        for i in range(k):
            if i >= row:
                break
            label = dataLabel[sortedIndice[i]]
            classCount[label] = classCount.get(label, 0) + 1

        # sort and return result
        return distance, sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)[0][0]