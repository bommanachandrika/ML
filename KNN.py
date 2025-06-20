import csv
import random
import math
import operator
from collections import Counter

def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(len(dataset[x]) - 1):  # Convert feature values to float
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += (instance1[x] - instance2[x]) ** 2
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1  # Exclude label
    for trainInstance in trainingSet:
        dist = euclideanDistance(testInstance, trainInstance, length)
        distances.append((trainInstance, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = [distances[x][0] for x in range(k)]
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for neighbor in neighbors:
        response = neighbor[-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0

def main():
    # prepare data
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset('knndat.data', split, trainingSet, testSet)
    print('Train set:', len(trainingSet))
    print('Test set:', len(testSet))

    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print(f'> predicted={result}, actual={testSet[x][-1]}')
    
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: %.2f%%' % accuracy)

if __name__ == '__main__':
    main()


5.1,3.5,1.4,0.2,0
4.9,3.0,1.4,0.2,0
4.7,3.2,1.3,0.2,0
6.4,3.2,4.5,1.5,1
6.9,3.1,4.9,1.5,1
5.5,2.3,4.0,1.3,1
6.5,3.0,5.2,2.0,2
6.2,3.4,5.4,2.3,2
5.9,3.0,5.1,1.8,2
5.0,3.6,1.4,0.2,0
5.4,3.9,1.7,0.4,0
5.8,2.7,4.1,1.0,1
6.0,2.2,4.0,1.0,1
6.3,2.9,5.6,1.8,2
5.6,2.8,4.9,2.0,2
