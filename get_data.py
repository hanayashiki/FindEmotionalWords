import csv
from settings import  *

data_table = list(csv.reader(open("food.csv", "r")))
DISTANCE_IDX = 3
OBJECT_IDX = 4
EMOTION_IDX = 5
ZERO_ONE_IDX = 6

batch_size = training_set_size

def getX():
    X_table = list()
    i = 0
    for line in data_table[0: batch_size]:
        i += 1
        vec = [eval(line[DISTANCE_IDX]) for _ in range(100)] + eval(line[OBJECT_IDX]) + eval(line[EMOTION_IDX])
        if i % 100 == 0:
            print(vec)
        X_table.append(vec)

    print("X fetched")
    return X_table

def getY():
    Y_table = list()
    i = 0
    for line in data_table[0: batch_size]:
        i += 1
        vec = [eval(line[ZERO_ONE_IDX])]
        if i % 100 == 0:
            print(vec)
        Y_table.append(vec)

    print("Y fetched")
    return Y_table

def getTestX():
    X_table = list()
    i = 0
    for line in data_table[batch_size: batch_size+test_set_size]:
        i += 1
        vec = [eval(line[DISTANCE_IDX]) for _ in range(100)] + eval(line[OBJECT_IDX]) + eval(line[EMOTION_IDX])
        if i % 100 == 0:
            print(vec)
        X_table.append(vec)

    print("TestX fetched")
    return X_table

def getTestY():
    Y_table = list()
    i = 0
    for line in data_table[batch_size: batch_size+test_set_size]:
        i += 1
        vec = [eval(line[ZERO_ONE_IDX])]
        if i % 100 == 0:
            print(vec)
        Y_table.append(vec)

    print("TestY fetched")
    return Y_table
