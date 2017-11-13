from settings_cnn import *
import random
import csv
import numpy as np

def getTrialDataX():
    return \
    [
        [
            [
                    [random.random()] for _ in range(embedding_size)
            ]
                for __ in range(sequence_length)
        ]
        for ___ in range(training_set_size)
    ]

def getTrialDataY():
    listY = []
    for _ in range(training_set_size):
        list = [0 for _ in range(category_count)]
        list[random.randint(0, len(list)-1)] = 1
        listY.append(list)
    return listY

def getDataXY(begin=0, end=training_set_size):
    data_table = list(csv.reader(open(training_source, "r")))
    listX = []
    listY = []
    i = 0
    print("start loading data")
    for line in data_table[begin:end]:
        i += 1
        if i%50==0 or i >= 17700:
            print(i)
        if len(line)-3 > sequence_length:
            # too long
            continue
        vec_emotion = eval(line[-1])
        vec_object = eval(line[-2])
        y = [0 for _ in range(category_count)]
        for emotion, object in list(zip(vec_emotion, vec_object)):
            y[sequence_length*(emotion-1) + object - 1] = 1
        listY.append(y)

        x = []
        for x_vec in line[1:-2]:
            x.append(list(np.array(eval(x_vec)).reshape((-1, 1))))
        listX.append(x)
        # padding
        while len(x) < sequence_length:
            x.append([[0] for _ in range(embedding_size)])

    print("load data complete")
    return (listX, listY)


def getTestData(begin=0, end=test_set_size):
    data_table = list(csv.reader(open(test_source, "r")))
    listX = []
    jumped = []
    i = 0
    print("start loading test data")
    for line in data_table[begin:end]:
        i += 1
        if len(line) > sequence_length:
            jumped.append(i)
            print("jumped " + str(i))
            continue
        x = []
        for x_vec in line:
            x.append(list(np.array(eval(x_vec)).reshape((-1, 1))))
        listX.append(x)
        # padding
        while len(x) < sequence_length:
            x.append([[0] for _ in range(embedding_size)])

    return (listX, jumped)


class Data():
    def __init__(self, listX, listY, batch_size):
        self.listX = listX
        self.listY = listY
        self.batch_size = batch_size
        self.current_base = 0

    def getNextXY(self):
        if self.current_base + self.batch_size >= len(self.listX):
            ret = (self.listX[self.current_base:], self.listY[self.current_base:])
            self.current_base = 0
        else:
            ret = (self.listX[self.current_base:self.current_base+self.batch_size],
                   self.listY[self.current_base:self.current_base+self.batch_size])
            self.current_base += self.batch_size
        return ret