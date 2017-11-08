from settings import *
import random

def getX():
    return [[2*random.random()-1 for _ in range(xs_width)] for __ in range(training_set_size)]

def getY():
    return [[random.randint(0, 1)] for _ in range(training_set_size)]