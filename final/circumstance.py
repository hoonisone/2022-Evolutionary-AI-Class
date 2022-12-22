import numpy as np
from numpy import dot
from numpy.linalg import norm

class  Circumstance:
    def __init__(self) -> None:
        with open("datapoint3d.txt", "r") as f:
            f.readline()
            lines = f.readlines()

        self.input_list = []
        self.output_list = []
        for line in lines:
            line = line.strip()
            line = line.split(',')
            self.input_list.append([float(line[0]), float(line[0])])
            self.output_list.append(float(line[2]))

    def mean_squared_error(self, x, y):
        return ((x-y)**2).mean(axis=None)

    def cos_similarity(self, x, y):
        return dot(x, y)/(norm(x)*norm(y))

    def get_predict_list(self, function):
        return [function(input) for input in self.input_list]

    def evaluate(self, function):
        predict_list = []
        for (x, y) in self.input_list: 
            predict_list.append(function(x, y))
        return self.mean_squared_error(np.array(self.output_list), np.array(predict_list))

    def accuracy(self, function):
        return self.cos_similarity(self.get_predict_list(function), self.output_list)

    def loss(self, function):
        return self.cos_similarity(self.get_predict_list(function), self.output_list)