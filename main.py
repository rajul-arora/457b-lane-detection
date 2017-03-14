import math
from network.Neuron import Neuron
from network.NeuralLayer import NeuralLayer
from network.Network import Network

EPSILON = 0.0001

def convolution(inputs, feature):
    assert len(inputs) == len(feature)
    sum = 0
    numPixels = len(inputs)
    for i in range(numPixels):
        sum += inputs[i] * feature[i]

    return sum / numPixels

def sigmoid(input):
    return 1 / (1 + math.exp(-1 * input))

def pool():
    pass

def main():

    inputs = [1]
    feature = [1]

    conv = NeuralLayer(16, convolution(inputs, feature))
    activ = NeuralLayer(16, sigmoid(1))
    pooll = NeuralLayer(16, pool())

    layers = []
    layers.append(conv)
    layers.append(activ)
    layers.append(pool)

    network = Network(16)

if __name__ == '__main__':
    main()