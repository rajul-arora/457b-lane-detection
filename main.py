import math

EPSILON = 0.0001

def convolution(inputs, weights):
    assert len(inputs) == len(weights)

    numPixels = len(inputs)
    for i in range(numPixels):
        sum += inputs[i] * weights[i]

    return sum / numPixels

def sigmoid(input):
    return 1 / (1 + math.exp(-1 * input))

def pool():
    pass

def main():
    
    conv = NeuralLayer(16, convolution)
    activ = NeuralLayer(16, sigmoid)
    pool = NeuralLayer(16, pool)

    layers = []
    layers.append(conv)
    layers.append(activ)
    layers.append(pool)

    network = Network(16, layers)

if __name__ == '__main__':
    main()