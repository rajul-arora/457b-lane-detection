package Ported;

import java.util.*;

class WeightedNeuron extends Neuron {

    private final List<Matrix<Double>> weights;

    public WeightedNeuron(Callabe<Object> func) {
        this(func, null);
    }

    public WeightedNeuron(Callabe func, Callabe<Object> activation) {
        this(func, activation, new int[]{16, 16});
    }

    public WeightedNeuron(Callabe<Object> func, Callabe<Object> activation, int[] weightsDim) {
        this(func, activation, weightsDim, 1);
    }

    public WeightedNeuron(Callabe<Object> func, Callabe<Object> activation, int[] weightsDim, int numForwardNeurons) {
        super(func, activation, numForwardNeurons = numForwardNeurons);
        this.weights = new ArrayList<Matrix<Double>>(numForwardNeurons);

        for (int i = 0; i < numForwardNeurons; i++) {
            this.weights.add(this.generateWeights(weightsDim));
        }
    }

    public Object callFunc(Object input) {
        return this.func.call(input, this.weights);
    }


    public Matrix<Double> generateWeights(int[] weightsDim) {
        int width = weightsDim[0];
        int height = weightsDim[1];

        Matrix<Double> output = new Matrix(weightsDim);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                output.set(i, j, Math.random());
            }
        }

        return output;
    }

    public Matrix<Double> multiply(double input) {
        Matrix<Double> multWeights = new Matrix(this.weights.size());
        for (int x = 0; x < this.weights.height(); x++) {
            for (int y = 0; y < this.weights.width(); y++) {
                multWeights.set(x, y, this.weights.get(x, y) * input);
            }
        }

        return multWeights;
    }

    /**
     * Adjusts the weights using the values of delta propagated to it.
     */
    public void adjustWeights(List<Double> deltas) {
//        assert len(deltas)==len(this.weights)

        for (int i = 0; i < this.weights.size(); i++) {
            Matrix<Double> weight = this.weights.get(i);
            for (int y = 0; y < weight.height(); y++) {
                for (int x = 0; x < weight.width(); x++) {
                    weight.set(y, x, weight.get(y, x) + Utils.LEARNING_RATE * deltas.get(i) * this.prevResult[i]);
                }
            }
        }
    }
}