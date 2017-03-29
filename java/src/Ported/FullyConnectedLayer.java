package Ported;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

class FullyConnectedLayer extends NeuralLayer {

    public FullyConnectedLayer() {
        this(null);
    }

    public FullyConnectedLayer(Callable<Object> activation) {
        this(activation, Utils.NEURON_COUNT);
    }

    public FullyConnectedLayer(Callable<Object> activation, int neuronCount) {
        this(activation, neuronCount, 1);
    }

    public FullyConnectedLayer(Callable<Object> activation, int neuronCount, int numForwardNeurons) {
        super(this.combine, activation,neuronCount, numForwardNeurons);
    }

    public WeightedNeuron createNueron(Callable<Object> func, Callable<Object> activation, int numForwardNeurons) {
        return new WeightedNeuron(func, activation, new int {32,32}, numForwardNeurons);
    }

    /**
     * Passes the inputs to all neurons.
     */
    public List<Object> process(List<List<Object>> inputs) {

        List<Object> outputs = new ArrayList<>();
        for (int i = 0; i < this.neurons.size(); i++) {
            Neuron neuron = this.neurons[i];

            // Process each neuron with all the inputs
            List<Object> neuronInputs = new ArrayList<>();
            for (List<Object> input : inputs) {
                neuronInputs.append(input.get(i));
            }

            result = neuron.process(neuronInputs);

            // Take the dot product of result with weights to get output
            for (int j = 0; j < this.numForwardNeurons; j++) {
                neuron.prevResults[i] = this.dotProduct(result, neuron.weights[j]);
            }

            outputs.add(neuron.prevResults);
        }

        return outputs;
    }

    /**
     * Calculates the delta for the given output and expected output.
     */
    public void calculateDeltas(List<Double> prevDeltas) {
//        assert prevDeltas.size() == this.neurons.size()

        List<Double> deltas = new ArrayList<>();

        for (Neuron neuron : this.neurons) {
            int sum = 0;
            for (int i = 0; i < deltas.size(); i++) {
                sum += Utils.multiply(deltas[i], neuron.weights[i]);
            }

            delta = constants.sigderiv(neuron.prevResult) * sum;
            deltas.append(delta);
        }


        return deltas;
    }

    /**
     * Takes a list of matrices and smushes their entries together into a single matrix
     * Each row is each input as a a 1d array
     */
    public static <T> Matrix<T> combine(List<T> inputs, empty) {

        output =[];
        for (T input : inputs) {
            vector =[]
            for (int i = 0; i < input.height(); i++) {
                for (int j = 0; j < input.width(); j++) {
                    vector.append(input[i][j]);
                }
            }

            output.append(vector);
        }

        return Matrix.convert(output);
    }

    /**
     * Performs dot-product of input with weights
     */
    public static double dotProduct(Matrix<Integer> input, Matrix<Integer> weights) {

        double sum = 0;
        int[] flat1 = new int[input.width() * input.height()];
        input.flatten(flat1);
        int[] flat2 = new int[weights.width() * weights.height()];
        weights.flatten(flat2);

        for (int x = 0; x < Math.min(flat1.length, flat2.length); x++) {
            sum += flat1[x] * flat2[x];
        }

        return sum;
    }
}