import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

class NeuralLayer implements Layer {

    private final List<Neuron> neurons;
    private final int numForwardNeurons;

    public NeuralLayer(Callable<Object> func) {
        this(func, null);
    }

    public NeuralLayer(Callable<Object> func, Callable<Object> activation) {
        this(func, activation, Utils.NEURON_COUNT);
    }

    public NeuralLayer(Callable<Object> func, Callable<Object> activation, int neuronCount) {
        this(func, activation, Utils.NEURON_COUNT, 1);
    }

    public NeuralLayer(Callable<Object> func, Callable<Object> activation, int neuronCount, int numForwardNeurons) {
        this.neurons = new ArrayList<>();
        for (int i = 0; i < neuronCount; i++) {
            this.neurons.add(this.createNueron(func, activation, numForwardNeurons));
        }

        this.numForwardNeurons = numForwardNeurons;
    }

    protected Neuron createNueron(Callable<Object> func, Callable<Object> activation, int numForwardNeurons) {
        return new Neuron(func, activation, numForwardNeurons);
    }


    public List<Double> calculateDeltas(List<Double> deltas) {
        return deltas;
    }

    /**
     * Adjusts the weights using the values of delta propagated to it.
     */
    public void adjustWeights(List<Double> deltas) {

        for (Neuron neuron : this.neurons) {
            neuron.adjustWeights(deltas);
        }
    }

    /**
     * Passes the inputs to their corresponding neuron.
     * That is, input[i] -> neuron[i]
     */
    public List<Object> process(List<Object> inputs) {

//        assert len(inputs) == len(this.neurons)

        List<Object> outputs = new ArrayList<>();
        for (int i = 0; i < inputs.size(); i++) {
            Neuron neuron = this.neurons.get(i);

            // Process each neuron with the corresponding input
            Object result = neuron.process(inputs.get(i));
            outputs.add(result);
        }

        // print("Input to layer(" + str(this) + "): " + str(inputs))
        // print("Output from layer(" + str(this) + "): " + str(outputs))
        return outputs;
    }
}