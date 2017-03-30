package network.mlp;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by fikayo on 2017-03-29.
 */
public class NeuralLayer {

    private final List<Neuron> neurons;

    public NeuralLayer(Activations.Activator activator, int numNeurons, int numConnectedNeurons) {

        this.neurons = new ArrayList<>(numNeurons);

        for (int i = 0; i < numNeurons; i++) {
            this.neurons.add(new Neuron(activator, numConnectedNeurons));
        }
    }

    public int getNeuronCount() {
        return this.neurons.size();
    }

    public List<Neuron> getNeurons() {
        return this.neurons;
    }

    /**
     * Pass all inputs to every neuron in the layer
     *
     * @param inputs
     * @return
     */
    public List<ArrayList<Double>> process(final List<ArrayList<Double>> inputs) {

        List<ArrayList<Double>> output = new ArrayList<>();

        for (Neuron neuron : this.neurons) {
            ArrayList<Double> outputVector = neuron.process(inputs);
            output.add(outputVector);
        }

        return output;
    }
}
