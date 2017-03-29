package network.mlp;

import java.util.*;
import java.util.concurrent.Callable;

/**
 * Created by fikayo on 2017-03-29.
 */
public class Neuron {

    /**
     * Each weight is a ArrayList of Doubles which is the same size as the input ArrayLists
     */
    private final List<ArrayList<Double>> weights;

    private final Activations.Activator activator;

    private ArrayList<Double> currentValue;
    private ArrayList<Double> currentDelta;
    private List<ArrayList<Double>> prevInput;

    public Neuron(Activations.Activator activator) {
        this.weights = new ArrayList<>();
        this.activator = activator;
        this.currentValue = null;
    }

    public List<ArrayList<Double>> getDeltaWeightProducts() {

        List<ArrayList<Double>> deltaWeightProducts = new ArrayList<>();
        for (ArrayList<Double> weightVector : this.weights) {

            deltaWeightProducts.add(VectorOperations.product(this.currentDelta, weightVector));
        }

        return deltaWeightProducts;
    }

    public void adjustWeights(List<ArrayList<Double>> deltaWeightProducts) {

        this.currentDelta = this.calculateDelta(deltaWeightProducts);
        for (int i = 0; i < this.prevInput.size(); i++) {
            ArrayList<Double> deltaWeights = VectorOperations.scalarProduct(Utils.LEARNING_RATE, VectorOperations.product(this.currentDelta, this.prevInput.get(i)));
            this.weights.set(i, VectorOperations.add(this.weights.get(i), deltaWeights));
        }

    }

    private ArrayList<Double> calculateDelta(List<ArrayList<Double>> deltaWeightProducts) {

        // Sum up all the delta-weight products
        ArrayList<Double> sum = VectorOperations.sum(deltaWeightProducts);

        // Multiply sum with act'(currentValue)
        ArrayList<Double> deriv = Activations.deriv(this.activator, this.currentValue);
        return VectorOperations.product(deriv, sum);
    }

    /**
     * Performs a dot-product of the set of inputs with the set of weights
     *
     * @param inputs - A set of inputs (each input is a vector)
     * @return A single output (which is a vector)
     */
    public ArrayList<Double> process(final List<ArrayList<Double>> inputs) {

        this.prevInput = inputs;

        List<ArrayList<Double>> products = new ArrayList<>(inputs.size());

        // First multiply each input with the corresponding weight
        for (int i = 0; i < inputs.size(); i++) {
            ArrayList<Double> input = inputs.get(i);
            ArrayList<Double> weight = this.weights.get(i);

            // Multiply input with weight
            ArrayList<Double> product = new ArrayList<>();
            for (int j = 0; j < input.size(); j++) {
                product.add(input.get(j) * weight.get(j));
            }

            products.add(product);
        }

        // Now sum up the products
        ArrayList<Double> output = VectorOperations.sum(products);

        this.currentValue = Activations.call(this.activator, output);
        return this.currentValue;
    }
}
