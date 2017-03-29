package Ported;

import java.util.*;
import java.util.concurrent.Callable;

/**
 * Created by fikayo on 2017-03-29.
 */
public class ConvolutionLayer implements Layer {

    private final List<Filter> filters;

    public ConvolutionLayer(Callable<Object> activation) {
        this(activation, 3);
    }

    public ConvolutionLayer(Callable<Object> activation, int numFilters) {
        this(activation, numFilters, 9);
    }

    public ConvolutionLayer(Callable<Object> activation, int numFilters, int neuronsPerFilter) {
        this.filters = new ArrayList<>(neuronsPerFilter);
        for (int i = 0; i < numFilters; i++) {
            this.filters.add(new Filter(activation, neuronsPerFilter));
        }
    }

    /**
     * Adjusts the weights using the values of delta propagated to it.
     */
    public void adjustWeights(List<Double> deltas) {

        for (Filter filter : this.filters) {
            filter.adjustNeuronWeights(deltas);
        }
    }

    /**
     * Passes the inputs to their corresponding neuron.
     * That is, input[i] -> neuron[i]
     */
    public List<Object> process(List<Object> inputs) {
//            assert len(inputs) == len(this.filters)

        List<Object> outputs = new ArrayList<>();
        for (int i = 0; i < inputs.size(); i++) {

            Filter filter = this.filters.get(i);

            // Process each neuron with the corresponding input
            Object result = filter.process(inputs.get(i));
            outputs.add(result);
        }

        return outputs;
    }
}