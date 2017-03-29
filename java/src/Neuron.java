import java.util.*;
import java.util.concurrent.Callable;

class Neuron {

    protected Callable<Object> func;
    protected Callable<Object> activation;
    protected Object previousInput;
    protected int numForwardNeurons;
    protected final Object[] prevResult;
    protected Object currentValue;

    public Neuron(Callable<Object> func) {
        this(func, null);
    }

    public Neuron(Callable<Object> func, Callable<Object> activation) {
        this(func, activation, 1);
    }

    public Neuron(Callable<Object> func, Callable<Object> activation, int numForwardNeurons) {
        this.func = func;
        this.activation = activation;
        this.previousInput = null;
        this.numForwardNeurons = numForwardNeurons
        this.prevResult = new Object[numForwardNeurons];
        this.currentValue = null;
    }

    public Object process(Object input) {
        this.previousInput = input;

        Object value = this.callFunc(input);
        if (this.activation != null) {
            value = this.activation.call()(value);
        }

        this.currentValue = value;
        return value;
    }

    protected void adjustWeights(List<Double> deltas) {

    }

    protected Object callFunc(Object input) {
        return this.func.call(input);
    }
}