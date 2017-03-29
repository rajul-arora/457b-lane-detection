import java.util.*;
import java.util.concurrent.Callable;

//import static org.junit.Assert.*;

class Filter {

    private Callable<Object> activation;
    private final Matrix<WeightedNeuron> neurons;
    private Object previousInput;
    private Object prevResult;

    public Filter() {
        this(null, 9);
    }

    public Filter(Callable<Object> activation, int numNeurons) {
        this.activation = activation;
        int width = (int)Math.sqrt(numNeurons);
        this.neurons = new Matrix(width, width);
        this.previousInput = null;
        this.prevResult = null;

        for (int i = 0; i < width; i++) {
            for (int j = 0; j < width; j++) {
                this.neurons.set(i, j, new WeightedNeuron(constants.multiply, activation, weightsDim =[1, 1]));
            }
        }
    }

    public void adjustNeuronWeights(List<Double> deltas) {

//        assertEqual(deltas.size(), this.neurons.width() * this.neurons.height());

        for (int y = 0; y < this.neurons.height(); y++) {
            for (int x = 0; x < this.neurons.width(); x++) {
                int index = x + (y * this.neurons.width());
                this.neurons.get(y, x).adjustWeights(deltas.get(index));
            }
        }
    }

    public Object process(Object input) {
        this.previousInput = input;
        Object result = Filter.convolve(input, this.neurons);
        if (this.activation != null) {
            result = this.activation.call(result);
        }

        this.prevResult = result;
        return result;
    }

    /**
     * Convolves the input matrix with the given feature matrix.
     * Returns the convolution as an output matrix (smaller in size from input)
     */
    private static Object convolve(Matrix<Integer> input, Matrix<WeightedNeuron> feature) {

        int outputWidth = input.width() - feature.width() + 1;
        int outputHeight = input.height() - feature.height() + 1;

        Matrix<Integer> output = new Matrix(outputWidth, outputHeight);
        int[] featureDim = feature.size();
        int denom = featureDim[0] * featureDim[1];

        for (int i = 0; i < outputHeight; i++) {
            for (int j = 0; j < outputWidth; j++) {
                int sum = 0;

                for (int x = 0; x < feature.height(); x++) {
                    for (int y = 0; y < feature.width() ; y++) {
                        sum += Matrix.multiply(input.get(x + i, y + j), feature.get(x, y))[0][0];
                    }
                }

                output.set(i, j , sum / denom);
            }
        }

        return output;
    }
}