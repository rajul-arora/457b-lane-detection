
public final class Utils {

    public static final int NEURON_COUNT = 3;
    public static final double EPSILON = 0.1;
    public static final double LEARNING_RATE = 0.1;
    public static final boolean GREYSCALE = true;
    public static final int[] PARTIAL_IN_IMG_DIM = {32, 32};
    public static final String OUTPUT_DIRECTORY = "./lane_images/cordova1_output_matrices/";
    public static final int IMAGE_WIDTH = 640;
    public static final int IMAGE_HEIGHT = 480;

    /**
     * Performs sigmoid on all elements in input and returns a matrix of the same size
     */
    public static Matrix<Double> sigmoid(Matrix<Double> input) {

        // Instantiate output as a matrix same dimensions as input
        Matrix<Double> output = new Matrix(input.size());

        // Perform sigmoid on all elements in input matrix
        for (int x = 0; x < input.height(); x++) {
            for (int y = 0; y < input.width(); y++) {
                output.set(x, y, 1 / (1 + Math.exp(-1 * input.get(x, y))));
            }
        }

        return output;
    }

    public static double sigderiv(double input) {
        return Math.pow((-1 * Math.exp(-1 * input)) / (1 + Math.exp(-1 * input)), 2);
    }

    /**
     * Sets all negative numbers to 0. Returns a matrix of the same size
     */
    public static Matrix<Double> ReLU(Matrix<Double> input) {
        Matrix<Double> output = new Matrix(input.size());

        // Perform sigmoid on all elements in input matrix
        for (int x = 0; x < input.height(); x++) {
            for (int y = 0; y < input.width(); y++) {
                output.set(x, y, input.get(x, y) > 0 ? input.get(x, y) : 0.0);
            }
        }

        return output;
    }

    /**
     * Calculates the derivative of ReLU of the given input matrix X.
     */
    public static Matrix<Double> dReLU(Matrix<Double> X) {
        Matrix<Double> output = new Matrix(X.size());

        for (int i = 0; i < X.height(); i++) {
            for (int j = 0; j < X.width(); j++) {
                output.set(i, j, X.get(i, j) > 0 ? 1.0 : 0.0);
            }
        }

        return output;
    }

    public static double multiply(double x, double y) {
        return x * y;
    }

    public static Matrix<Double> multiply(double input, Matrix<Double> matrix) {

        Matrix<Double> multWeights = new Matrix(matrix.size());
        for (int x = 0; x < matrix.height(); x++) {
            for (int y = 0; y < matrix.width(); y++) {
                multWeights.set(x, y, matrix.get(x, y) * input);
            }
        }

        return multWeights;
    }
}