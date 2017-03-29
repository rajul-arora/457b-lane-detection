package network.mlp;

import java.util.ArrayList;

/**
 * Created by fikayo on 2017-03-29.
 */
public final class Activations {

    public static ArrayList<Double> call(Activator activator, ArrayList<Double> input) {

        ArrayList<Double> output = new ArrayList<Double>();
        switch (activator) {

            case SIGMOID: {
                for (double value : input) {
                    output.add(sigmoid(value));
                }

                return output;
            }

            case ReLU: {
                for (double value : input) {
                    output.add(ReLU(value));
                }

                return output;
            }

            case NONE:
                return input;

            default:
                throw new RuntimeException("Unknown Activator");
        }
    }

    public static double call(Activator activator, double input) {

        switch (activator) {

            case SIGMOID:
                return sigmoid(input);

            case ReLU:
                return ReLU(input);

            case NONE:
                return input;

            default:
                throw new RuntimeException("Unknown Activator");
        }
    }

    public static double deriv(Activator activator, double input) {

        switch (activator) {

            case SIGMOID:
                return sigderiv(input);

            case ReLU:
                return dReLU(input);

            case NONE:
                return input;

            default:
                throw new RuntimeException("Unknown Activator");
        }
    }

    public static ArrayList<Double> deriv(Activator activator, ArrayList<Double> input) {

        ArrayList<Double> output = new ArrayList<Double>();
        switch (activator) {

            case SIGMOID: {
                for (double value : input) {
                    output.add(sigderiv(value));
                }

                return output;
            }

            case ReLU: {
                for (double value : input) {
                    output.add(dReLU(value));
                }

                return output;
            }

            case NONE:
                return input;

            default:
                throw new RuntimeException("Unknown Activator");
        }
    }

    private static Double sigmoid(double input) {
        return 1 / (1 + Math.exp(-1 * input));
    }

    private static double sigderiv(double input) {
        return Math.pow((-1 * Math.exp(-1 * input)) / (1 + Math.exp(-1 * input)), 2);
    }

    /**
     * Sets all negative numbers to 0. Returns a matrix of the same size
     */
    private static double ReLU(double input) {
        return Math.max(input, 0);
    }

    /**
     * Calculates the derivative of ReLU of the given input matrix X.
     */
    private static double dReLU(double input) {
        return input > 0 ? 1 : 0;
    }

    public enum Activator {
        SIGMOID, ReLU, NONE
    }
}
