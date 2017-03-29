package network.mlp;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by fikayo on 2017-03-29.
 */
public final class VectorOperations {

    public static ArrayList<Double> sum(List<ArrayList<Double>> vector) {

        ArrayList<Double> sum = new ArrayList<>();
        sum.addAll(vector.get(0));

        for(int i = 1; i < vector.size(); i++) {
            ArrayList<Double> value = vector.get(i);

            for(int j = 0; j < value.size(); j++) {
                sum.set(j, sum.get(j) + value.get(j));
            }
        }

        return sum;
    }

    public static ArrayList<Double> scalarProduct(double scalar, ArrayList<Double> vector) {

        ArrayList<Double> output = new ArrayList<>();

        for(int i = 0; i < vector.size(); i++) {
            double a = vector.get(i);

            output.add(scalar * a);
        }

        return output;
    }

    public static ArrayList<Double> add(ArrayList<Double> vector1, ArrayList<Double> vector2) {

        ArrayList<Double> output = new ArrayList<>();

        for(int i = 0; i < vector1.size(); i++) {
            double a = vector1.get(i);
            double b = vector2.get(i);

            output.add(a + b);
        }

        return output;
    }

    public static ArrayList<Double> product(ArrayList<Double> vector1, ArrayList<Double> vector2) {

        ArrayList<Double> output = new ArrayList<>();

        for(int i = 0; i < vector1.size(); i++) {
            double a = vector1.get(i);
            double b = vector2.get(i);

            output.add(a * b);
        }

        return output;
    }
}
