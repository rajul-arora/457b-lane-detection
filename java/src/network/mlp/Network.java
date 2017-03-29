package network.mlp;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by fikayo on 2017-03-29.
 */
public class Network {

    private final List<NeuralLayer> layers;

    public Network(List<NeuralLayer> layers) {
        this.layers = layers;
    }

    public void train(List<ArrayList<Double>> input) {

        List<ArrayList<Double>> data = input;
        for(NeuralLayer layer : this.layers) {
            data = layer.process(data);
        }

    }
}
