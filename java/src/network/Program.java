package network;

import network.mlp.Activations;
import network.mlp.Network;
import network.mlp.NeuralLayer;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by fikayo on 2017-03-29.
 */
public class Program {

    public static void main(String[] args) {

        List<NeuralLayer> layers = new ArrayList<>();
        layers.add(new NeuralLayer(3, Activations.Activator.ReLU));
        layers.add(new NeuralLayer(3, Activations.Activator.ReLU));
        layers.add(new NeuralLayer(3, Activations.Activator.ReLU));
        Network network = new Network(layers);
    }
}
