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
        NeuralLayer layer3 = new NeuralLayer(Activations.Activator.ReLU, 3, 0);
        NeuralLayer layer2 = new NeuralLayer(Activations.Activator.ReLU, 3, layer3.getNeuronCount());
        NeuralLayer layer1 = new NeuralLayer(Activations.Activator.ReLU, 3, layer2.getNeuronCount());

        layers.add(layer1);
        layers.add(layer2);
        layers.add(layer3);
        Network network = new Network(layers);
    }
}
