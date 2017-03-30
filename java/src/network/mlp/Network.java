package network.mlp;

import java.util.ArrayList;
import java.util.HashMap;
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
        boolean running = true;

        while(running) {
            for (NeuralLayer layer : this.layers) {
                data = layer.process(data);
            }

            if(running) {
                NeuralLayer lastLayer =this.layers.get(this.layers.size() - 1);

//                List<ArrayList<Double>> deltaWeightProducts = new ArrayList<>();
//                for(Neuron neuron : lastLayer.getNeurons()) {
//                    neuron.adjustWeights(diff);
//                    deltaWeightProducts.add(neuron.getDeltaWeightProducts());
//                }

                for(int i = layers.size() - 2; i >= 0; i--) {
                    NeuralLayer layer = this.layers.get(i);
                    NeuralLayer prevLayer = this.layers.get(i - 1);

                    HashMap<Integer, List<ArrayList<Double>>> deltaWeightMap = new HashMap<>();
                    for(int k = 0; k < layer.getNeuronCount(); k++) {
                        Neuron neuron = layer.getNeurons().get(k);

                        // Adjust weights for this neuron
                        List<ArrayList<Double>> deltaWeightProducts = deltaWeightMap.get(k);
                        neuron.adjustWeights(deltaWeightProducts);
                    }

                    // Set up deltaWeightProducts for the next layer
                    deltaWeightMap.clear();
                    for(int k = 0; k < layer.getNeuronCount(); k++) {
                        Neuron neuron = layer.getNeurons().get(k);

                        // Initialise entry in the map
                        if(deltaWeightMap.get(k) == null) {
                            List<ArrayList<Double>> list = new ArrayList<>();
                            for(int x = 0; x < prevLayer.getNeuronCount(); x++) {
                                list.add(null);
                            }

                            deltaWeightMap.put(k, list);
                        }

                        // Get deltaWeights from this neuron
                        List<ArrayList<Double>> deltaWeightProducts = neuron.getDeltaWeightProducts();

                        // Distribute deltaWeights accross neurons in the next layer
                        for( int key: deltaWeightMap.keySet()) {


                        }
                    }
                }
            }

            running = false;
        }


    }
}
