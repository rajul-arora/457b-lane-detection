import java.util.ArrayList;
import java.util.List;

class InputLayer extends  NeuralLayer {

    public List<Object> process(Object input) {

        List<Object> outputs = new ArrayList<>();
        for (int i = 0; i < Utils.NEURON_COUNT; i++) {
            outputs.add(input);

            return outputs;
        }
    }
}