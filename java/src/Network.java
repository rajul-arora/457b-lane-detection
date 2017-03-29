import java.util.ArrayList;
import java.util.List;

class Network {

    private final List<NeuralLayer> layers;
    private final InputLayer inputLayer;
    private final FullyConnectedLayer midFCL;
    private final FullyConnectedLayer finalFCL;

    public Network(List<NeuralLayer> layers) {
        this.layers = layers;
        this.inputLayer = new InputLayer(null, 1);
        this.midFCL = new FullyConnectedLayer(Utils.sigmoid, 9);
        this.finalFCL = new FullyConnectedLayer(Utils.sigmoid, 2);
    }
    // this.outputLayer = OutputLayer([4, 3])

    private boolean run(Matrix<Integer> input) {

        List<Object> data = this.inputLayer.process(input);
        for (NeuralLayer layer :this.layers){
            data = layer.process(data);
        }

        data = this.midFCL.process(data);
        List<Double> outputs = this.finalFCL.process(data);
//        List<Double> outputs = this.outputLayer.process(data);

        // Return your decision based on the vote
        return outputs.get(0) > outputs.get(1);
    }

    private void train(Matrix<Integer> input, List<Double> expectedOutput) {

        double error = 0;
        boolean running = true;

        while running {
            data = this.inputLayer.process(input)

            for (NeuralLayer layer : this.layers) {
                data = layer.process(data);
            }

            List<Object> midOutput = this.midFCL.process(data);

            ArrayList<List<Object>> innerlist = new ArrayList<>();
            innerlist.append(midOUtput);
            ArrayList<Matrix<Object>> list = new ArrayList<>();
            list.append(Matrix < Object >.convert(innerlist));

            List<Object> output = this.finalFCL.process(list);
            // outputs = this.outputLayer.process(data);
            double error = this.calculateLoss(output, expectedOutput);
            running = error > Utils.EPSILON;

//            assert (len(output) == len(expectedOutput))
            if (running) {
                List<Double> initialDeltas = new ArrayList<>(output.size());
                for (int i = 0; i < len(output); i++) {
                    initialDeltas.add(expectedOutput[i] - output[i]);
                }

                List<Double> deltas = this.finalFCL.calculateDeltas(initialDeltas);
                deltas = this.midFCL.calculateDeltas(deltas);
                for (int i = this.layers.size() - 1; i >= 0; i++) {
                    NeuralLayer layer = this.layers.get(i);
                    layer.adjustWeights(deltas);
                    deltas = layer.calculateDeltas(deltas);
                }
            }

            System.out.println("Output dim: " + str(data[0].size()) + " Error " + str(error));
        }

        System.out.println("Final Outputs: " + str(output));
        System.out.println("\n\nHooray!!! we're done! Final Error: " + str(error));
    }

    private double calculateLoss(this,List<Double> output, List<Double> expectedOutput) {
        return this.mse(output, expectedOutput);
    }

    /**
     * Mean Squared Error
     */
    private double mse(List<Double> output, List<Double> expectedOutput) {
//        assert len(output) == len(expectedOutput)

        double result = 0;
        for (int i = 0; i < output.size(); i++) {
            result += Math.pow(expectedOutput.get(i) - output.get(i), 2);
        }

        return result / 2;
    }

    private double sum(List<Double> output, List<Double> expectedOutput) {
//        assert len(output) == len(expectedOutput)

        double result = 0;
        for (int i = 0; i < output.size(); i++) {
            result += Math.abs(expectedOutput.get(i) - output.get(i));
        }

        return result;
    }
}