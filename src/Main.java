/**
 * Created by khudiakov on 08.12.2016.
 */
import MLP.MLP;

import java.util.ArrayList;
import java.util.List;

class TrainingData {
    double[] input;
    double[] output;

    public TrainingData(double[] input, double[] output) {
        this.input = input;
        this.output = output;
    }
}

public class Main {
    private static double[] forward(MLP network, double[] inputs) {
        return network.forward(inputs);
    }

    private static void training(MLP network, List<TrainingData> dataset) {
        for (TrainingData data:dataset){
            network.training(data.input, data.output);
        }
    }

    public static void main(String[] args) {
        List<TrainingData> dataset = new ArrayList<>();
        dataset.add(new TrainingData(new double[]{0.0, 0.0}, new double[]{0.0}));
        dataset.add(new TrainingData(new double[]{1.0, 0.0}, new double[]{1.0}));
        dataset.add(new TrainingData(new double[]{0.0, 1.0}, new double[]{1.0}));
        dataset.add(new TrainingData(new double[]{1.0, 1.0}, new double[]{1.0}));

        MLP network = new MLP(new int[]{2,2,1}, 0.25);

        System.out.println("Before training");
        System.out.println(forward(network, new double[]{0.0, 0.0})[0]);
        System.out.println(forward(network, new double[]{1.0, 0.0})[0]);
        System.out.println(forward(network, new double[]{0.0, 1.0})[0]);
        System.out.println(forward(network, new double[]{1.0, 1.0})[0]);
        for (int i=0; i<500000; i++) {
            training(network, dataset);
        }

        System.out.println("\nAfter training");
        System.out.println(forward(network, new double[]{0.0, 0.0})[0]);
        System.out.println(forward(network, new double[]{1.0, 0.0})[0]);
        System.out.println(forward(network, new double[]{0.0, 1.0})[0]);
        System.out.println(forward(network, new double[]{1.0, 1.0})[0]);
    }
}
