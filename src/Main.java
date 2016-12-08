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
    private static double[] forward(MLP network, double[] input) {
        return network.forward(input);
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

        MLP network = new MLP(new int[]{2,2,1}, 0.15);

        for (int i=0; i<50000; i++) {
            if (i%1000==0) {
                System.out.println("After " + i + " trainings: "+network.getAvgDifference());
            }
            training(network, dataset);
        }
        System.out.println("\nAfter all trainings: "+network.getAvgDifference());


        double[][][] weights = network.getWeights();
        MLP network_copy = new MLP(weights, 0.15);

        System.out.println(network);
        System.out.println(network_copy);

        training(network_copy, dataset);
        System.out.println("\nCopy network: "+network_copy.getAvgDifference());
    }
}
