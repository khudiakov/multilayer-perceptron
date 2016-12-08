/**
 * Created by khudiakov on 08.12.2016.
 */
import MLP.MLP;
import datastream.Data;
import datastream.DataStream;

import java.io.IOException;

public class Main {
    private static double[] forward(MLP network, double[] input) {
        return network.forward(input);
    }

    private static void training(MLP network, Data data) {
        network.training(data.inputs, data.outputs);
    }

    public static void main(String[] args) throws IOException {
        MLP network = new MLP(new int[]{1,4,2,2,1}, 0.15);

        String dataPath = "C:\\Users\\khudiakov\\Projects\\fi.muni\\NeuralNetwork\\src\\datastream\\data\\sin.data";
        DataStream dataStream;
        Data data;

        System.out.println(network);
        for (int i=0; i<10000; i++) {
            dataStream = new DataStream(dataPath);
            while ((data = dataStream.next()) != null) {
                training(network, data);
            }
        }
        System.out.println();
        System.out.println(network);
        System.out.println();

        System.out.println(network.getAvgDifference());

        double[] out = forward(network, new double[]{2.586});
        System.out.println(out[0]);
    }
}
