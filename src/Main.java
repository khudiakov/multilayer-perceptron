/**
 * Created by khudiakov on 08.12.2016.
 */
import MLP.MLP;
import datastream.Data;
import datastream.DataStream;

import java.io.IOException;
import java.util.List;

public class Main {
    public static void main(String[] args) throws IOException {
        int nInput = 4;
        int nOutput = 1;

        MLP network = new MLP(new int[]{nInput, 2, nOutput}, 0.1);

        System.out.println("TRAINING");
        DataStream dataStream;

        int epochs = 250000;
        double percent = epochs/100;
        double progress = 0;
        String dataPath = "C:\\Users\\khudiakov\\Projects\\fi.muni\\NeuralNetwork\\src\\datastream\\data\\iris.data";
        for (int i=0; i<epochs; i++) {
            dataStream = new DataStream(dataPath, nInput, nOutput, true);
            List<Data> dataset;
            while (!(dataset=dataStream.getNextBatch()).isEmpty()) {
                network.training(dataset, false);
            }

            if (i/percent - progress>0.1) {
                progress = i/percent;
                System.out.print("\rProgress: "+progress+"%");
            }
        }
        System.out.println("\rProgress: 100%");
        System.out.println("Global training error: "+network.getGlobalError());
        System.out.println();


        System.out.println("TESTING");
        dataPath = "C:\\Users\\khudiakov\\Projects\\fi.muni\\NeuralNetwork\\src\\datastream\\data\\iris.data";
        dataStream = new DataStream(dataPath, nInput, nOutput, false);
        int all = 0;
        int success = 0;
        List<Data> dataset;
        while (!(dataset=dataStream.getNextBatch()).isEmpty()) {
            for (Data data : dataset) {
                all++;
                double[] out = network.forward(data.inputs);
                if (Math.abs(data.outputs[0] - out[0])<0.25) {
                    success++;
                }
            }
        }
        System.out.println("Success: "+(double)success/all*100+"%");
        System.out.println("Dataset fulfillness: "+dataStream.getFulfillness(3)*100+"%");
    }
}
