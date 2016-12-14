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
        int nInput = 64;
        int nOutput = 1;

        MLP network = new MLP(new int[]{nInput, 20, nOutput});

        System.out.println("TRAINING");
        DataStream dataStream;

        String dataPath = "C:\\Users\\khudiakov\\Projects\\fi.muni\\NeuralNetwork\\src\\datastream\\data\\optdigits.tra";
        double maxEpochs = 5000;
        while (network.getGlobalError()>0.05 && maxEpochs-->0) {
            dataStream = new DataStream(dataPath, nInput, nOutput, true, true);
            List<Data> dataset;
            while (!(dataset=dataStream.getNextBatch()).isEmpty()) {
                network.training(dataset, false);
            }
            System.out.print("\rGlobal training error: "+network.getGlobalError());
        }
        System.out.println("\rGlobal training error: "+network.getGlobalError());
        System.out.println();


        System.out.println("TESTING");
        dataPath = "C:\\Users\\khudiakov\\Projects\\fi.muni\\NeuralNetwork\\src\\datastream\\data\\optdigits.tes";
        dataStream = new DataStream(dataPath, nInput, nOutput, false, true);
        int all = 0;
        int success = 0;
        List<Data> dataset;
        while (!(dataset=dataStream.getNextBatch()).isEmpty()) {
            for (Data data : dataset) {
                all++;
                double[] out = network.forward(data.inputs);
                if (Math.abs(data.outputs[0] - out[0])<0.05) {
                    success++;
                }
            }
        }
        System.out.println("Success: "+(double)success/all*100+"%");
        System.out.println("Dataset fulfillness: "+dataStream.getFulfillness(3)*100+"%");
    }
}
