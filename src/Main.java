/**
 * Created by khudiakov on 08.12.2016.
 */
import MLP.Activations.ActivationType;
import MLP.MLP;
import datastream.Data;
import datastream.DataStream;

import java.io.IOException;
import java.util.List;

public class Main {
    public static void main(String[] args) throws IOException {
//        Parameters part
        int nInput = 64;
        int nOutput = 1;

        String strLayers = "20";
        ActivationType defaultActivation = ActivationType.Sigmoid;
        double learningRate = 0.01;
        boolean momentum = true;
        boolean dropout = true;

        String trainingFilePath = "C:\\Users\\khudiakov\\Projects\\fi.muni\\NeuralNetwork\\src\\datastream\\data\\optdigits.tra";
        String testingFilePath = "C:\\Users\\khudiakov\\Projects\\fi.muni\\NeuralNetwork\\src\\datastream\\data\\optdigits.tes";
        int batchSize = 100;
        boolean randomize = true;
        boolean normilize = true;

        int maxEpochs = 1000;
        double targetGlobalError = 0.05;

        double outputMistake = 0.05;

//        Logic part
        String[] hiddenLayers = strLayers.split(",");
        int[] layers = new int[hiddenLayers.length+2];
        layers[0] = nInput;
        layers[layers.length-1] = nOutput;
        for (int i=0; i<hiddenLayers.length; i++) {
            layers[i+1] = Integer.parseInt(hiddenLayers[i].trim());
        }
        MLP network = new MLP(layers, defaultActivation, learningRate, momentum, dropout);
        DataStream dataStream = new DataStream(trainingFilePath, testingFilePath, nInput, nOutput, batchSize, randomize, normilize);

        List<Data> dataset;

        System.out.println("TRAINING");
        while (network.getGlobalError()>targetGlobalError && maxEpochs-->0) {
            dataStream.load(true);
            while (!(dataset=dataStream.getNextBatch()).isEmpty()) {
                network.training(dataset, false);
            }
            System.out.print("\rGlobal training error: "+network.getGlobalError());
        }
        System.out.println("\rGlobal training error: "+network.getGlobalError());
        System.out.println();


        System.out.println("TESTING");
        int all = 0;
        int success = 0;
        dataStream.load(false);
        while (!(dataset=dataStream.getNextBatch()).isEmpty()) {
            for (Data data : dataset) {
                all++;
                double[] out = network.forward(data.inputs);
                if (Math.abs(data.outputs[0] - out[0])<outputMistake) {
                    success++;
                }
            }
        }
        System.out.println("Success: "+(double)success/all*100+"%");
    }
}
