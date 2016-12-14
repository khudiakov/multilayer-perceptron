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
//        *arg: -i <n>
        int nInput = 64;
//        *arg: -o <n>
        int nOutput = 1;

//        arg: -l <"n1,n2,n3">
        String strLayers = "20";
//        arg: --af <[sigmoid, tanh]>
        ActivationType defaultActivation = ActivationType.Sigmoid;
//        arg: --lr <d>
        double learningRate = 0.01;
//        arg: --momentum
        boolean momentum = true;
//        arg: --dropout
        boolean dropout = true;

//        *arg: --training-dataset <path>
        String trainingFilePath = "C:\\Users\\khudiakov\\Projects\\fi.muni\\NeuralNetwork\\src\\datastream\\data\\optdigits.tra";
//        *arg: --testing-dataset <path>
        String testingFilePath = "C:\\Users\\khudiakov\\Projects\\fi.muni\\NeuralNetwork\\src\\datastream\\data\\optdigits.tes";
//        arg: --bs <n>
        int batchSize = 100;
//        arg: --randomize
        boolean randomize = true;
//        arg: --normalize
        boolean normalize = true;

//        arg: --me <n>
        int maxEpochs = 1000;
//        arg: --tge <d>
        double targetGlobalError = 0.05;

//        arg: --sm <d>
        double successMistake = 0.05;

//        Logic part
        String[] hiddenLayers = strLayers.split(",");
        int[] layers = new int[hiddenLayers.length+2];
        layers[0] = nInput;
        layers[layers.length-1] = nOutput;
        for (int i=0; i<hiddenLayers.length; i++) {
            layers[i+1] = Integer.parseInt(hiddenLayers[i].trim());
        }
        MLP network = new MLP(layers, defaultActivation, learningRate, momentum, dropout);
        DataStream dataStream = new DataStream(trainingFilePath, testingFilePath, nInput, nOutput, batchSize, randomize, normalize);

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
                if (Math.abs(data.outputs[0] - out[0])<successMistake) {
                    success++;
                }
            }
        }
        System.out.println("Success: "+(double)success/all*100+"%");
    }
}
