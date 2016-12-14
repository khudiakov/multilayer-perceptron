/**
 * Created by khudiakov on 08.12.2016.
 */

import MLP.Activations.ActivationType;
import MLP.MLP;
import datastream.Data;
import datastream.DataStream;

import java.io.IOException;
import java.util.List;
import java.util.Scanner;

public class Main {
    private static String findArgumentValue(String[] args, String name, String defaultValue) {
        for (int i=0; i<args.length; i++) {
            if (args[i].equals(name)) {
                return (i+1<args.length?args[i+1]:defaultValue);
            }
        }
        return defaultValue;
    }
    private static boolean isArgument(String[] args, String name) {
        for (String arg : args) {
            if (arg.equals(name)) {
                return true;
            }
        }
        return false;
    }
    private static String formatTime(long totalTime) {
        long second = (totalTime / 1000) % 60;
        long minute = (totalTime / (1000 * 60)) % 60;
        long hour  = (totalTime / (1000 * 60 * 60)) % 24;
        return hour+"h "+minute+"m "+second+"s";
    }

    public static void main(String[] args) throws IOException {
//        Parameters part
//        *arg: -i <n>
        int nInput = Integer.parseInt(findArgumentValue(args, "-i", "-1"));
        if (nInput == -1) {
            System.out.println("Required -i parameter");
            return;
        }
//        *arg: -o <n>
        int nOutput = Integer.parseInt(findArgumentValue(args, "-o", "-1"));
        if (nOutput == -1) {
            System.out.println("Required -o parameter");
            return;
        }
//        arg: -l <"n1,n2,n3">
        String strLayers = findArgumentValue(args, "-l", "");
//        arg: -af <[sigmoid, tanh]>
        ActivationType defaultActivation;
        String af = findArgumentValue(args, "-af", "sigmoid");
        switch (af) {
            case ("sigmoid"):
                defaultActivation = ActivationType.Sigmoid;
                break;
            case ("tanh"):
                defaultActivation = ActivationType.TanH;
                break;
            default:
                System.out.println("Incorrect activation function");
                return;
        }
//        arg: -lr <d>
        double learningRate = Double.parseDouble(findArgumentValue(args, "-lr", "0.01"));
//        arg: --momentum
        boolean momentum = isArgument(args, "--momentum");
//        arg: --dropout
        boolean dropout = isArgument(args, "--dropout");

//        *arg: --training-dataset <path>
        String trainingFilePath = findArgumentValue(args, "--training-dataset", "");
        if (trainingFilePath.isEmpty()) {
            System.out.println("Required --training-dataset parameter");
            return;
        }
//        *arg: --testing-dataset <path>
        String testingFilePath = findArgumentValue(args, "--testing-dataset", "");
        if (testingFilePath.isEmpty()) {
            System.out.println("Required --testing-dataset parameter");
            return;
        }
//        arg: -bs <n>
        int batchSize = Integer.parseInt(findArgumentValue(args, "-bs", "10"));
//        arg: --stochastic
        boolean stochastic = isArgument(args, "--stochastic");
//        arg: --randomize
        boolean randomize = isArgument(args, "--randomize");
//        arg: --normalize
        boolean normalize = isArgument(args, "--normalize");

//        arg: -me <n>
        int maxEpochs = Integer.parseInt(findArgumentValue(args, "-me", "1000"));
//        arg: -tge <d>
        double targetGlobalError = Double.parseDouble(findArgumentValue(args, "-tge", "0.1"));

//        arg: -sm <d>
        double successMistake = Double.parseDouble(findArgumentValue(args, "-sm", "0.1"));

//        Preparing part
        System.out.println("NETWORK PARAMETERS:");
        System.out.println("\tDataset:");
        System.out.println("\t\ttesting dataset path: "+testingFilePath);
        System.out.println("\t\ttraining dataset path: "+trainingFilePath);
        if (stochastic) {
            System.out.println("\t\tgradient decent is stochastic");
        } else {
            System.out.println("\t\tgradient decent is mini-batch, batch size is "+batchSize);
        }
        System.out.println("\t\tnormalization is: "+normalize);
        System.out.println("\t\trandomization is: "+randomize);

        System.out.println("\n\tMLP:");
        System.out.println("\t\tinputs: "+nInput);
        System.out.println("\t\thidden layers: "+strLayers);
        System.out.println("\t\toutputs: "+nOutput);
        System.out.println("\t\tactivation function is: "+defaultActivation);
        System.out.println("\t\tlearning rate is: "+learningRate);
        System.out.println("\t\tmomentum is: "+momentum);
        System.out.println("\t\tdropout is: "+dropout);

        System.out.println("\n\tTraining:");
        System.out.println("\t\tmaximum number epochs is: "+maxEpochs);
        System.out.println("\t\ttarget global error is: "+targetGlobalError);

        System.out.println("\n\tTesting:");
        System.out.println("\t\tallowed success mistake is: "+successMistake);

        System.out.println("\n");


        System.out.println("Parameters are correct? (Enter `no` to abort, enter `yes` otherwise)");
        Scanner scanner = new Scanner(System.in);

        if(scanner.next().equalsIgnoreCase("no")) {
            return;
        }

//        Logic part
        String[] hiddenLayers = new String[0];
        if (!strLayers.isEmpty()) {
            hiddenLayers = strLayers.split(",");
        }
        int[] layers = new int[hiddenLayers.length+2];
        layers[0] = nInput;
        layers[layers.length-1] = nOutput;
        for (int i=0; i<hiddenLayers.length; i++) {
            layers[i+1] = Integer.parseInt(hiddenLayers[i].trim());
        }
        MLP network = new MLP(layers, defaultActivation, learningRate, momentum, dropout);
        DataStream dataStream = new DataStream(trainingFilePath, testingFilePath, nInput, nOutput, batchSize, randomize, normalize);

        List<Data> dataset;

        long startTime = System.currentTimeMillis();
        while (network.getGlobalError()>targetGlobalError && maxEpochs-->0) {
            dataStream.load(true);
            while (!(dataset=dataStream.getNextBatch()).isEmpty()) {
                network.training(dataset, stochastic);
            }
            System.out.print("\rTraining progress: \tGlobal training error: "+network.getGlobalError()+"\tLeft epochs: "+maxEpochs);
        }
        long endTime   = System.currentTimeMillis();
        System.out.println();
        long totalTime = endTime - startTime;
        System.out.println("Training complete in: "+formatTime(totalTime));
        System.out.println();


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
        System.out.println("Testing success: "+(double)success/all*100+"%");
    }
}
