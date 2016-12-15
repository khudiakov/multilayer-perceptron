package datastream;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Created by khudiakov on 24/10/2016
 */

public  class DataStream {
    public boolean training;

    private BufferedReader fileStream;
    private String trainingFilepath;
    private String testingFilepath;
    private int batchSize;
    private boolean normalize;
    private boolean randomize;
    private int nInput;
    private int nOutput;
    private List<Data> batch = new ArrayList<>();

    private double[] mean;
    private double[] std;

    public DataStream(String trainingFilepath, String testingFilepath, int nInput, int nOutput,int batchSize, boolean randomize, boolean normalize) throws IOException {
        this.trainingFilepath = trainingFilepath;
        this.testingFilepath = testingFilepath;
        this.nInput = nInput;
        this.nOutput = nOutput;
        this.batchSize = batchSize;
        this.randomize = randomize;
        this.normalize = normalize;
    }

    public void load(boolean trainingDataset) throws IOException {
        this.training = trainingDataset;
        if (trainingDataset) {
            fileStream = new BufferedReader(new FileReader(this.trainingFilepath));
        } else {
            fileStream = new BufferedReader(new FileReader(this.testingFilepath));
        }
    }

    private void defineDatasetMeanStd() throws IOException {
        mean = new double[this.nInput];
        std = new double[this.nInput];

        double[] inputSum = new double[this.nInput];
        double[] inputSqrSum = new double[this.nInput];
        int count = 0;

        for (Data data: this.batch) {
            count++;
            for (int i=0; i<data.inputs.length; i++) {
                inputSum[i] += data.inputs[i];
                inputSqrSum[i] += Math.pow(data.inputs[i], 2);
            }
        }

        if (count > 1) {
            for (int i = 0; i < inputSum.length; i++) {
                mean[i] = inputSum[i] / count;
                std[i] = (1.0 / (count - 1)) * inputSqrSum[i];
            }
        }
    }

    private void normalizeBatch() {
        for (Data data: batch) {
            for (int i=0; i<data.inputs.length && std[i]!=0; i++) {
                data.inputs[i] = (data.inputs[i]-mean[i])/std[i];
            }
        }
    }

    private void loadBatch() throws IOException {
        int count = 0;
        batch.clear();
        while (count < batchSize) {
            count++;
            String line = fileStream.readLine();
            if (line == null) {
                return;
            }

            String[] values = line.split(",");
            double[] inputs = new double[this.nInput];
            for (int i=0; i<inputs.length; i++) {
                inputs[i] = Double.parseDouble(values[i]);
            }

            double[] outputs = new double[this.nOutput];
            for (int i=0; i<outputs.length; i++) {
                outputs[i] = Double.parseDouble(values[nInput+i]);
            }
            batch.add(new Data(inputs, outputs));
        }

        if (normalize) {
            if (mean == null || std == null) {
                defineDatasetMeanStd();
            }
            normalizeBatch();
        }
    }

    public List<Data> getNextBatch() throws IOException {
        this.loadBatch();
        if (this.training && this.randomize) {
            long seed = System.nanoTime();
            Collections.shuffle(batch, new Random(seed));
        }
        return this.batch;
    }

    protected void finalize() throws Throwable {
        super.finalize();
        fileStream.close();
    }
}
