package datastream;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * Created by khudiakov on 24/10/2016
 */

class DatasetClass {
    double[] outputs;
    public int count;

    public DatasetClass(double[] outputs) {
        this.outputs = outputs;
        this.count = 1;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof double[])
            return Arrays.equals(this.outputs, (double[]) obj);
        if (obj instanceof DatasetClass)
            return Arrays.equals(this.outputs, ((DatasetClass) obj).outputs);
        if (obj instanceof Data)
            return Arrays.equals(this.outputs, ((Data) obj).outputs);
        return false;
    }
}

public  class DataStream {
    final private BufferedReader fileStream;
    final private int batchSize = 100;
    private boolean normalize;
    private boolean randomize;
    private int nInput;
    private int nOutput;
    private List<Data> batch = new ArrayList<>();
    private List<DatasetClass> datasetClasses = new ArrayList<>();

    private double[] mean;
    private double[] std;

    public DataStream(String filepath, int nInput, int nOutput, boolean randomize, boolean normalize) throws IOException {
        fileStream = new BufferedReader(new FileReader(filepath));
        this.randomize = randomize;
        this.normalize = normalize;
        this.nInput = nInput;
        this.nOutput = nOutput;
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

            Optional<DatasetClass> datasetFilter = datasetClasses.stream().filter(x -> x.equals(outputs)).findFirst();
            if (datasetFilter.isPresent()) {
                datasetFilter.get().count++;
            } else {
                datasetClasses.add(new DatasetClass(outputs));
            }
        }

        if (normalize) {
            if (mean == null || std == null) {
                defineDatasetMeanStd();
            }
            normalizeBatch();
        }
    }

    public double getFulfillness(int numberOfExamplesOnOneParameter) {
        int satisfied = 0;
        int fulfill = nInput*numberOfExamplesOnOneParameter;
        for (DatasetClass datasetClass: datasetClasses) {
            if (datasetClass.count >= fulfill) {
                satisfied++;
            }
        }
        return satisfied/datasetClasses.size();
    }

    public List<Data> getNextBatch() throws IOException {
        this.loadBatch();
        if (this.randomize) {
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
