package datastream;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * Created by khudiakov on 24/10/2016
 */
public  class DataStream {
    final private BufferedReader fileStream;
    final private int batchLimit = 100;
    private boolean randomize;
    private int nInput;
    private int nOutput;
    private List<Data> batch = new ArrayList<>();

    public DataStream(String filepath, int nInput, int nOutput, boolean randomize) throws IOException {
        fileStream = new BufferedReader(new FileReader(filepath));
        this.randomize = randomize;
        this.nInput = nInput;
        this.nOutput = nOutput;
    }

    private void loadBatch() throws IOException {
        int count = 0;
        batch.clear();
        while (count < batchLimit) {
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
