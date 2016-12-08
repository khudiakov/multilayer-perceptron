package datastream;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/**
 * Created by khudiakov on 24/10/2016
 */
public  class DataStream {
    final private BufferedReader fileStream;

    public DataStream(String filepath) throws IOException {
        fileStream = new BufferedReader(new FileReader(filepath));
        // skip description line
        fileStream.readLine();
    }

    public Data next() throws IOException {
        String line = fileStream.readLine();
        if (line == null) {
            return null;
        }

        String[] values = line.split(",");

        double[] inputs = new double[values.length-1];
        for (int i=0; i<inputs.length; i++) {
            inputs[i] = Double.parseDouble(values[i]);
        }

        double[] outputs = new double[1];
        outputs[0] = Double.parseDouble(values[values.length-1]);

        return new Data(inputs, outputs);
    }

    protected void finalize() throws Throwable {
        super.finalize();
        fileStream.close();
    }
}
