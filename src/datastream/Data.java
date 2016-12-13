package datastream;

/**
 * Created by khudiakov on 24/10/2016
 */
public class Data {
    public double[] inputs;
    public double[] outputs;

    public Data(double[] input, double[] output) {
        this.inputs = input;
        this.outputs = output;
    }

    @Override
    public String toString() {
        String output = "Inputs: ( ";
        for (double input: this.inputs) {
            output += input+" ";
        }
        output += ")\nOutputs: ( ";
        for (double out: this.outputs) {
            output += out+" ";
        }
        output += ")\n";
        return output;
    }
}
