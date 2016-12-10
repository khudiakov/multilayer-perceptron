package MLP;

import MLP.Activations.Sigmoid;

/**
 * Created by khudiakov on 08.12.2016.
 */
public class Layer {
    public Neuron[] neurons;

    public Layer(int previousNumberNeurons, int numberNeurons) {
        this.neurons = new Neuron[numberNeurons];
        final double glorotBengioConstant = Math.sqrt(6 / (numberNeurons + ((double) previousNumberNeurons)));
        for (int i=0; i<this.neurons.length; i++) {
            this.neurons[i] = new Neuron(previousNumberNeurons, new Sigmoid(), glorotBengioConstant);
        }
    }

    public double[] exec(double[] inputs) {
        double[] output = new double[neurons.length];
        for (int i=0; i<this.neurons.length; i++) {
            output[i] = this.neurons[i].exec(inputs);
        }
        return output;
    }
}
