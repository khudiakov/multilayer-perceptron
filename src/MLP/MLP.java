package MLP;
/**
 * Created by khudiakov on 08.12.2016.
 */
public class MLP {
    Layer[] layers;
    double step;

    public MLP(int[] layers, double step) {
        this.step = step;
        this.layers = new Layer[layers.length-1];
        for (int i=1; i<layers.length; i++) {
            this.layers[i-1] = new Layer(layers[i-1], layers[i]);
        }
    }

    public double[] forward(double[] inputs) {
        double[] output = layers[0].exec(inputs);
        for (int i=1; i<this.layers.length; i++) {
            output = this.layers[i].exec(output);
        }

        return output;
    }

    public void training(double[] inputs, double[] waiting) {
        forward(inputs);

        Layer lastLayer = this.layers[this.layers.length-1];
        Layer beforeLastLayer = this.layers[this.layers.length-2];

        for (int i=0; i<lastLayer.neurons.length; i++) {
            Neuron iNeuron = lastLayer.neurons[i];
            iNeuron.difference = iNeuron.output*(1-iNeuron.output)*(waiting[i]-iNeuron.output);
            iNeuron.biasWeight += step*iNeuron.difference*1;

            for (int j=0; j<lastLayer.neurons[i].inputWeights.length; j++) {
                iNeuron.inputWeights[j] += step*iNeuron.difference*beforeLastLayer.neurons[j].output;
            }
        }

        for (int i=this.layers.length-2; i>1; i--) {
            for (int j=0; j<layers[i].neurons.length; j++) {
                double sumDifference = 0.0;
                for (int q=0; q<this.layers[i+1].neurons.length; q++) {
                    sumDifference += this.layers[i+1].neurons[q].difference*this.layers[i+1].neurons[q].inputWeights[j];
                }

                Neuron jNeuron = layers[i].neurons[j];
                jNeuron.difference = jNeuron.output*(1-jNeuron.output)*sumDifference;
                jNeuron.biasWeight += step*jNeuron.difference*1;

                for (int q=0; q<jNeuron.inputWeights.length; q++) {
                    jNeuron.inputWeights[q] += step*jNeuron.difference*this.layers[i-1].neurons[q].output;
                }
            }
        }
    }

    @Override
    public String toString() {
        String output = "";
        for (int l=this.layers.length-1; l>=0; l--) {
            output += "Layer: " + l;
            for (int n=0; n<this.layers[l].neurons.length; n++) {
                output += " Neuron: " + n + "(weights: ";
                for (int w=0; w<this.layers[l].neurons[n].inputWeights.length; w++) {
                    output += this.layers[l].neurons[n].inputWeights[w]+", ";
                }
                output += ") ";
            }
            output += "\n";
        }
        return output;
    }
}
