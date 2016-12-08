package MLP;
/**
 * Created by khudiakov on 08.12.2016.
 */
public class MLP {
    private Layer[] layers;
    private double step;

    public MLP(int[] layers, double step) {
        this.step = step;
        this.layers = new Layer[layers.length-1];
        for (int i=1; i<layers.length; i++) {
            this.layers[i-1] = new Layer(layers[i-1], layers[i]);
        }
    }
    public MLP(double[][][] weights, double step) {
        this.step = step;
        this.layers = new Layer[weights.length];
        for (int l=0; l<weights.length; l++) {
            this.layers[l] = new Layer(weights[l][0].length-1, weights[l].length);
            for (int n=0; n<weights[l].length; n++) {
                for (int w=0; w<weights[l][n].length-1; w++) {
                    this.layers[l].neurons[n].inputWeights[w]=weights[l][n][w];
                }
                this.layers[l].neurons[n].biasWeight=weights[l][n][weights[l][n].length-1];
            }
        }
    }

    public double getAvgDifference() {
        double sum = 0.0;
        for (Neuron neuron: layers[layers.length-1].neurons) {
            sum += neuron.difference;
        }
        return sum/layers[layers.length-1].neurons.length;
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
            iNeuron.difference = iNeuron.activationFunction.evaluateDerivative(iNeuron.output)*(waiting[i]-iNeuron.output);
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
                jNeuron.difference = jNeuron.activationFunction.evaluateDerivative(jNeuron.output)*sumDifference;
                jNeuron.biasWeight += step*jNeuron.difference*1;

                for (int q=0; q<jNeuron.inputWeights.length; q++) {
                    jNeuron.inputWeights[q] += step*jNeuron.difference*this.layers[i-1].neurons[q].output;
                }
            }
        }
    }

    public double[][][] getWeights() {
        double[][][] weights = new double[layers.length][][];
        for (int i=0; i<layers.length; i++) {
            weights[i] = new double[layers[i].neurons.length][];
            for (int j=0; j<layers[i].neurons.length; j++) {
                weights[i][j] = new double[layers[i].neurons[j].inputWeights.length+1];
                for (int z=0; z<layers[i].neurons[j].inputWeights.length; z++) {
                    weights[i][j][z] = layers[i].neurons[j].inputWeights[z];
                }
                weights[i][j][weights[i][j].length-1] = layers[i].neurons[j].biasWeight;
            }
        }
        return weights;
    }

    @Override
    public String toString() {
        String output = "";
        for (int l=this.layers.length-1; l>=0; l--) {
            output += "Layer " + l + ":\n";
            for (int n=0; n<this.layers[l].neurons.length; n++) {
                output += "\tNeuron " + n + ": weights: ";
                for (int w=0; w<this.layers[l].neurons[n].inputWeights.length; w++) {
                    output += this.layers[l].neurons[n].inputWeights[w]+", ";
                }
                output += "bias: "+this.layers[l].neurons[n].biasWeight;
                output += "\t\t";
            }
            output += "\n";
        }
        return output;
    }
}
