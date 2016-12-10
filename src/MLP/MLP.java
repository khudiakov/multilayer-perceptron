package MLP;
/**
 * Created by khudiakov on 08.12.2016.
 */

public class MLP {
    private Layer[] layers;
    private double step;

    private double localErrorsSum = 0.0;
    private int trainingsCount = 0;

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

    public double[] forward(double[] inputs) {
        double[] output = layers[0].exec(inputs);
        for (int i=1; i<this.layers.length; i++) {
            output = this.layers[i].exec(output);
        }

        return output;
    }

    public double getGlobalError() {
        return Math.sqrt(localErrorsSum/trainingsCount);
    }

    public void training(double[] inputs, double[] waiting) {
        forward(inputs);

        for (int i=this.layers.length-1; i>=0; i--) {
            for (int q=0; q<layers[i].neurons.length; q++) {
                Neuron qNeuron = layers[i].neurons[q];
                double difference = 0.0;

                if (i+1 != this.layers.length) {
                    for (int k = 0; k < this.layers[i + 1].neurons.length; k++) {
                        difference += this.layers[i + 1].neurons[k].delta * this.layers[i + 1].neurons[k].inputWeights[q];
                    }
                } else {
                    difference = waiting[q]-qNeuron.output;
                }

                qNeuron.delta = qNeuron.activationFunction.evaluateDerivative(qNeuron.output)*difference;
                qNeuron.biasWeight += step*qNeuron.delta *1;

                for (int j=0; j<qNeuron.inputWeights.length; j++) {
                    double output = (i>0?this.layers[i-1].neurons[j].output:inputs[j]);
                    qNeuron.inputWeights[j] += step*qNeuron.delta *output;
                }
            }
        }

        double sum = 0.0;
        for (Neuron neuron:this.layers[this.layers.length-1].neurons) {
            sum += Math.pow(neuron.delta, 2);
        }

        localErrorsSum += Math.sqrt(sum/this.layers[this.layers.length-1].neurons.length);
        trainingsCount++;
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
