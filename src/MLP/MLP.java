package MLP;

import datastream.Data;

import java.util.List;

/**
 * Created by khudiakov on 08.12.2016.
 */

public class MLP {
    private Layer[] layers;
    private double learningRate;

    private double localErrorsSum = 0.0;
    private int trainingsCount = 0;

    public MLP(int[] layers, double startLearningRate) {
        this.learningRate = startLearningRate;
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

    public double getGlobalError() {
        return Math.sqrt(localErrorsSum/trainingsCount);
    }

    public void training(List<Data> batch, boolean stochastic) {
        for (Data data:batch) {
            forward(data.inputs);

            for (int i = this.layers.length - 1; i >= 0; i--) {
                for (int q = 0; q < layers[i].neurons.length; q++) {
                    Neuron qNeuron = layers[i].neurons[q];
                    double difference = 0.0;

                    if (i + 1 != this.layers.length) {
                        for (int k = 0; k < this.layers[i + 1].neurons.length; k++) {
                            difference += this.layers[i + 1].neurons[k].delta * this.layers[i + 1].neurons[k].inputWeights[q];
                        }
                    } else {
                        difference = data.outputs[q] - qNeuron.output;
                    }

                    qNeuron.delta = qNeuron.activationFunction.evaluateDerivative(qNeuron.output) * difference;
                }
            }
            double sum = 0.0;
            for (Neuron neuron:this.layers[this.layers.length-1].neurons) {
                sum += Math.pow(neuron.delta, 2);
            }

            localErrorsSum += Math.sqrt(sum/this.layers[this.layers.length-1].neurons.length);
            trainingsCount++;

            if (stochastic) {
                updateWeights();
            }
        }

        if (!stochastic) {
            updateWeights();
        }
    }

    private void updateWeights() {
        for (int i=this.layers.length-1; i>=0; i--) {
            for (int q = 0; q < layers[i].neurons.length; q++) {
                Neuron qNeuron = layers[i].neurons[q];
                qNeuron.biasWeight += learningRate * qNeuron.delta * 1;

                for (int j = 0; j < qNeuron.inputWeights.length; j++) {
                    qNeuron.inputWeights[j] += learningRate * qNeuron.delta * qNeuron.inputs[j];
                }
            }
        }
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
