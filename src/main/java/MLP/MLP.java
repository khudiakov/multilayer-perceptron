package MLP;

import MLP.Activations.ActivationFunction;
import MLP.Activations.ActivationType;
import datastream.Data;

import java.util.List;
import java.util.Random;

/**
 * Created by khudiakov on 08.12.2016.
 */

public class MLP {
    public Layer[] layers;

    private double learningRate;
    private boolean momentum;
    private boolean dropout;
    private double localErrorsSum = 0.0;
    private int trainingsCount = 0;

    public MLP(int[] layers, ActivationType defaultActivation, double learningRate, boolean momentum, boolean dropout) {
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.dropout = dropout;

        this.layers = new Layer[layers.length-1];
        for (int i=1; i<layers.length; i++) {
            this.layers[i-1] = new Layer(layers[i-1], layers[i], defaultActivation);
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
        if (trainingsCount == 0) {
            return Double.MAX_VALUE;
        }
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

                    qNeuron.delta = ActivationFunction.evaluateDerivative(qNeuron.output, qNeuron.activationFunctionType) * difference;
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
        Random random = new Random();

        for (int i=this.layers.length-1; i>=0; i--) {
            for (int q = 0; q < layers[i].neurons.length; q++) {
                Neuron qNeuron = layers[i].neurons[q];
                qNeuron.biasWeight += learningRate * qNeuron.delta * 1;

                for (int j = 0; j < qNeuron.inputWeights.length; j++) {
                    if(!this.dropout || random.nextBoolean()) {
                        double change = learningRate * qNeuron.delta * qNeuron.inputs[j];
                        if (momentum) {
                            change += 0.9 * qNeuron.weightsChange[j];
                            qNeuron.weightsChange[j] = change;
                        }
                        qNeuron.inputWeights[j] += change;
                    }
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
