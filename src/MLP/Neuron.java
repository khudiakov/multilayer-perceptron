package MLP;

import MLP.Activations.ActivationFunction;

/**
 * Created by khudiakov on 07.12.2016.
 */

public class Neuron {
    public ActivationFunction activationFunction;
    public double[] inputWeights;
    public double biasWeight;
    public double output;

    public double difference = 0;

    public Neuron(int inputsNumber, ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
        this.inputWeights = new double[inputsNumber];

        this.biasWeight = Math.random();
        for (int i=0; i<inputWeights.length; i++) {
            this.inputWeights[i] = Math.random();
        }
    }

    public double exec(double[] inputs) {
        double sum = this.biasWeight;

        for (int i=0; i<inputs.length; i++) {
            sum += inputs[i]*this.inputWeights[i];
        }
        this.output = this.activationFunction.evaluate(sum);
        return this.output;
    }
}
