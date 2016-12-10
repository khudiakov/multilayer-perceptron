package MLP;

import MLP.Activations.ActivationFunction;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Created by khudiakov on 07.12.2016.
 */

public class Neuron {
    public ActivationFunction activationFunction;
    public double[] inputs;
    public double[] inputWeights;
    public double biasWeight;
    public double output;
    public double delta;

    public Neuron(int inputsNumber, ActivationFunction activationFunction, double glorotBengioConstant) {
        this.activationFunction = activationFunction;
        this.inputWeights = new double[inputsNumber];

        this.biasWeight = ThreadLocalRandom.current().nextDouble(-glorotBengioConstant, glorotBengioConstant);
        for (int i=0; i<inputWeights.length; i++) {
            this.inputWeights[i] = ThreadLocalRandom.current().nextDouble(-glorotBengioConstant, glorotBengioConstant);
        }
    }

    public double exec(double[] inputs) {
        this.inputs = inputs;

        double sum = this.biasWeight;

        for (int i=0; i<inputs.length; i++) {
            sum += inputs[i]*this.inputWeights[i];
        }
        this.output = this.activationFunction.evaluate(sum);
        return this.output;
    }
}
