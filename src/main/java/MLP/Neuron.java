package MLP;

import MLP.Activations.ActivationFunction;
import MLP.Activations.ActivationType;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Created by khudiakov on 07.12.2016.
 */

public class Neuron {
    public ActivationType activationFunctionType;
    public double[] inputs;
    public double[] inputWeights;
    public double[] weightsChange;
    public double biasWeight;
    public double output;
    public double delta;

    public Neuron(int inputsNumber, ActivationType activationFunctionType, double glorotBengioConstant) {
        this.activationFunctionType = activationFunctionType;
        this.inputWeights = new double[inputsNumber];
        this.weightsChange = new double[inputsNumber];

        this.biasWeight = 0.0;
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
        this.output = ActivationFunction.evaluate(sum, activationFunctionType);
        return this.output;
    }
}
