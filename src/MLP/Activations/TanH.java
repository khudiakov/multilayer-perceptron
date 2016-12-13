package MLP.Activations;

/**
 * Created by khudiakov on 12.12.2016.
 */
public class TanH extends ActivationFunction {
    @Override
    public double evaluate(double neuronInput) {
        return Math.tanh(neuronInput);
    }
    @Override
    public double evaluateDerivative(double neuronOutput) {
        return 1-Math.pow(neuronOutput, 2);
    }
}
