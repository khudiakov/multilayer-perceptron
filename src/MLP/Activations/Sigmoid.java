package MLP.Activations;

/**
 * Created by khudiakov on 08.12.2016.
 */
public class Sigmoid extends ActivationFunction {
    @Override
    public double evaluate(double neuronInput) {
        return 1/(1+Math.pow(Math.E, -neuronInput));
    }
    @Override
    public double evaluateDerivative(double neuronOutput) {
        return neuronOutput*(1-neuronOutput);
    }
}
