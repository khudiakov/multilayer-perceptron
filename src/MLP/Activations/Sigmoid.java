package MLP.Activations;

/**
 * Created by khudiakov on 08.12.2016.
 */
public class Sigmoid extends ActivationFunction {
    @Override
    public double evaluate(double value) {
        return 1/(1+Math.pow(Math.E, -value));
    }
    @Override
    public double evaluateDerivative(double value) {
        return value*(1-value);
    }
}