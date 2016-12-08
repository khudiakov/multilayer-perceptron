package MLP.Activations;

/**
 * Created by khudiakov on 08.12.2016.
 */
public final class Sigmoid {
    public static double evaluate(double value) {
        return 1/(1+Math.pow(Math.E, -value));
    }
    public static double evaluateDerivate(double value) {
        return value*(1-value);
    }
}
