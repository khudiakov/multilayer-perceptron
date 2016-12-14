package MLP.Activations;

public final class ActivationFunction {
    public static double evaluate(double neuronInput, ActivationType activationType) {
        switch (activationType) {
            case Sigmoid:
                return 1/(1+Math.pow(Math.E, -neuronInput));
            case TanH:
                return Math.tanh(neuronInput);
            default:
                return neuronInput;
        }
    }

    public static double evaluateDerivative(double neuronOutput, ActivationType activationType) {
        switch (activationType) {
            case Sigmoid:
                return neuronOutput*(1-neuronOutput);
            case TanH:
                return 1-Math.pow(neuronOutput, 2);
            default:
                return neuronOutput;
        }
    }
}
