namespace RankLib.Learning.NeuralNet;

public class HyperTangentFunction : ITransferFunction
{
    public double Compute(double x)
    {
        return 1.7159 * Math.Tanh(x * 2 / 3);
    }

    public double ComputeDerivative(double x)
    {
        double output = Math.Tanh(x * 2 / 3);
        return 1.7159 * (1.0 - output * output) * 2 / 3;
    }
}