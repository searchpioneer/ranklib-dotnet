namespace RankLib.Learning.NeuralNet;

public class HyperTangentFunction : ITransferFunction
{
	public double Compute(double x) => 1.7159 * Math.Tanh(x * 2 / 3);

	public double ComputeDerivative(double x)
	{
		var output = Math.Tanh(x * 2 / 3);
		return 1.7159 * (1.0 - output * output) * 2 / 3;
	}
}
