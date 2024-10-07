namespace RankLib.Learning.NeuralNet;

public class HyperTangentFunction : ITransferFunction
{
	private const double ScalingFactor = 1.7159;

	public double Compute(double x) => ScalingFactor * Math.Tanh(x * 2 / 3);

	public double ComputeDerivative(double x)
	{
		var output = Math.Tanh(x * 2 / 3);
		return ScalingFactor * (1.0 - output * output) * 2 / 3;
	}
}
