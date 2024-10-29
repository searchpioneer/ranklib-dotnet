namespace RankLib.Learning.NeuralNet;

/// <summary>
/// Scaled Hyperbolic Tangent (tanh) function that maps inputs to an output range
/// between -1.7159 and 1.7159.
/// </summary>
/// <remarks>
/// <a href="https://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf">
/// Y. LeCun, L. Bottou, G. Orr and K. Muller: Efficient BackProp, in Orr, G. and Muller K. (Eds),
/// Neural Networks: Tricks of the trade, Springer, 1998,
/// </a>
/// </remarks>
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
