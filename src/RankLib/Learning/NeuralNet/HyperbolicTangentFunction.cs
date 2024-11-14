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
public class HyperbolicTangentFunction : ITransferFunction
{
	/// <summary>
	/// The scaling factor
	/// </summary>
	public const double ScalingFactor = 1.7159;

	private const double TwoThirds = 2.0 / 3.0;

	/// <inheritdoc />
	public double Compute(double x) => ScalingFactor * Math.Tanh(x * TwoThirds);

	/// <inheritdoc />
	public double ComputeDerivative(double x)
	{
		var output = Math.Tanh(x * TwoThirds);
		return ScalingFactor * (1.0 - output * output) * TwoThirds;
	}
}
