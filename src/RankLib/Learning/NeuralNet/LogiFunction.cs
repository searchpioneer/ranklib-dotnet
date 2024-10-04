namespace RankLib.Learning.NeuralNet;

using System;

public class LogiFunction : ITransferFunction
{
	public double Compute(double x) => 1.0 / (1.0 + Math.Exp(-x));

	public double ComputeDerivative(double x)
	{
		var output = Compute(x);
		return output * (1.0 - output);
	}
}
