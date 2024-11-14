namespace RankLib.Learning.NeuralNet;

using System;

/// <summary>
/// Logistic (Sigmoid) function that maps input values to an output range between 0 and 1.
/// </summary>
public class LogisticFunction : ITransferFunction
{
	public static readonly LogisticFunction Instance = new();

	public double Compute(double x) => 1.0 / (1.0 + Math.Exp(-x));

	public double ComputeDerivative(double x)
	{
		var output = Compute(x);
		return output * (1.0 - output);
	}
}
