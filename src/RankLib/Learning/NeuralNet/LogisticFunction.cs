namespace RankLib.Learning.NeuralNet;

using System;

/// <summary>
/// Logistic function, also known as the Sigmoid function, is a mathematical function
/// that maps input values to an output range between 0 and 1. It is commonly used in
/// neural networks for binary classification tasks, converting linear outputs into
/// probabilities.
/// </summary>
public class LogisticFunction : ITransferFunction
{
	public double Compute(double x) => 1.0 / (1.0 + Math.Exp(-x));

	public double ComputeDerivative(double x)
	{
		var output = Compute(x);
		return output * (1.0 - output);
	}
}
