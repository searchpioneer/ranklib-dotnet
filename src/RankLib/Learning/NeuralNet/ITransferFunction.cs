namespace RankLib.Learning.NeuralNet;

public interface ITransferFunction
{
	/// <summary>
	/// Computes the output.
	/// </summary>
	/// <param name="x">the input</param>
	/// <returns>the output</returns>
	double Compute(double x);

	/// <summary>
	/// Computes the derivative.
	/// </summary>
	/// <param name="x">the input</param>
	/// <returns>the derivative</returns>
	double ComputeDerivative(double x);
}
