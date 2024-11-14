namespace RankLib.Learning.NeuralNet;

/// <summary>
/// A list of neurons in a neural network
/// </summary>
public class ListNeuron : Neuron
{
	private double[] _d1 = [];
	private double[] _d2 = [];

	/// <summary>
	/// Initializes a new instance of <see cref="ListNeuron"/>, using <see cref="LogisticFunction"/>
	/// as the transfer function.
	/// </summary>
	/// <param name="learningRate">The learning rate</param>
	public ListNeuron(double learningRate) : base(learningRate, LogisticFunction.Instance)
	{
	}

	/// <summary>
	/// Initializes a new instance of <see cref="ListNeuron"/>
	/// </summary>
	/// <param name="learningRate">The learning rate</param>
	/// <param name="transferFunction">The transfer function</param>
	public ListNeuron(double learningRate, ITransferFunction transferFunction) : base(learningRate, transferFunction)
	{
	}

	public override void ComputeDelta(PropParameter param)
	{
		double sumLabelExp = 0;
		double sumScoreExp = 0;

		// Calculate sums of exponentiated labels and scores
		for (var i = 0; i < Outputs.Count; i++)
		{
			sumLabelExp += Math.Exp(param.Labels[i]);
			sumScoreExp += Math.Exp(Outputs[i]);
		}

		_d1 = new double[Outputs.Count];
		_d2 = new double[Outputs.Count];

		// Calculate d1 and d2 based on the above sums
		for (var i = 0; i < Outputs.Count; i++)
		{
			_d1[i] = Math.Exp(param.Labels[i]) / sumLabelExp;
			_d2[i] = Math.Exp(Outputs[i]) / sumScoreExp;
		}
	}

	public override void UpdateWeight(PropParameter param)
	{
		for (var k = 0; k < InLinks.Count; k++)
		{
			var s = InLinks[k];
			double dw = 0;

			// Update weights based on the difference between d1 and d2
			for (var l = 0; l < _d1.Length; l++)
				dw += (_d1[l] - _d2[l]) * s.Source.GetOutput(l);

			dw *= LearningRate;
			s.UpdateWeight(dw);
		}
	}
}
