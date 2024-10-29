namespace RankLib.Learning.NeuralNet;

public class ListNeuron : Neuron
{
	private double[] _d1 = [];
	private double[] _d2 = [];

	public ListNeuron(double learningRate) : base(learningRate)
	{
	}

	public override void ComputeDelta(PropParameter param)
	{
		double sumLabelExp = 0;
		double sumScoreExp = 0;

		// Calculate sums of exponentiated labels and scores
		for (var i = 0; i < _outputs.Count; i++)
		{
			sumLabelExp += Math.Exp(param.Labels[i]);
			sumScoreExp += Math.Exp(_outputs[i]);
		}

		_d1 = new double[_outputs.Count];
		_d2 = new double[_outputs.Count];

		// Calculate d1 and d2 based on the above sums
		for (var i = 0; i < _outputs.Count; i++)
		{
			_d1[i] = Math.Exp(param.Labels[i]) / sumLabelExp;
			_d2[i] = Math.Exp(_outputs[i]) / sumScoreExp;
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
