using RankLib.Utilities;

namespace RankLib.Learning.NeuralNet;

public class Synapse
{
	public Synapse(Neuron source, Neuron target)
	{
		Source = source;
		Target = target;
		Source.OutLinks.Add(this);
		Target.InLinks.Add(this);
		Weight = (ThreadsafeSeedableRandom.Shared.Next(2) == 0 ? 1 : -1) * ThreadsafeSeedableRandom.Shared.NextDouble() / 10;
	}

	public Neuron Source { get; }

	public Neuron Target { get; }

	public double Weight { get; set; }

	public double WeightAdjustment { get; private set; }

	public void UpdateWeight(double weightAdjustment)
	{
		WeightAdjustment = weightAdjustment;
		Weight += WeightAdjustment;
	}
}
