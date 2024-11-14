using RankLib.Utilities;

namespace RankLib.Learning.NeuralNet;

/// <summary>
/// A connection between two neurons,
/// transmitting signals from the source neuron to the target neuron,
/// with an associated weight that determines the strength of the connection and
/// influences the learning process during training.
/// </summary>
public class Synapse
{
	private Synapse(Neuron source, Neuron target)
	{
		Source = source;
		Target = target;
		Weight = (ThreadsafeSeedableRandom.Shared.Next(2) == 0 ? 1 : -1) *
			ThreadsafeSeedableRandom.Shared.NextDouble() / 10;
	}

	/// <summary>
	/// Connects two neurons together
	/// </summary>
	/// <param name="source">The source neuron</param>
	/// <param name="target">The target neuron</param>
	public static void Connect(Neuron source, Neuron target)
	{
		var synapse = new Synapse(source, target);
		source.OutLinks.Add(synapse);
		target.InLinks.Add(synapse);
	}

	/// <summary>
	/// Gets the source neuron.
	/// </summary>
	public Neuron Source { get; }

	/// <summary>
	/// Gets the target neuron.
	/// </summary>
	public Neuron Target { get; }

	/// <summary>
	/// Gets or sets the weight.
	/// </summary>
	public double Weight { get; set; }

	/// <summary>
	/// Gets the weight adjustment.
	/// </summary>
	public double WeightAdjustment { get; private set; }

	/// <summary>
	/// Updates <see cref="WeightAdjustment"/>, and <see cref="Weight"/> with the weight adjustment.
	/// </summary>
	/// <param name="weightAdjustment">The weight adjustment.</param>
	public void UpdateWeight(double weightAdjustment)
	{
		WeightAdjustment = weightAdjustment;
		Weight += WeightAdjustment;
	}
}
