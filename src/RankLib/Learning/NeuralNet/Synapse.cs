namespace RankLib.Learning.NeuralNet;

using System;

public class Synapse
{
	private static readonly Random Random = Random.Shared;
	private readonly Neuron _source;
	private readonly Neuron _target;

	public Synapse(Neuron source, Neuron target)
	{
		_source = source;
		_target = target;
		_source.OutLinks.Add(this);
		_target.InLinks.Add(this);
		Weight = (Random.Next(2) == 0 ? 1 : -1) * Random.NextDouble() / 10;
	}

	public Neuron Source => _source;

	public Neuron Target => _target;

	public double Weight { get; set; }

	public double WeightAdjustment { get; set; }

	public void UpdateWeight() => Weight += WeightAdjustment;
}
