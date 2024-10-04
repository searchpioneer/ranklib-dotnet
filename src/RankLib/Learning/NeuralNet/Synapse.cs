namespace RankLib.Learning.NeuralNet;

using System;

public class Synapse
{
    private static readonly Random random = new Random();
    private double _weight = 0.0;
    private double _dW = 0.0; // last weight adjustment
    private readonly Neuron _source;
    private readonly Neuron _target;

    public Synapse(Neuron source, Neuron target)
    {
        _source = source;
        _target = target;
        _source.GetOutLinks().Add(this);
        _target.GetInLinks().Add(this);
        _weight = (random.Next(2) == 0 ? 1 : -1) * random.NextDouble() / 10;
    }

    public Neuron Source => _source;

    public Neuron Target => _target;

    public double Weight
    {
        get => _weight;
        set => _weight = value;
    }

    public double LastWeightAdjustment => _dW;

    public void SetWeightAdjustment(double dW)
    {
        _dW = dW;
    }

    public void UpdateWeight()
    {
        _weight += _dW;
    }
}
