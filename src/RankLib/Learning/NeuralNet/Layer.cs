namespace RankLib.Learning.NeuralNet;

public class Layer
{
	private readonly List<Neuron> _neurons;

	public Layer(int count)
	{
		_neurons = new List<Neuron>(count);
		for (var i = 0; i < count; i++)
		{
			_neurons.Add(new Neuron());
		}
	}

	public Layer(int count, NeuronType neuronType)
	{
		_neurons = new List<Neuron>(count);
		for (var i = 0; i < count; i++)
		{
			_neurons.Add(neuronType == NeuronType.Single
				? new Neuron()
				: new ListNeuron());
		}
	}

	public Neuron this[int i] => _neurons[i];

	public int Count => _neurons.Count;

	public void ComputeOutput(int i)
	{
		foreach (var neuron in _neurons)
		{
			neuron.ComputeOutput(i);
		}
	}

	public void ComputeOutput()
	{
		foreach (var neuron in _neurons)
		{
			neuron.ComputeOutput();
		}
	}

	public void ClearOutputs()
	{
		foreach (var neuron in _neurons)
		{
			neuron.ClearOutputs();
		}
	}

	public void ComputeDelta(PropParameter param)
	{
		foreach (var neuron in _neurons)
		{
			neuron.ComputeDelta(param);
		}
	}

	public void UpdateDelta(PropParameter param)
	{
		foreach (var neuron in _neurons)
		{
			neuron.UpdateDelta(param);
		}
	}

	public void UpdateWeight(PropParameter param)
	{
		foreach (var neuron in _neurons)
		{
			neuron.UpdateWeight(param);
		}
	}
}
