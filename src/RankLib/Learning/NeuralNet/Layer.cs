namespace RankLib.Learning.NeuralNet;

public class Layer
{
	protected List<Neuron> _neurons;

	public Layer(int size)
	{
		_neurons = new List<Neuron>();
		for (var i = 0; i < size; i++)
		{
			_neurons.Add(new Neuron());
		}
	}

	public Layer(int size, int nType)
	{
		_neurons = new List<Neuron>();
		for (var i = 0; i < size; i++)
		{
			if (nType == 0)
			{
				_neurons.Add(new Neuron());
			}
			else
			{
				_neurons.Add(new ListNeuron());
			}
		}
	}

	public Neuron Get(int k) => _neurons[k];

	public int Size() => _neurons.Count;

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
