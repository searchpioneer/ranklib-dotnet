namespace RankLib.Learning.NeuralNet;

/// <summary>
/// A layer of neurons in a neural network.
/// </summary>
public class Layer
{
	private readonly List<Neuron> _neurons;

	/// <summary>
	/// Initializes a new instance of <see cref="Layer"/>.
	/// </summary>
	/// <param name="count">The number of neurons to add at this layer.</param>
	/// <param name="learningRate">The learning rate</param>
	public Layer(int count, double learningRate)
	{
		_neurons = new List<Neuron>(count);
		for (var i = 0; i < count; i++)
			_neurons.Add(new Neuron(learningRate));
	}

	/// <summary>
	/// Initializes a new instance of <see cref="Layer"/>.
	/// </summary>
	/// <param name="count">The number of neurons to add at this layer.</param>
	/// <param name="neuronType">The type of neurons to add</param>
	/// <param name="learningRate">The learning rate</param>
	public Layer(int count, NeuronType neuronType, double learningRate)
	{
		_neurons = new List<Neuron>(count);
		for (var i = 0; i < count; i++)
		{
			_neurons.Add(neuronType == NeuronType.Single
				? new Neuron(learningRate)
				: new ListNeuron(learningRate));
		}
	}

	public Neuron this[int i] => _neurons[i];

	public int Count => _neurons.Count;

	public void ComputeOutput(int i)
	{
		foreach (var neuron in _neurons)
			neuron.ComputeOutput(i);
	}

	public void ComputeOutput()
	{
		foreach (var neuron in _neurons)
			neuron.ComputeOutput();
	}

	public void ClearOutputs()
	{
		foreach (var neuron in _neurons)
			neuron.ClearOutputs();
	}

	public void ComputeDelta(PropParameter param)
	{
		foreach (var neuron in _neurons)
			neuron.ComputeDelta(param);
	}

	public void UpdateDelta(PropParameter param)
	{
		foreach (var neuron in _neurons)
			neuron.UpdateDelta(param);
	}

	public void UpdateWeight(PropParameter param)
	{
		foreach (var neuron in _neurons)
			neuron.UpdateWeight(param);
	}
}
