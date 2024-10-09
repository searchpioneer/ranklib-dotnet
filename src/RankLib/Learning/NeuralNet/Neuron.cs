namespace RankLib.Learning.NeuralNet;

public class Neuron
{
	public static double Momentum = 0.9;
	public static double LearningRate = 0.001;

	protected ITransferFunction _transferFunction = new LogiFunction();

	protected double _output;
	protected List<double> _outputs = new();
	protected double _delta_i = 0.0;
	protected double[] _deltas_j;

	public List<Synapse> InLinks { get; } = new();
	public List<Synapse> OutLinks { get; } = new();

	public Neuron() => _output = 0.0;

	public double GetOutput() => _output;

	public double GetOutput(int k) => _outputs[k];

	public void SetOutput(double output) => _output = output;

	public void AddOutput(double output) => _outputs.Add(output);

	public void ComputeOutput()
	{
		var wsum = 0.0;
		foreach (var synapse in InLinks)
		{
			wsum += synapse.Source.GetOutput() * synapse.Weight;
		}
		_output = _transferFunction.Compute(wsum);
	}

	public void ComputeOutput(int i)
	{
		var wsum = 0.0;
		foreach (var synapse in InLinks)
		{
			wsum += synapse.Source.GetOutput(i) * synapse.Weight;
		}
		_output = _transferFunction.Compute(wsum);
		_outputs.Add(_output);
	}

	public void ClearOutputs() => _outputs.Clear();

	public virtual void ComputeDelta(PropParameter param)
	{
		var pairMap = param.PairMap;
		var current = param.Current;

		_delta_i = 0.0;
		_deltas_j = new double[pairMap[current].Length];

		for (var k = 0; k < pairMap[current].Length; k++)
		{
			var j = pairMap[current][k];
			float weight = 1;
			double pij;

			if (param.PairWeight == null)
			{
				pij = 1.0 / (1.0 + Math.Exp(_outputs[current] - _outputs[j]));
			}
			else
			{
				weight = param.PairWeight[current][k];
				pij = param.TargetValue[current][k] - 1.0 / (1.0 + Math.Exp(_outputs[current] - _outputs[j]));
			}

			var lambda = weight * pij;
			_delta_i += lambda;
			_deltas_j[k] = lambda * _transferFunction.ComputeDerivative(_outputs[j]);
		}

		_delta_i *= _transferFunction.ComputeDerivative(_outputs[current]);
	}

	public void UpdateDelta(PropParameter param)
	{
		var pairMap = param.PairMap;
		var pairWeight = param.PairWeight;
		var current = param.Current;

		_delta_i = 0;
		_deltas_j = new double[pairMap[current].Length];

		for (var k = 0; k < pairMap[current].Length; k++)
		{
			var j = pairMap[current][k];
			var weight = pairWeight != null ? pairWeight[current][k] : 1.0f;
			var errorSum = 0.0;

			foreach (var synapse in OutLinks)
			{
				errorSum += synapse.Target._deltas_j[k] * synapse.Weight;
				if (k == 0)
				{
					_delta_i += synapse.Target._delta_i * synapse.Weight;
				}
			}

			if (k == 0)
			{
				_delta_i *= weight * _transferFunction.ComputeDerivative(_outputs[current]);
			}

			_deltas_j[k] = errorSum * weight * _transferFunction.ComputeDerivative(_outputs[j]);
		}
	}

	public virtual void UpdateWeight(PropParameter param)
	{
		foreach (var synapse in InLinks)
		{
			var sum_j = 0.0;
			for (var l = 0; l < _deltas_j.Length; l++)
			{
				sum_j += _deltas_j[l] * synapse.Source.GetOutput(param.PairMap[param.Current][l]);
			}

			var dw = LearningRate * (_delta_i * synapse.Source.GetOutput(param.Current) - sum_j);
			synapse.UpdateWeight(dw);
		}
	}
}
