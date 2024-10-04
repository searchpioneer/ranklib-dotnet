namespace RankLib.Learning.NeuralNet;

public class Neuron
{
	public static double Momentum = 0.9;
	public static double LearningRate = 0.001;

	protected ITransferFunction _tfunc = new LogiFunction();

	protected double _output;
	protected List<double> _outputs = new List<double>();
	protected double _delta_i = 0.0;
	protected double[] _deltas_j = null;

	protected List<Synapse> _inLinks = new List<Synapse>();
	protected List<Synapse> _outLinks = new List<Synapse>();

	public Neuron() => _output = 0.0;

	public double GetOutput() => _output;

	public double GetOutput(int k) => _outputs[k];

	public List<Synapse> GetInLinks() => _inLinks;

	public List<Synapse> GetOutLinks() => _outLinks;

	public void SetOutput(double output) => _output = output;

	public void AddOutput(double output) => _outputs.Add(output);

	public void ComputeOutput()
	{
		var wsum = 0.0;
		foreach (var synapse in _inLinks)
		{
			wsum += synapse.Source.GetOutput() * synapse.Weight;
		}
		_output = _tfunc.Compute(wsum);
	}

	public void ComputeOutput(int i)
	{
		var wsum = 0.0;
		foreach (var synapse in _inLinks)
		{
			wsum += synapse.Source.GetOutput(i) * synapse.Weight;
		}
		_output = _tfunc.Compute(wsum);
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
			_deltas_j[k] = lambda * _tfunc.ComputeDerivative(_outputs[j]);
		}

		_delta_i *= _tfunc.ComputeDerivative(_outputs[current]);
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

			foreach (var synapse in _outLinks)
			{
				errorSum += synapse.Target._deltas_j[k] * synapse.Weight;
				if (k == 0)
				{
					_delta_i += synapse.Target._delta_i * synapse.Weight;
				}
			}

			if (k == 0)
			{
				_delta_i *= weight * _tfunc.ComputeDerivative(_outputs[current]);
			}

			_deltas_j[k] = errorSum * weight * _tfunc.ComputeDerivative(_outputs[j]);
		}
	}

	public virtual void UpdateWeight(PropParameter param)
	{
		foreach (var synapse in _inLinks)
		{
			var sum_j = 0.0;
			for (var l = 0; l < _deltas_j.Length; l++)
			{
				sum_j += _deltas_j[l] * synapse.Source.GetOutput(param.PairMap[param.Current][l]);
			}

			var dw = LearningRate * (_delta_i * synapse.Source.GetOutput(param.Current) - sum_j);
			synapse.SetWeightAdjustment(dw);
			synapse.UpdateWeight();
		}
	}
}
