namespace RankLib.Learning.NeuralNet;

public class Neuron
{
	public const double DefaultLearningRate = 0.001;

	protected readonly double LearningRate;
	protected readonly List<double> Outputs = [];

	private readonly ITransferFunction _transferFunction;
	private double _output;
	private double _deltaI;
	private double[] _deltasJ = [];

	public List<Synapse> InLinks { get; } = [];
	public List<Synapse> OutLinks { get; } = [];

	public Neuron(double learningRate, ITransferFunction? transferFunction = null)
	{
		LearningRate = learningRate;
		_transferFunction = transferFunction ?? new LogisticFunction();
		_output = 0.0;
	}

	public double GetOutput() => _output;

	public double GetOutput(int k) => Outputs[k];

	public void SetOutput(double output) => _output = output;

	public void AddOutput(double output) => Outputs.Add(output);

	public void ComputeOutput()
	{
		var wsum = 0.0;
		foreach (var synapse in InLinks)
			wsum += synapse.Source.GetOutput() * synapse.Weight;

		_output = _transferFunction.Compute(wsum);
	}

	public void ComputeOutput(int i)
	{
		var wsum = 0.0;
		foreach (var synapse in InLinks)
			wsum += synapse.Source.GetOutput(i) * synapse.Weight;

		_output = _transferFunction.Compute(wsum);
		Outputs.Add(_output);
	}

	public void ClearOutputs() => Outputs.Clear();

	public virtual void ComputeDelta(PropParameter param)
	{
		var pairMap = param.PairMap;
		var current = param.Current;

		_deltaI = 0.0;
		_deltasJ = new double[pairMap[current].Length];

		for (var k = 0; k < pairMap[current].Length; k++)
		{
			var j = pairMap[current][k];
			float weight;
			double pij;

			if (param.PairWeight is null)
			{
				weight = 1;
				pij = 1.0 / (1.0 + Math.Exp(Outputs[current] - Outputs[j]));
			}
			else
			{
				weight = param.PairWeight[current][k];
				pij = param.TargetValue[current][k] - 1.0 / (1.0 + Math.Exp(-(Outputs[current] - Outputs[j])));
			}

			var lambda = weight * pij;
			_deltaI += lambda;
			_deltasJ[k] = lambda * _transferFunction.ComputeDerivative(Outputs[j]);
		}

		_deltaI *= _transferFunction.ComputeDerivative(Outputs[current]);
	}

	public void UpdateDelta(PropParameter param)
	{
		var pairMap = param.PairMap;
		var pairWeight = param.PairWeight;
		var current = param.Current;

		_deltaI = 0;
		_deltasJ = new double[pairMap[current].Length];

		for (var k = 0; k < pairMap[current].Length; k++)
		{
			var j = pairMap[current][k];
			var weight = pairWeight != null ? pairWeight[current][k] : 1.0f;
			var errorSum = 0.0;

			foreach (var synapse in OutLinks)
			{
				errorSum += synapse.Target._deltasJ[k] * synapse.Weight;
				if (k == 0)
					_deltaI += synapse.Target._deltaI * synapse.Weight;
			}

			if (k == 0)
				_deltaI *= weight * _transferFunction.ComputeDerivative(Outputs[current]);

			_deltasJ[k] = errorSum * weight * _transferFunction.ComputeDerivative(Outputs[j]);
		}
	}

	public virtual void UpdateWeight(PropParameter param)
	{
		foreach (var synapse in InLinks)
		{
			var sumJ = 0.0;
			for (var l = 0; l < _deltasJ.Length; l++)
				sumJ += _deltasJ[l] * synapse.Source.GetOutput(param.PairMap[param.Current][l]);

			var dw = LearningRate * (_deltaI * synapse.Source.GetOutput(param.Current) - sumJ);
			synapse.UpdateWeight(dw);
		}
	}
}
