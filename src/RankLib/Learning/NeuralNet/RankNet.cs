using System.Globalization;
using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning.NeuralNet;

/// <summary>
/// Parameters for <see cref="RankNet"/>
/// </summary>
public class RankNetParameters : IRankerParameters
{
	/// <summary>
	/// Number of iterations (epochs).
	/// </summary>
	public int IterationCount { get; set; } = 100;

	/// <summary>
	/// Number of hidden layers
	/// </summary>
	public int HiddenLayerCount { get; set; } = 1;

	/// <summary>
	/// Number of hidden nodes per layer
	/// </summary>
	public int HiddenNodePerLayerCount { get; set; } = 10;

	/// <summary>
	/// The learning rate.
	/// </summary>
	public double LearningRate { get; set; } = 0.00005;

	public void Log(ILogger logger)
	{
		logger.LogInformation($"No. of epochs: {IterationCount}");
		logger.LogInformation($"No. of hidden layers: {HiddenLayerCount}");
		logger.LogInformation($"No. of hidden nodes per layer: {HiddenNodePerLayerCount}");
		logger.LogInformation($"Learning rate: {LearningRate}");
	}
}

/// <summary>
/// RankNet is a neural network-based ranking algorithm that learns to rank items
/// by minimizing a pairwise loss function, optimizing the ordering of items
/// based on their predicted relevance.
/// </summary>
public class RankNet : Ranker<RankNetParameters>
{
	internal const string RankerName = "RankNet";
	private readonly ILogger<RankNet> _logger;

	protected readonly List<Layer> _layers = new();
	private Layer _inputLayer;
	protected Layer OutputLayer;

	protected List<List<double>> _bestModelOnValidation = new();
	protected int _totalPairs;
	protected int _misorderedPairs;
	protected double _error;
	protected double _lastError = double.MaxValue;
	protected int _straightLoss = 0;

	public override string Name => RankerName;

	public RankNet(ILogger<RankNet>? logger = null) =>
		_logger = logger ?? NullLogger<RankNet>.Instance;

	public RankNet(List<RankList> samples, int[] features, MetricScorer scorer, ILogger<RankNet>? logger = null)
		: base(samples, features, scorer) =>
		_logger = logger ?? NullLogger<RankNet>.Instance;

	protected void SetInputOutput(int inputCount, int outputCount)
	{
		_inputLayer = new Layer(inputCount + 1, Parameters.LearningRate);
		OutputLayer = new Layer(outputCount, Parameters.LearningRate);
		_layers.Clear();
		_layers.Add(_inputLayer);
		_layers.Add(OutputLayer);
	}

	protected void SetInputOutput(int inputCount, int outputCount, NeuronType nType)
	{
		_inputLayer = new Layer(inputCount + 1, nType, Parameters.LearningRate);
		OutputLayer = new Layer(outputCount, nType, Parameters.LearningRate);
		_layers.Clear();
		_layers.Add(_inputLayer);
		_layers.Add(OutputLayer);
	}

	protected void AddHiddenLayer(int size) => _layers.Insert(_layers.Count - 1, new Layer(size, Parameters.LearningRate));

	protected void Wire()
	{
		for (var i = 0; i < _inputLayer.Count - 1; i++)
		{
			for (var j = 0; j < _layers[1].Count; j++)
				Connect(0, i, 1, j);
		}

		for (var i = 1; i < _layers.Count - 1; i++)
		{
			for (var j = 0; j < _layers[i].Count; j++)
			{
				for (var k = 0; k < _layers[i + 1].Count; k++)
					Connect(i, j, i + 1, k);
			}
		}

		for (var i = 1; i < _layers.Count; i++)
		{
			for (var j = 0; j < _layers[i].Count; j++)
				Connect(0, _inputLayer.Count - 1, i, j);
		}
	}

	protected void Connect(int sourceLayer, int sourceNeuron, int targetLayer, int targetNeuron) =>
		_ = new Synapse(_layers[sourceLayer][sourceNeuron], _layers[targetLayer][targetNeuron]);

	protected void AddInput(DataPoint p)
	{
		for (var k = 0; k < _inputLayer.Count - 1; k++)
			_inputLayer[k].AddOutput(p.GetFeatureValue(Features[k]));

		var k1 = _inputLayer.Count - 1;
		_inputLayer[k1].AddOutput(1.0);
	}

	protected void Propagate(int i)
	{
		for (var k = 1; k < _layers.Count; k++)
			_layers[k].ComputeOutput(i);
	}

	protected virtual int[][] BatchFeedForward(RankList rankList)
	{
		var pairMap = new int[rankList.Count][];
		for (var i = 0; i < rankList.Count; i++)
		{
			AddInput(rankList[i]);
			Propagate(i);

			var count = 0;
			for (var j = 0; j < rankList.Count; j++)
			{
				if (rankList[i].Label > rankList[j].Label)
					count++;
			}

			pairMap[i] = new int[count];
			var k = 0;
			for (var j = 0; j < rankList.Count; j++)
			{
				if (rankList[i].Label > rankList[j].Label)
					pairMap[i][k++] = j;
			}
		}
		return pairMap;
	}

	protected virtual void BatchBackPropagate(int[][] pairMap, float[][] pairWeight)
	{
		for (var i = 0; i < pairMap.Length; i++)
		{
			var p = new PropParameter(i, pairMap);
			OutputLayer.ComputeDelta(p);
			for (var j = _layers.Count - 2; j >= 1; j--)
				_layers[j].UpdateDelta(p);

			OutputLayer.UpdateWeight(p);
			for (var j = _layers.Count - 2; j >= 1; j--)
				_layers[j].UpdateWeight(p);
		}
	}

	protected void ClearNeuronOutputs()
	{
		foreach (var layer in _layers)
			layer.ClearOutputs();
	}

	protected virtual float[][]? ComputePairWeight(int[][] pairMap, RankList rl) => null;

	protected virtual RankList InternalReorder(RankList rl) => rl;

	/**
     * Model validation
     */
	protected void SaveBestModelOnValidation()
	{
		for (var i = 0; i < _layers.Count - 1; i++)//loop through all layers
		{
			var l = _bestModelOnValidation[i];
			l.Clear();
			for (var j = 0; j < _layers[i].Count; j++)//loop through all neurons on in the current layer
			{
				var n = _layers[i][j];
				for (var k = 0; k < n.OutLinks.Count; k++)
					l.Add(n.OutLinks[k].Weight);
			}
		}
	}

	protected void RestoreBestModelOnValidation()
	{
		try
		{
			for (var i = 0; i < _layers.Count - 1; i++)//loop through all layers
			{
				var l = _bestModelOnValidation[i];
				var c = 0;
				for (var j = 0; j < _layers[i].Count; j++)//loop through all neurons on in the current layer
				{
					var n = _layers[i][j];
					for (var k = 0; k < n.OutLinks.Count; k++)
						n.OutLinks[k].Weight = l[c++];
				}
			}
		}
		catch (Exception ex)
		{
			throw RankLibException.Create(ex);
		}
	}

	protected virtual void EstimateLoss()
	{
		_misorderedPairs = 0;
		_error = 0.0;

		foreach (var rl in Samples)
		{
			for (var k = 0; k < rl.Count - 1; k++)
			{
				var o1 = Eval(rl[k]);
				for (var l = k + 1; l < rl.Count; l++)
				{
					if (rl[k].Label > rl[l].Label)
					{
						var o2 = Eval(rl[l]);
						_error += CrossEntropy(o1, o2, 1.0);
						if (o1 < o2)
							_misorderedPairs++;
					}
				}
			}
		}
		_error = Math.Round(_error / _totalPairs, 4);
		_lastError = _error;
	}

	public override Task InitAsync()
	{
		_logger.LogInformation("Initializing...");

		SetInputOutput(Features.Length, 1);
		for (var i = 0; i < Parameters.HiddenLayerCount; i++)
			AddHiddenLayer(Parameters.HiddenNodePerLayerCount);

		Wire();

		_totalPairs = 0;
		foreach (var rl in Samples)
		{
			var correctRanking = rl.GetCorrectRanking();
			for (var j = 0; j < correctRanking.Count - 1; j++)
			{
				for (var k = j + 1; k < correctRanking.Count; k++)
				{
					if (correctRanking[j].Label > correctRanking[k].Label)
						_totalPairs++;
				}
			}
		}

		return Task.CompletedTask;
	}

	public override Task LearnAsync()
	{
		_logger.LogInformation("Training starts...");
		_logger.PrintLog([7, 14, 9, 9],
			["#epoch", "% mis-ordered", Scorer.Name + "-T", Scorer.Name + "-V"]);
		_logger.PrintLog([7, 14, 9, 9], [" ", "  pairs", " ", " "]);

		var bufferedLogger = new BufferedLogger(_logger, new StringBuilder());

		for (var i = 1; i <= Parameters.IterationCount; i++)
		{
			for (var j = 0; j < Samples.Count; j++)
			{
				var rl = InternalReorder(Samples[j]);
				var pairMap = BatchFeedForward(rl);
				var pairWeight = ComputePairWeight(pairMap, rl)!;
				BatchBackPropagate(pairMap, pairWeight);
				ClearNeuronOutputs();
			}

			TrainingDataScore = Scorer.Score(Rank(Samples));
			EstimateLoss();

			bufferedLogger.PrintLog([7, 14],
				[i.ToString(), SimpleMath.Round((double)_misorderedPairs / _totalPairs, 4).ToString(CultureInfo.InvariantCulture)]);

			bufferedLogger.PrintLog([9], [SimpleMath.Round(TrainingDataScore, 4).ToString(CultureInfo.InvariantCulture)]);
			if (ValidationSamples != null)
			{
				var score = Scorer.Score(Rank(ValidationSamples));
				if (score > ValidationDataScore)
				{
					ValidationDataScore = score;
					SaveBestModelOnValidation();
				}

				bufferedLogger.PrintLog([9], [SimpleMath.Round(score, 4).ToString(CultureInfo.InvariantCulture)]);
			}

			bufferedLogger.FlushLog();
		}

		// Restore the best model if validation data was specified
		if (ValidationSamples != null)
			RestoreBestModelOnValidation();

		TrainingDataScore = SimpleMath.Round(Scorer.Score(Rank(Samples)), 4);
		_logger.LogInformation("Finished successfully.");
		_logger.LogInformation("{Scorer} on training data: {TrainingScore}", Scorer.Name, TrainingDataScore);

		if (ValidationSamples != null)
		{
			ValidationDataScore = Scorer.Score(Rank(ValidationSamples));
			_logger.LogInformation("{Scorer} on validation data: {ValidationScore}", Scorer.Name, SimpleMath.Round(ValidationDataScore, 4));
		}

		return Task.CompletedTask;
	}


	public override double Eval(DataPoint dataPoint)
	{
		for (var k = 0; k < _inputLayer.Count - 1; k++)
			_inputLayer[k].SetOutput(dataPoint.GetFeatureValue(Features[k]));

		var k1 = _inputLayer.Count - 1;
		_inputLayer[k1].SetOutput(1.0);

		for (var k = 1; k < _layers.Count; k++)
			_layers[k].ComputeOutput();

		return OutputLayer[0].GetOutput();
	}

	protected string GetModelLayerWeights()
	{
		var output = new StringBuilder();
		for (var i = 0; i < _layers.Count - 1; i++)
		{
			for (var j = 0; j < _layers[i].Count; j++)
			{
				var neuron = _layers[i][j];
				output.AppendLine($"{i} {j} {string.Join(" ", neuron.OutLinks.Select(link => link.Weight))}");
			}
		}
		return output.ToString();
	}

	public override string GetModel()
	{
		var output = new StringBuilder();
		output.AppendLine($"## {Name}");
		output.AppendLine($"## Epochs = {Parameters.IterationCount}");
		output.AppendLine($"## No. of features = {Features.Length}");
		output.AppendLine($"## No. of hidden layers = {_layers.Count - 2}");
		for (var i = 1; i < _layers.Count - 1; i++)
			output.AppendLine($"## Layer {i}: {_layers[i].Count} neurons");

		for (var i = 0; i < Features.Length; i++)
			output.Append($"{Features[i]}{(i == Features.Length - 1 ? "" : " ")}");

		output.AppendLine();
		output.AppendLine($"{_layers.Count - 2}");
		for (var i = 1; i < _layers.Count - 1; i++)
			output.AppendLine($"{_layers[i].Count}");

		output.AppendLine(GetModelLayerWeights());

		return output.ToString();
	}

	public override void LoadFromString(string model)
	{
		var lines = model.Split(['\n', '\r'], StringSplitOptions.RemoveEmptyEntries);
		var features = lines[0].Split(' ');
		Features = new int[features.Length];
		for (var i = 0; i < features.Length; i++)
			Features[i] = int.Parse(features[i]);

		var nhl = int.Parse(lines[1]);
		var nn = new int[nhl];
		var lineIndex = 2;
		for (; lineIndex < 2 + nhl; lineIndex++)
			nn[lineIndex - 2] = int.Parse(lines[lineIndex]);

		SetInputOutput(Features.Length, 1);
		for (var j = 0; j < nhl; j++)
			AddHiddenLayer(nn[j]);

		Wire();

		for (; lineIndex < lines.Length; lineIndex++)
		{
			var s = lines[lineIndex].Split(' ');
			var iLayer = int.Parse(s[0]);
			var iNeuron = int.Parse(s[1]);
			var neuron = _layers[iLayer][iNeuron];
			for (var k = 0; k < neuron.OutLinks.Count; k++)
				neuron.OutLinks[k].Weight = double.Parse(s[k + 2]);
		}
	}

	private static double CrossEntropy(double o1, double o2, double targetValue)
	{
		var oij = o1 - o2;
		return -targetValue * oij + SimpleMath.LogBase2(1 + Math.Exp(oij));
	}
}
