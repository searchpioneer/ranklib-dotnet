using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Learning.Tree;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning.NeuralNet;

public class RankNet : Ranker
{
	private readonly ILogger<RankNet> _logger;

	public static int NIteration { get; set; } = 100;
	public static int NHiddenLayer { get; set; } = 1;
	public static int NHiddenNodePerLayer { get; set; } = 10;
	public static double LearningRate { get; set; } = 0.00005;

	protected List<Layer> _layers = new();
	protected Layer _inputLayer;
	protected Layer _outputLayer;

	protected List<List<double>> _bestModelOnValidation = new();
	protected int _totalPairs;
	protected int _misorderedPairs;
	protected double _error;
	protected double _lastError = double.MaxValue;
	protected int _straightLoss = 0;

	public RankNet(ILogger<RankNet>? logger = null) =>
		_logger = logger ?? NullLogger<RankNet>.Instance;

	public RankNet(List<RankList> samples, int[] features, MetricScorer scorer, ILogger<RankNet>? logger = null)
		: base(samples, features, scorer, logger) =>
		_logger = logger ?? NullLogger<RankNet>.Instance;

	protected void SetInputOutput(int nInput, int nOutput)
	{
		_inputLayer = new Layer(nInput + 1);
		_outputLayer = new Layer(nOutput);
		_layers.Clear();
		_layers.Add(_inputLayer);
		_layers.Add(_outputLayer);
	}

	protected void SetInputOutput(int nInput, int nOutput, int nType)
	{
		_inputLayer = new Layer(nInput + 1, nType);
		_outputLayer = new Layer(nOutput, nType);
		_layers.Clear();
		_layers.Add(_inputLayer);
		_layers.Add(_outputLayer);
	}

	protected void AddHiddenLayer(int size) => _layers.Insert(_layers.Count - 1, new Layer(size));

	protected void Wire()
	{
		for (var i = 0; i < _inputLayer.Count - 1; i++)
		{
			for (var j = 0; j < _layers[1].Count; j++)
			{
				Connect(0, i, 1, j);
			}
		}

		for (var i = 1; i < _layers.Count - 1; i++)
		{
			for (var j = 0; j < _layers[i].Count; j++)
			{
				for (var k = 0; k < _layers[i + 1].Count; k++)
				{
					Connect(i, j, i + 1, k);
				}
			}
		}

		for (var i = 1; i < _layers.Count; i++)
		{
			for (var j = 0; j < _layers[i].Count; j++)
			{
				Connect(0, _inputLayer.Count - 1, i, j);
			}
		}
	}

	protected void Connect(int sourceLayer, int sourceNeuron, int targetLayer, int targetNeuron)
	{
		var tempQualifier = _layers[sourceLayer];
		new Synapse(tempQualifier[sourceNeuron], _layers[targetLayer][targetNeuron]);
	}

	protected void AddInput(DataPoint p)
	{
		for (var k = 0; k < _inputLayer.Count - 1; k++)
		{
			_inputLayer[k].AddOutput(p.GetFeatureValue(Features[k]));
		}

		var k1 = _inputLayer.Count - 1;
		_inputLayer[k1].AddOutput(1.0);
	}

	protected void Propagate(int i)
	{
		for (var k = 1; k < _layers.Count; k++)
		{
			_layers[k].ComputeOutput(i);
		}
	}

	protected virtual int[][] BatchFeedForward(RankList rl)
	{
		var pairMap = new int[rl.Count][];
		for (var i = 0; i < rl.Count; i++)
		{
			AddInput(rl[i]);
			Propagate(i);

			var count = 0;
			for (var j = 0; j < rl.Count; j++)
			{
				if (rl[i].Label > rl[j].Label)
				{
					count++;
				}
			}

			pairMap[i] = new int[count];
			var k = 0;
			for (var j = 0; j < rl.Count; j++)
			{
				if (rl[i].Label > rl[j].Label)
				{
					pairMap[i][k++] = j;
				}
			}
		}
		return pairMap;
	}

	protected virtual void BatchBackPropagate(int[][] pairMap, float[][] pairWeight)
	{
		for (var i = 0; i < pairMap.Length; i++)
		{
			var p = new PropParameter(i, pairMap);
			_outputLayer.ComputeDelta(p);
			for (var j = _layers.Count - 2; j >= 1; j--)
			{
				_layers[j].UpdateDelta(p);
			}

			_outputLayer.UpdateWeight(p);
			for (var j = _layers.Count - 2; j >= 1; j--)
			{
				_layers[j].UpdateWeight(p);
			}
		}
	}

	protected void ClearNeuronOutputs()
	{
		foreach (var layer in _layers)
		{
			layer.ClearOutputs();
		}
	}

	// ----

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
				var tempQualifier = _layers[i];
				var n = tempQualifier[j];
				for (var k = 0; k < n.OutLinks.Count; k++)
				{
					l.Add(n.OutLinks[k].Weight);
				}
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
					var tempQualifier = _layers[i];
					var n = tempQualifier[j];
					for (var k = 0; k < n.OutLinks.Count; k++)
					{
						n.OutLinks[k].Weight = l[c++];
					}
				}
			}
		}
		catch (Exception ex)
		{
			throw RankLibError.Create("Error in NeuralNetwork.restoreBestModelOnValidation(): ", ex);
		}
	}

	protected double CrossEntropy(double o1, double o2, double targetValue)
	{
		var oij = o1 - o2;
		return -targetValue * oij + SimpleMath.LogBase2(1 + Math.Exp(oij));
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

	public override void Init()
	{
		_logger.LogInformation("Initializing...");

		SetInputOutput(Features.Length, 1);
		for (var i = 0; i < NHiddenLayer; i++)
		{
			AddHiddenLayer(NHiddenNodePerLayer);
		}
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
					{
						_totalPairs++;
					}
				}
			}
		}

		Neuron.LearningRate = LearningRate;
	}

	public override void Learn()
	{
		_logger.LogInformation("Training starts...");
		PrintLogLn(new[] { 7, 14, 9, 9 },
			new[] { "#epoch", "% mis-ordered", Scorer.Name + "-T", Scorer.Name + "-V" });
		PrintLogLn(new[] { 7, 14, 9, 9 }, new[] { " ", "  pairs", " ", " " });

		for (var i = 1; i <= NIteration; i++)
		{
			for (var j = 0; j < Samples.Count; j++)
			{
				var rl = InternalReorder(Samples[j]);
				var pairMap = BatchFeedForward(rl);
				var pairWeight = ComputePairWeight(pairMap, rl)!;
				BatchBackPropagate(pairMap, pairWeight);
				ClearNeuronOutputs();
			}

			ScoreOnTrainingData = Scorer.Score(Rank(Samples));
			EstimateLoss();

			PrintLog(new[] { 7, 14 },
				new[] { i.ToString(), SimpleMath.Round((double)_misorderedPairs / _totalPairs, 4).ToString() });

			if (i % 1 == 0)
			{
				PrintLog(new[] { 9 }, new[] { SimpleMath.Round(ScoreOnTrainingData, 4).ToString() });

				if (ValidationSamples != null)
				{
					var score = Scorer.Score(Rank(ValidationSamples));
					if (score > BestScoreOnValidationData)
					{
						BestScoreOnValidationData = score;
						SaveBestModelOnValidation();
					}

					PrintLog(new[] { 9 }, new[] { SimpleMath.Round(score, 4).ToString() });
				}
			}

			FlushLog();
		}

		// Restore the best model if validation data was specified
		if (ValidationSamples != null)
		{
			RestoreBestModelOnValidation();
		}

		ScoreOnTrainingData = SimpleMath.Round(Scorer.Score(Rank(Samples)), 4);
		_logger.LogInformation("Finished successfully.");
		_logger.LogInformation($"{Scorer.Name} on training data: {ScoreOnTrainingData}");

		if (ValidationSamples != null)
		{
			BestScoreOnValidationData = Scorer.Score(Rank(ValidationSamples));
			_logger.LogInformation($"{Scorer.Name} on validation data: {SimpleMath.Round(BestScoreOnValidationData, 4)}");
		}
	}


	public override double Eval(DataPoint p)
	{
		for (var k = 0; k < _inputLayer.Count - 1; k++)
		{
			_inputLayer[k].SetOutput(p.GetFeatureValue(Features[k]));
		}

		var k1 = _inputLayer.Count - 1;
		_inputLayer[k1].SetOutput(1.0);

		for (var k = 1; k < _layers.Count; k++)
		{
			_layers[k].ComputeOutput();
		}

		return _outputLayer[0].GetOutput();
	}

	public override Ranker CreateNew() => new RankNet();

	public override string ToString()
	{
		var output = new System.Text.StringBuilder();
		for (var i = 0; i < _layers.Count - 1; i++)
		{
			for (var j = 0; j < _layers[i].Count; j++)
			{
				var tempQualifier = _layers[i];
				var neuron = tempQualifier[j];
				output.AppendLine($"{i} {j} {string.Join(" ", neuron.OutLinks.Select(link => link.Weight))}");
			}
		}
		return output.ToString();
	}

	public override string Model()
	{
		var output = new System.Text.StringBuilder();
		output.AppendLine($"## {Name}");
		output.AppendLine($"## Epochs = {NIteration}");
		output.AppendLine($"## No. of features = {Features.Length}");
		output.AppendLine($"## No. of hidden layers = {_layers.Count - 2}");
		for (var i = 1; i < _layers.Count - 1; i++)
		{
			output.AppendLine($"## Layer {i}: {_layers[i].Count} neurons");
		}

		for (var i = 0; i < Features.Length; i++)
		{
			output.Append($"{Features[i]}{(i == Features.Length - 1 ? "" : " ")}");
		}
		output.AppendLine();
		output.AppendLine($"{_layers.Count - 2}");
		for (var i = 1; i < _layers.Count - 1; i++)
		{
			output.AppendLine($"{_layers[i].Count}");
		}
		output.AppendLine(ToString());

		return output.ToString();
	}

	public override void LoadFromString(string fullText)
	{
		var lines = fullText.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

		var features = lines[0].Split(' ');
		Features = new int[features.Length];
		for (var i = 0; i < features.Length; i++)
		{
			Features[i] = int.Parse(features[i]);
		}

		var nhl = int.Parse(lines[1]);
		var nn = new int[nhl];
		var lineIndex = 2;
		for (; lineIndex < 2 + nhl; lineIndex++)
		{
			nn[lineIndex - 2] = int.Parse(lines[lineIndex]);
		}

		SetInputOutput(Features.Length, 1);
		for (var j = 0; j < nhl; j++)
		{
			AddHiddenLayer(nn[j]);
		}
		Wire();

		for (; lineIndex < lines.Length; lineIndex++)
		{
			var s = lines[lineIndex].Split(' ');
			var iLayer = int.Parse(s[0]);
			var iNeuron = int.Parse(s[1]);
			var tempQualifier = _layers[iLayer];
			var neuron = tempQualifier[iNeuron];
			for (var k = 0; k < neuron.OutLinks.Count; k++)
			{
				neuron.OutLinks[k].Weight = double.Parse(s[k + 2]);
			}
		}
	}

	public override void PrintParameters()
	{
		_logger.LogInformation($"No. of epochs: {NIteration}");
		_logger.LogInformation($"No. of hidden layers: {NHiddenLayer}");
		_logger.LogInformation($"No. of hidden nodes per layer: {NHiddenNodePerLayer}");
		_logger.LogInformation($"Learning rate: {LearningRate}");
	}

	public override string Name => "RankNet";
}
