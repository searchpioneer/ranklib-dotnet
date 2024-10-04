using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning.NeuralNet;

public class ListNet : RankNet
{
	private static readonly ILogger logger = NullLogger.Instance;

	// Parameters
	public static int nIteration = 1500;
	public static double learningRate = 0.00001;
	public static int nHiddenLayer = 0; // FIXED, it doesn't work with hidden layer

	public ListNet() { }

	public ListNet(List<RankList> samples, int[] features, MetricScorer scorer)
		: base(samples, features, scorer)
	{ }

	protected float[] FeedForward(RankList rl)
	{
		var labels = new float[rl.Count];
		for (var i = 0; i < rl.Count; i++)
		{
			AddInput(rl[i]);
			Propagate(i);
			labels[i] = rl[i].Label;
		}
		return labels;
	}

	protected void BackPropagate(float[] labels)
	{
		// Back-propagate
		var p = new PropParameter(labels);
		_outputLayer.ComputeDelta(p); // Starting at the output layer

		// Weight update
		_outputLayer.UpdateWeight(p);
	}

	protected override void EstimateLoss()
	{
		_error = 0.0;
		double sumLabelExp = 0;
		double sumScoreExp = 0;
		for (var i = 0; i < _samples.Count; i++)
		{
			var rl = _samples[i];
			var scores = new double[rl.Count];
			double err = 0;
			for (var j = 0; j < rl.Count; j++)
			{
				scores[j] = Eval(rl[j]);
				sumLabelExp += Math.Exp(rl[j].Label);
				sumScoreExp += Math.Exp(scores[j]);
			}
			for (var j = 0; j < rl.Count; j++)
			{
				var p1 = Math.Exp(rl[j].Label) / sumLabelExp;
				var p2 = Math.Exp(scores[j]) / sumScoreExp;
				err += -p1 * SimpleMath.LogBase2(p2);
			}
			_error += err / rl.Count;
		}
		_lastError = _error;
	}

	public override void Init()
	{
		logger.LogInformation("Initializing...");

		// Set up the network
		SetInputOutput(_features.Length, 1, 1);
		Wire();

		if (_validationSamples != null)
		{
			for (var i = 0; i < _layers.Count; i++)
			{
				_bestModelOnValidation.Add(new List<double>());
			}
		}

		Neuron.LearningRate = learningRate;
	}

	public override void Learn()
	{
		logger.LogInformation("Training starts...");
		PrintLogLn(new[] { 7, 14, 9, 9 }, new[] { "#epoch", "C.E. Loss", _scorer.Name() + "-T", _scorer.Name() + "-V" });

		for (var i = 1; i <= nIteration; i++)
		{
			for (var j = 0; j < _samples.Count; j++)
			{
				var labels = FeedForward(_samples[j]);
				BackPropagate(labels);
				ClearNeuronOutputs();
			}

			PrintLog(new[] { 7, 14 }, new[] { i.ToString(), SimpleMath.Round(_error, 6).ToString() });

			if (i % 1 == 0)
			{
				_scoreOnTrainingData = _scorer.Score(Rank(_samples));
				PrintLog(new[] { 9 }, new[] { SimpleMath.Round(_scoreOnTrainingData, 4).ToString() });

				if (_validationSamples != null)
				{
					var score = _scorer.Score(Rank(_validationSamples));
					if (score > _bestScoreOnValidationData)
					{
						_bestScoreOnValidationData = score;
						SaveBestModelOnValidation();
					}
					PrintLog(new[] { 9 }, new[] { SimpleMath.Round(score, 4).ToString() });
				}
			}
			FlushLog();
		}

		// Restore the best model if validation data is used
		if (_validationSamples != null)
		{
			RestoreBestModelOnValidation();
		}

		_scoreOnTrainingData = SimpleMath.Round(_scorer.Score(Rank(_samples)), 4);
		logger.LogInformation("Finished successfully.");
		logger.LogInformation($"{_scorer.Name} on training data: {_scoreOnTrainingData}");

		if (_validationSamples != null)
		{
			_bestScoreOnValidationData = _scorer.Score(Rank(_validationSamples));
			logger.LogInformation($"{_scorer.Name} on validation data: {SimpleMath.Round(_bestScoreOnValidationData, 4)}");
		}
	}

	public override double Eval(DataPoint p) => base.Eval(p);

	public override Ranker CreateNew() => new ListNet();

	public override string ToString() => base.ToString();

	public override string Model()
	{
		var output = new System.Text.StringBuilder();
		output.Append($"## {Name()}\n");
		output.Append($"## Epochs = {nIteration}\n");
		output.Append($"## No. of features = {_features.Length}\n");

		// Print used features
		for (var i = 0; i < _features.Length; i++)
		{
			output.Append(_features[i] + (i == _features.Length - 1 ? "" : " "));
		}
		output.Append("\n");

		// Print network information
		output.Append("0\n"); // [# hidden layers, *ALWAYS* 0 since we're using linear net]
							  // Print learned weights
		output.Append(ToString());
		return output.ToString();
	}

	public override void LoadFromString(string fullText)
	{
		try
		{
			using (var inStream = new StringReader(fullText))
			{
				string content;
				var l = new List<string>();
				while ((content = inStream.ReadLine()) != null)
				{
					content = content.Trim();
					if (string.IsNullOrEmpty(content) || content.StartsWith("##"))
						continue;
					l.Add(content);
				}

				// Load the network
				// The first line contains feature information
				var tmp = l[0].Split(' ');
				_features = new int[tmp.Length];
				for (var i = 0; i < tmp.Length; i++)
				{
					_features[i] = int.Parse(tmp[i]);
				}

				// The 2nd line is a scalar indicating the number of hidden layers
				var nHiddenLayer = int.Parse(l[1]);
				var nn = new int[nHiddenLayer];

				// The next @nHiddenLayer lines contain the number of neurons in each layer
				var index = 2;
				for (; index < 2 + nHiddenLayer; index++)
				{
					nn[index - 2] = int.Parse(l[index]);
				}

				// Create the network
				SetInputOutput(_features.Length, 1);
				for (var j = 0; j < nHiddenLayer; j++)
				{
					AddHiddenLayer(nn[j]);
				}
				Wire();

				// Fill in weights
				for (; index < l.Count; index++) // Loop through all layers
				{
					var s = l[index].Split(' ');
					var iLayer = int.Parse(s[0]); // Which layer?
					var iNeuron = int.Parse(s[1]); // Which neuron?
					var n = _layers[iLayer].Get(iNeuron);
					for (var k = 0; k < n.GetOutLinks().Count; k++)
					{
						n.GetOutLinks()[k].Weight = double.Parse(s[k + 2]);
					}
				}
			}
		}
		catch (Exception ex)
		{
			throw RankLibError.Create("Error in ListNet::LoadFromString(): ", ex);
		}
	}

	public override void PrintParameters()
	{
		logger.LogInformation($"No. of epochs: {nIteration}");
		logger.LogInformation($"Learning rate: {learningRate}");
	}

	public override string Name() => "ListNet";
}
