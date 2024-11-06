using System.Globalization;
using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning.NeuralNet;

public class ListNetParameters : RankNetParameters
{
	public ListNetParameters()
	{
		// FIXED, it doesn't work with hidden layer
		HiddenLayerCount = 0;
		IterationCount = 1500;
		LearningRate = 0.00001;
	}
}

public class ListNet : RankNet
{
	internal new const string RankerName = "ListNet";

	private readonly ILogger<ListNet> _logger;

	public override string Name => RankerName;

	public ListNet(ILogger<ListNet>? logger = null) : base(logger) =>
		_logger = logger ?? NullLogger<ListNet>.Instance;

	public ListNet(List<RankList> samples, int[] features, MetricScorer scorer, ILogger<ListNet>? logger = null)
		: base(samples, features, scorer) =>
		_logger = logger ?? NullLogger<ListNet>.Instance;

	private float[] FeedForward(RankList rl)
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

	private void BackPropagate(float[] labels)
	{
		// Back-propagate
		var p = new PropParameter(labels);
		OutputLayer.ComputeDelta(p); // Starting at the output layer

		// Weight update
		OutputLayer.UpdateWeight(p);
	}

	protected override void EstimateLoss()
	{
		_error = 0.0;
		double sumLabelExp = 0;
		double sumScoreExp = 0;
		for (var i = 0; i < Samples.Count; i++)
		{
			var rl = Samples[i];
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

	public override Task InitAsync()
	{
		_logger.LogInformation("Initializing...");

		// Set up the network
		SetInputOutput(Features.Length, 1, NeuronType.List);
		Wire();

		if (ValidationSamples != null)
		{
			for (var i = 0; i < _layers.Count; i++)
				_bestModelOnValidation.Add(new List<double>());
		}

		return Task.CompletedTask;
	}

	public override Task LearnAsync()
	{
		_logger.LogInformation("Training starts...");
		PrintLogLn([7, 14, 9, 9], ["#epoch", "C.E. Loss", Scorer.Name + "-T", Scorer.Name + "-V"]);

		for (var i = 1; i <= Parameters.IterationCount; i++)
		{
			for (var j = 0; j < Samples.Count; j++)
			{
				var labels = FeedForward(Samples[j]);
				BackPropagate(labels);
				ClearNeuronOutputs();
			}

			PrintLog([7, 14], [i.ToString(), SimpleMath.Round(_error, 6).ToString(CultureInfo.InvariantCulture)]);

			if (i % 1 == 0)
			{
				ScoreOnTrainingData = Scorer.Score(Rank(Samples));
				PrintLog([9], [SimpleMath.Round(ScoreOnTrainingData, 4).ToString(CultureInfo.InvariantCulture)]);

				if (ValidationSamples != null)
				{
					var score = Scorer.Score(Rank(ValidationSamples));
					if (score > BestScoreOnValidationData)
					{
						BestScoreOnValidationData = score;
						SaveBestModelOnValidation();
					}
					PrintLog([9], [SimpleMath.Round(score, 4).ToString(CultureInfo.InvariantCulture)]);
				}
			}
			FlushLog();
		}

		// Restore the best model if validation data is used
		if (ValidationSamples != null)
			RestoreBestModelOnValidation();

		ScoreOnTrainingData = SimpleMath.Round(Scorer.Score(Rank(Samples)), 4);
		_logger.LogInformation("Finished successfully.");
		_logger.LogInformation($"{Scorer.Name} on training data: {ScoreOnTrainingData}");

		if (ValidationSamples != null)
		{
			BestScoreOnValidationData = Scorer.Score(Rank(ValidationSamples));
			_logger.LogInformation($"{Scorer.Name} on validation data: {SimpleMath.Round(BestScoreOnValidationData, 4)}");
		}

		return Task.CompletedTask;
	}

	public override string Model
	{
		get
		{
			var output = new StringBuilder();
			output.Append($"## {Name}\n");
			output.Append($"## Epochs = {Parameters.IterationCount}\n");
			output.Append($"## No. of features = {Features.Length}\n");

			// Print used features
			for (var i = 0; i < Features.Length; i++)
				output.Append(Features[i] + (i == Features.Length - 1 ? "" : " "));

			output.Append('\n');

			// Print network information
			output.Append("0\n"); // [# hidden layers, *ALWAYS* 0 since we're using linear net]
								  // Print learned weights
			output.Append(ToString());
			return output.ToString();
		}
	}

	public override void LoadFromString(string model)
	{
		try
		{
			using var reader = new StringReader(model);
			var l = new List<string>();
			while (reader.ReadLine() is { } content)
			{
				content = content.Trim();
				if (string.IsNullOrEmpty(content) || content.StartsWith("##"))
					continue;

				l.Add(content);
			}

			// Load the network
			// The first line contains feature information
			var tmp = l[0].Split(' ');
			Features = new int[tmp.Length];
			for (var i = 0; i < tmp.Length; i++)
				Features[i] = int.Parse(tmp[i]);

			// The 2nd line is a scalar indicating the number of hidden layers
			var nHiddenLayer = int.Parse(l[1]);
			var nn = new int[nHiddenLayer];

			// The next @nHiddenLayer lines contain the number of neurons in each layer
			var index = 2;
			for (; index < 2 + nHiddenLayer; index++)
				nn[index - 2] = int.Parse(l[index]);

			// Create the network
			SetInputOutput(Features.Length, 1);
			for (var j = 0; j < nHiddenLayer; j++)
				AddHiddenLayer(nn[j]);

			Wire();

			// Fill in weights
			for (; index < l.Count; index++) // Loop through all layers
			{
				var s = l[index].Split(' ');
				var iLayer = int.Parse(s[0]); // Which layer?
				var iNeuron = int.Parse(s[1]); // Which neuron?
				var tempQualifier = _layers[iLayer];
				var n = tempQualifier[iNeuron];
				for (var k = 0; k < n.OutLinks.Count; k++)
					n.OutLinks[k].Weight = double.Parse(s[k + 2]);
			}
		}
		catch (Exception ex)
		{
			throw RankLibException.Create("Error in ListNet::LoadFromString(): ", ex);
		}
	}
}
