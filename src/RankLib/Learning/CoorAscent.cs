using System.Globalization;
using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Features;
using RankLib.Learning.Tree;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning;

/// <summary>
/// Parameters for <see cref="CoorAscent"/>
/// </summary>
public class CoorAscentParameters : IRankerParameters
{
	public int nRestart { get; set; } = 5;
	public int nMaxIteration { get; set; } = 25;
	public double stepBase { get; set; } = 0.05;
	public double stepScale { get; set; } = 2.0;
	public double tolerance { get; set; } = 0.001;
	public bool regularized { get; set; }
	public double slack { get; set; } = 0.001;

	public void Log(ILogger logger)
	{
		logger.LogInformation("No. of random restarts: {Restart}", nRestart);
		logger.LogInformation("No. of iterations to search in each direction: {NMaxIteration}", nMaxIteration);
		logger.LogInformation("Tolerance: {Tolerance}", tolerance);
		if (regularized)
			logger.LogInformation("Reg. param: {Slack}", slack);
		else
			logger.LogInformation("Regularization: No");
	}
}

/// <summary>
/// Coordinate Ascent is an optimization algorithm for ranking tasks
/// that iteratively improves a ranking model by optimizing one parameter
/// at a time while keeping the others fixed, often used to maximize
/// metrics like NDCG or MAP.
/// </summary>
/// <remarks>
/// <a href="https://link.springer.com/content/pdf/10.1007/s10791-006-9019-z.pdf">
/// D. Metzler and W.B. Croft. Linear feature-based models for information retrieval. Information Retrieval, 10(3): 257-274, 2007.
/// </a>
/// </remarks>
public class CoorAscent : Ranker<CoorAscentParameters>
{
	internal const string RankerName = "Coordinate Ascent";

	private readonly ILogger<CoorAscent> _logger;

	private int _currentFeature = -1;
	private double _weightChange = -1.0;

	public double[] Weight { get; private set; } = [];

	public override string Name => RankerName;

	public CoorAscent(ILogger<CoorAscent>? logger = null) => _logger = logger ?? NullLogger<CoorAscent>.Instance;

	public CoorAscent(List<RankList> samples, int[] features, MetricScorer scorer, ILogger<CoorAscent>? logger = null)
		: base(samples, features, scorer, logger) =>
		_logger = logger ?? NullLogger<CoorAscent>.Instance;

	public override Task InitAsync()
	{
		_logger.LogInformation("Initializing...");
		Weight = new double[Features.Length];
		Array.Fill(Weight, 1.0d / Features.Length);
		return Task.CompletedTask;
	}

	public override Task LearnAsync()
	{
		var regVector = new double[Weight.Length];
		Array.Copy(Weight, regVector, Weight.Length); // Uniform weight distribution

		double[]? bestModel = null;
		var bestModelScore = 0.0;
		int[] sign = [1, -1, 0];

		_logger.LogInformation("Training starts...");

		for (var r = 0; r < Parameters.nRestart; r++)
		{
			_logger.LogInformation($"[+] Random restart #{r + 1}/{Parameters.nRestart}...");
			var consecutiveFails = 0;

			for (var i = 0; i < Weight.Length; i++)
				Weight[i] = 1.0f / Features.Length;

			_currentFeature = -1;
			var startScore = Scorer.Score(Rank(Samples));

			var bestScore = startScore;
			var bestWeight = new double[Weight.Length];
			Array.Copy(Weight, bestWeight, Weight.Length);

			while ((Weight.Length > 1 && consecutiveFails < Weight.Length - 1) || (Weight.Length == 1 && consecutiveFails == 0))
			{
				_logger.LogInformation("Shuffling features' order...");
				_logger.LogInformation("Optimizing weight vector... ");
				PrintLogLn([7, 8, 7], ["Feature", "weight", Scorer.Name]);

				var shuffledFeatures = GetShuffledFeatures();

				for (var i = 0; i < shuffledFeatures.Length; i++)
				{
					_currentFeature = shuffledFeatures[i];
					var origWeight = Weight[shuffledFeatures[i]];
					double bestTotalStep = 0;
					var succeeds = false;

					for (var s = 0; s < sign.Length; s++)
					{
						var dir = sign[s];
						var step = 0.001 * dir;
						if (origWeight != 0.0 && Math.Abs(step) > 0.5 * Math.Abs(origWeight))
						{
							step = Parameters.stepBase * Math.Abs(origWeight);
						}

						var totalStep = step;
						var numIter = dir == 0 ? 1 : Parameters.nMaxIteration;

						for (var j = 0; j < numIter; j++)
						{
							var newWeight = origWeight + totalStep;
							_weightChange = step;
							Weight[shuffledFeatures[i]] = newWeight;

							var score = Scorer.Score(Rank(Samples));
							if (Parameters.regularized)
							{
								var penalty = Parameters.slack * GetDistance(Weight, regVector);
								score -= penalty;
							}

							if (score > bestScore)
							{
								bestScore = score;
								bestTotalStep = totalStep;
								succeeds = true;
								var bw = Weight[shuffledFeatures[i]] > 0 ? "+" : "";
								PrintLogLn([7, 8, 7], [Features[shuffledFeatures[i]].ToString(),
									$"{bw}{Math.Round(Weight[shuffledFeatures[i]], 4)}",
									Math.Round(bestScore, 4).ToString(CultureInfo.InvariantCulture)
								]);
							}

							if (j < Parameters.nMaxIteration - 1)
							{
								step *= Parameters.stepScale;
								totalStep += step;
							}
						}

						if (succeeds)
							break;

						if (s < sign.Length - 1)
						{
							_weightChange = -totalStep;
							UpdateCached();
							Weight[shuffledFeatures[i]] = origWeight;
						}
					}

					if (succeeds)
					{
						_weightChange = bestTotalStep - Weight[shuffledFeatures[i]];
						UpdateCached();
						Weight[shuffledFeatures[i]] = origWeight + bestTotalStep;
						consecutiveFails = 0;

						var sum = Normalize(Weight);
						ScaleCached(sum);
						Array.Copy(Weight, bestWeight, Weight.Length);
					}
					else
					{
						consecutiveFails++;
						_weightChange = -Weight[shuffledFeatures[i]];
						UpdateCached();
						Weight[shuffledFeatures[i]] = origWeight;
					}
				}

				if (bestScore - startScore < Parameters.tolerance)
					break;
			}

			if (ValidationSamples != null)
			{
				_currentFeature = -1;
				bestScore = Scorer.Score(Rank(ValidationSamples));
			}

			if (bestModel == null || bestScore > bestModelScore)
			{
				bestModelScore = bestScore;
				bestModel = new double[bestWeight.Length];
				Array.Copy(bestWeight, bestModel, bestWeight.Length);
			}
		}

		Array.Copy(bestModel!, Weight, bestModel!.Length);
		_currentFeature = -1;
		ScoreOnTrainingData = Math.Round(Scorer.Score(Rank(Samples)), 4);
		_logger.LogInformation("Finished successfully.");
		_logger.LogInformation($"{Scorer.Name} on training data: {ScoreOnTrainingData}");

		if (ValidationSamples != null)
		{
			BestScoreOnValidationData = Scorer.Score(Rank(ValidationSamples));
			_logger.LogInformation($"{Scorer.Name} on validation data: {Math.Round(BestScoreOnValidationData, 4)}");
		}

		return Task.CompletedTask;
	}

	public override RankList Rank(RankList rankList)
	{
		var score = new double[rankList.Count];
		if (_currentFeature == -1)
		{
			for (var i = 0; i < rankList.Count; i++)
			{
				for (var j = 0; j < Features.Length; j++)
				{
					score[i] += Weight[j] * rankList[i].GetFeatureValue(Features[j]);
				}
				rankList[i].Cached = score[i];
			}
		}
		else
		{
			for (var i = 0; i < rankList.Count; i++)
			{
				score[i] = rankList[i].Cached + _weightChange * rankList[i].GetFeatureValue(Features[_currentFeature]);
				rankList[i].Cached = score[i];
			}
		}

		var idx = MergeSorter.Sort(score, false);
		return new RankList(rankList, idx);
	}

	public override double Eval(DataPoint dataPoint)
	{
		var score = 0.0;
		for (var i = 0; i < Features.Length; i++)
		{
			score += Weight[i] * dataPoint.GetFeatureValue(Features[i]);
		}
		return score;
	}

	public override string ToString()
	{
		var output = new StringBuilder();
		for (var i = 0; i < Weight.Length; i++)
			output.Append($"{Features[i]}:{Weight[i]}{(i == Weight.Length - 1 ? "" : " ")}");

		return output.ToString();
	}

	public override string Model
	{
		get
		{
			var output = new StringBuilder();
			output.AppendLine($"## {Name}");
			output.AppendLine($"## Restart = {Parameters.nRestart}");
			output.AppendLine($"## MaxIteration = {Parameters.nMaxIteration}");
			output.AppendLine($"## StepBase = {Parameters.stepBase.ToRankLibString()}");
			output.AppendLine($"## StepScale = {Parameters.stepScale.ToRankLibString()}");
			output.AppendLine($"## Tolerance = {Parameters.tolerance.ToRankLibString()}");
			output.AppendLine($"## Regularized = {(Parameters.regularized ? "true" : "false")}");
			output.AppendLine($"## Slack = {Parameters.slack.ToRankLibString()}");
			output.AppendLine(ToString());
			return output.ToString();
		}
	}

	public override void LoadFromString(string model)
	{
		using var reader = new StringReader(model);
		while (reader.ReadLine() is { } line)
		{
			if (line.StartsWith("##"))
				continue;

			var kvp = new KeyValuePairs(line);
			Weight = new double[kvp.Count];
			Features = new int[kvp.Count];

			for (var i = 0; i < kvp.Count; i++)
			{
				var kv = kvp[i];
				Features[i] = int.Parse(kv.Key);
				Weight[i] = double.Parse(kv.Value);
			}
			break;
		}
	}

	// Private helper methods
	private void UpdateCached()
	{
		for (var j = 0; j < Samples.Count; j++)
		{
			var rl = Samples[j];
			for (var i = 0; i < rl.Count; i++)
			{
				var score = rl[i].Cached + _weightChange * rl[i].GetFeatureValue(Features[_currentFeature]);
				rl[i].Cached = score;
			}
		}
	}

	private void ScaleCached(double sum)
	{
		for (var j = 0; j < Samples.Count; j++)
		{
			var rl = Samples[j];
			for (var i = 0; i < rl.Count; i++)
				rl[i].Cached = (rl[i].Cached / sum);
		}
	}

	private int[] GetShuffledFeatures()
	{
		var fids = new int[Features.Length];
		var l = new List<int>(Features.Length);
		for (var i = 0; i < Features.Length; i++)
			l.Add(i);

		l.Shuffle();
		for (var i = 0; i < l.Count; i++)
			fids[i] = l[i];

		return fids;

	}

	private double GetDistance(double[] w1, double[] w2)
	{
		var s1 = 0.0;
		var s2 = 0.0;
		for (var i = 0; i < w1.Length; i++)
		{
			s1 += Math.Abs(w1[i]);
			s2 += Math.Abs(w2[i]);
		}
		var dist = 0.0;
		for (var i = 0; i < w1.Length; i++)
		{
			var t = w1[i] / s1 - w2[i] / s2;
			dist += t * t;
		}
		return Math.Sqrt(dist);
	}

	private double Normalize(double[] weights)
	{
		var sum = weights.Sum(Math.Abs);
		if (sum > 0)
		{
			for (var i = 0; i < weights.Length; i++)
				weights[i] /= sum;
		}
		else
		{
			sum = 1;
			for (var i = 0; i < weights.Length; i++)
				weights[i] = 1.0 / weights.Length;
		}
		return sum;
	}

	public void CopyModel(CoorAscent ranker)
	{
		Weight = new double[Features.Length];
		if (ranker.Weight.Length != Weight.Length)
		{
			throw RankLibException.Create("These two models use different feature set!!");
		}
		Array.Copy(ranker.Weight, Weight, ranker.Weight.Length);
		_logger.LogInformation("Model loaded.");
	}

	public double Distance(CoorAscent ca) => GetDistance(Weight, ca.Weight);
}
