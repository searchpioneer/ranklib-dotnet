using System.Globalization;
using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning;

/// <summary>
/// Parameters for <see cref="CoordinateAscent"/>
/// </summary>
public class CoordinateAscentParameters : IRankerParameters
{
	/// <summary>
	/// Number of random restarts.
	/// </summary>
	public int RandomRestartCount { get; set; } = 5;

	/// <summary>
	/// Number of iterations to search in each direction.
	/// </summary>
	public int MaximumIterationCount { get; set; } = 25;

	/// <summary>
	/// The base step size that the algorithm initially uses for adjusting weights.
	/// This small base step is scaled to vary the search around the current solution in each iteration.
	/// </summary>
	public double StepBase { get; set; } = 0.05;

	/// <summary>
	/// A scaling factor applied to <see cref="StepBase"/> to increase or decrease the step incrementally.
	/// It allows the algorithm to take progressively larger or smaller steps in each iteration,
	/// improving the precision of convergence.
	/// </summary>
	public double StepScale { get; set; } = 2.0;

	/// <summary>
	/// The threshold for the minimal improvement required between iterations to continue the optimization.
	/// When changes fall below this threshold, the algorithm stops adjusting further, assuming convergence.
	/// </summary>
	public double Tolerance { get; set; } = 0.001;

	/// <summary>
	/// Whether regularization is applied to help avoid overfitting. If <c>true</c>, <see cref="Slack"/> value
	/// is applied. If <c>false</c>, No regularization is performed.
	/// </summary>
	public bool Regularized { get; set; }

	/// <summary>
	/// The regularization parameter (if regularized is true) that controls the penalty strength.
	/// A small slack value limits the coefficient magnitude more strictly, enforcing a
	/// simpler model with smaller weights.
	/// </summary>
	public double Slack { get; set; } = 0.001;

	public void Log(ILogger logger)
	{
		logger.LogInformation("No. of random restarts: {Restart}", RandomRestartCount);
		logger.LogInformation("No. of iterations to search in each direction: {NMaxIteration}", MaximumIterationCount);
		logger.LogInformation("Tolerance: {Tolerance}", Tolerance);
		if (Regularized)
			logger.LogInformation("Reg. param: {Slack}", Slack);
		else
			logger.LogInformation("Regularization: No");
	}
}

/// <summary>
/// Coordinate Ascent is a linear ranking model for ranking tasks
/// that iteratively improves a ranking model by optimizing one parameter
/// at a time while keeping the others fixed.
/// </summary>
/// <remarks>
/// <a href="https://link.springer.com/content/pdf/10.1007/s10791-006-9019-z.pdf">
/// D. Metzler and W.B. Croft. Linear feature-based models for information retrieval.
/// Information Retrieval, 10(3): 257-274, 2007.
/// </a>
/// </remarks>
public class CoordinateAscent : Ranker<CoordinateAscentParameters>
{
	internal const string RankerName = "Coordinate Ascent";

	private readonly ILogger<CoordinateAscent> _logger;

	private int _currentFeature = -1;
	private double _weightChange = -1.0;

	public double[] Weight { get; private set; } = [];

	public override string Name => RankerName;

	public CoordinateAscent(ILogger<CoordinateAscent>? logger = null) => _logger = logger ?? NullLogger<CoordinateAscent>.Instance;

	public CoordinateAscent(List<RankList> samples, int[] features, MetricScorer scorer, ILogger<CoordinateAscent>? logger = null)
		: base(samples, features, scorer) =>
		_logger = logger ?? NullLogger<CoordinateAscent>.Instance;

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

		for (var r = 0; r < Parameters.RandomRestartCount; r++)
		{
			_logger.LogInformation($"[+] Random restart #{r + 1}/{Parameters.RandomRestartCount}...");
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
				_logger.PrintLog([7, 8, 7], ["Feature", "weight", Scorer.Name]);

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
							step = Parameters.StepBase * Math.Abs(origWeight);

						var totalStep = step;
						var numIter = dir == 0 ? 1 : Parameters.MaximumIterationCount;

						for (var j = 0; j < numIter; j++)
						{
							var newWeight = origWeight + totalStep;
							_weightChange = step;
							Weight[shuffledFeatures[i]] = newWeight;

							var score = Scorer.Score(Rank(Samples));
							if (Parameters.Regularized)
							{
								var penalty = Parameters.Slack * GetDistance(Weight, regVector);
								score -= penalty;
							}

							if (score > bestScore)
							{
								bestScore = score;
								bestTotalStep = totalStep;
								succeeds = true;
								var bw = Weight[shuffledFeatures[i]] > 0 ? "+" : "";
								_logger.PrintLog([7, 8, 7], [Features[shuffledFeatures[i]].ToString(),
									$"{bw}{Math.Round(Weight[shuffledFeatures[i]], 4)}",
									Math.Round(bestScore, 4).ToString(CultureInfo.InvariantCulture)
								]);
							}

							if (j < Parameters.MaximumIterationCount - 1)
							{
								step *= Parameters.StepScale;
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

				if (bestScore - startScore < Parameters.Tolerance)
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
		TrainingDataScore = Math.Round(Scorer.Score(Rank(Samples)), 4);
		_logger.LogInformation("Finished successfully.");
		_logger.LogInformation($"{Scorer.Name} on training data: {TrainingDataScore}");

		if (ValidationSamples != null)
		{
			ValidationDataScore = Scorer.Score(Rank(ValidationSamples));
			_logger.LogInformation($"{Scorer.Name} on validation data: {Math.Round(ValidationDataScore, 4)}");
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
					score[i] += Weight[j] * rankList[i].GetFeatureValue(Features[j]);

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
			score += Weight[i] * dataPoint.GetFeatureValue(Features[i]);

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
			output.AppendLine($"## Restart = {Parameters.RandomRestartCount}");
			output.AppendLine($"## MaxIteration = {Parameters.MaximumIterationCount}");
			output.AppendLine($"## StepBase = {Parameters.StepBase.ToRankLibString()}");
			output.AppendLine($"## StepScale = {Parameters.StepScale.ToRankLibString()}");
			output.AppendLine($"## Tolerance = {Parameters.Tolerance.ToRankLibString()}");
			output.AppendLine($"## Regularized = {(Parameters.Regularized ? "true" : "false")}");
			output.AppendLine($"## Slack = {Parameters.Slack.ToRankLibString()}");
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

	private static double GetDistance(double[] w1, double[] w2)
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

	private static double Normalize(double[] weights)
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
}
