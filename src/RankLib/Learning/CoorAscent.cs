using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning;

public class CoorAscent : Ranker
{
	private readonly ILogger<CoorAscent> _logger;

	// Parameters
	public static int nRestart = 5;
	public static int nMaxIteration = 25;
	public static double stepBase = 0.05;
	public static double stepScale = 2.0;
	public static double tolerance = 0.001;
	public static bool regularized = false;
	public static double slack = 0.001;

	// Local variables
	public double[] weight = null;

	protected int currentFeature = -1; // Used only during learning
	protected double weightChange = -1.0; // Used only during learning

	public CoorAscent(ILogger<CoorAscent>? logger = null) => _logger = logger ?? NullLogger<CoorAscent>.Instance;

	public CoorAscent(List<RankList> samples, int[] features, MetricScorer scorer, ILogger<CoorAscent>? logger = null)
		: base(samples, features, scorer, logger) =>
		_logger = logger ?? NullLogger<CoorAscent>.Instance;

	public override void Init()
	{
		_logger.LogInformation("Initializing...");
		weight = new double[Features.Length];
		for (var i = 0; i < weight.Length; i++)
		{
			weight[i] = 1.0 / Features.Length;
		}
	}

	public override void Learn()
	{
		var regVector = new double[weight.Length];
		Array.Copy(weight, regVector, weight.Length); // Uniform weight distribution

		double[]? bestModel = null;
		var bestModelScore = 0.0;
		var sign = new int[] { 1, -1, 0 };

		_logger.LogInformation("Training starts...");

		for (var r = 0; r < nRestart; r++)
		{
			_logger.LogInformation($"[+] Random restart #{r + 1}/{nRestart}...");
			var consecutiveFails = 0;

			for (var i = 0; i < weight.Length; i++)
			{
				weight[i] = 1.0f / Features.Length;
			}

			currentFeature = -1;
			var startScore = Scorer.Score(Rank(Samples));

			var bestScore = startScore;
			var bestWeight = new double[weight.Length];
			Array.Copy(weight, bestWeight, weight.Length);

			while ((weight.Length > 1 && consecutiveFails < weight.Length - 1) || (weight.Length == 1 && consecutiveFails == 0))
			{
				_logger.LogInformation("Shuffling features' order...");
				_logger.LogInformation("Optimizing weight vector... ");
				PrintLogLn(new[] { 7, 8, 7 }, new[] { "Feature", "weight", Scorer.Name });

				var shuffledFeatures = GetShuffledFeatures();

				for (var i = 0; i < shuffledFeatures.Length; i++)
				{
					currentFeature = shuffledFeatures[i];
					var origWeight = weight[shuffledFeatures[i]];
					double bestTotalStep = 0;
					var succeeds = false;

					for (var s = 0; s < sign.Length; s++)
					{
						var dir = sign[s];
						var step = 0.001 * dir;
						if (origWeight != 0.0 && Math.Abs(step) > 0.5 * Math.Abs(origWeight))
						{
							step = stepBase * Math.Abs(origWeight);
						}

						var totalStep = step;
						var numIter = dir == 0 ? 1 : nMaxIteration;

						for (var j = 0; j < numIter; j++)
						{
							var newWeight = origWeight + totalStep;
							weightChange = step;
							weight[shuffledFeatures[i]] = newWeight;

							var score = Scorer.Score(Rank(Samples));
							if (regularized)
							{
								var penalty = slack * GetDistance(weight, regVector);
								score -= penalty;
							}

							if (score > bestScore)
							{
								bestScore = score;
								bestTotalStep = totalStep;
								succeeds = true;
								var bw = weight[shuffledFeatures[i]] > 0 ? "+" : "";
								PrintLogLn(new[] { 7, 8, 7 }, new[] { Features[shuffledFeatures[i]].ToString(), $"{bw}{Math.Round(weight[shuffledFeatures[i]], 4)}", Math.Round(bestScore, 4).ToString() });
							}

							if (j < nMaxIteration - 1)
							{
								step *= stepScale;
								totalStep += step;
							}
						}

						if (succeeds)
							break;

						if (s < sign.Length - 1)
						{
							weightChange = -totalStep;
							UpdateCached();
							weight[shuffledFeatures[i]] = origWeight;
						}
					}

					if (succeeds)
					{
						weightChange = bestTotalStep - weight[shuffledFeatures[i]];
						UpdateCached();
						weight[shuffledFeatures[i]] = origWeight + bestTotalStep;
						consecutiveFails = 0;

						var sum = Normalize(weight);
						ScaleCached(sum);
						Array.Copy(weight, bestWeight, weight.Length);
					}
					else
					{
						consecutiveFails++;
						weightChange = -weight[shuffledFeatures[i]];
						UpdateCached();
						weight[shuffledFeatures[i]] = origWeight;
					}
				}

				if (bestScore - startScore < tolerance)
					break;
			}

			if (ValidationSamples != null)
			{
				currentFeature = -1;
				bestScore = Scorer.Score(Rank(ValidationSamples));
			}

			if (bestModel == null || bestScore > bestModelScore)
			{
				bestModelScore = bestScore;
				bestModel = new double[bestWeight.Length];
				Array.Copy(bestWeight, bestModel, bestWeight.Length);
			}
		}

		Array.Copy(bestModel, weight, bestModel.Length);
		currentFeature = -1;
		ScoreOnTrainingData = Math.Round(Scorer.Score(Rank(Samples)), 4);
		_logger.LogInformation("Finished successfully.");
		_logger.LogInformation($"{Scorer.Name} on training data: {ScoreOnTrainingData}");

		if (ValidationSamples != null)
		{
			BestScoreOnValidationData = Scorer.Score(Rank(ValidationSamples));
			_logger.LogInformation($"{Scorer.Name} on validation data: {Math.Round(BestScoreOnValidationData, 4)}");
		}
	}

	public override RankList Rank(RankList rankList)
	{
		var score = new double[rankList.Count];
		if (currentFeature == -1)
		{
			for (var i = 0; i < rankList.Count; i++)
			{
				for (var j = 0; j < Features.Length; j++)
				{
					score[i] += weight[j] * rankList[i].GetFeatureValue(Features[j]);
				}
				rankList[i].Cached = score[i];
			}
		}
		else
		{
			for (var i = 0; i < rankList.Count; i++)
			{
				score[i] = rankList[i].Cached + weightChange * rankList[i].GetFeatureValue(Features[currentFeature]);
				rankList[i].Cached = score[i];
			}
		}

		var idx = MergeSorter.Sort(score, false);
		return new RankList(rankList, idx);
	}

	public override double Eval(DataPoint p)
	{
		var score = 0.0;
		for (var i = 0; i < Features.Length; i++)
		{
			score += weight[i] * p.GetFeatureValue(Features[i]);
		}
		return score;
	}

	public virtual Ranker CreateNew() => new CoorAscent(_logger);

	public override string ToString()
	{
		var output = new StringBuilder();
		for (var i = 0; i < weight.Length; i++)
		{
			output.Append($"{Features[i]}:{weight[i]}{(i == weight.Length - 1 ? "" : " ")}");
		}
		return output.ToString();
	}

	public override string Model
	{
		get
		{
			var output = new StringBuilder();
			output.AppendLine($"## {Name}");
			output.AppendLine($"## Restart = {nRestart}");
			output.AppendLine($"## MaxIteration = {nMaxIteration}");
			output.AppendLine($"## StepBase = {stepBase}");
			output.AppendLine($"## StepScale = {stepScale}");
			output.AppendLine($"## Tolerance = {tolerance}");
			output.AppendLine($"## Regularized = {regularized}");
			output.AppendLine($"## Slack = {slack}");
			output.AppendLine(ToString());
			return output.ToString();
		}
	}

	public override void LoadFromString(string fullText)
	{
		using var reader = new StringReader(fullText);
		while (reader.ReadLine() is { } line)
		{
			if (line.StartsWith("##"))
				continue;

			var kvp = new KeyValuePairs(line);
			weight = new double[kvp.Count];
			Features = new int[kvp.Count];

			for (var i = 0; i < kvp.Count; i++)
			{
				var kv = kvp[i];
				Features[i] = int.Parse(kv.Key);
				weight[i] = double.Parse(kv.Value);
			}
			break;
		}
	}

	public override void PrintParameters()
	{
		_logger.LogInformation($"No. of random restarts: {nRestart}");
		_logger.LogInformation($"No. of iterations to search in each direction: {nMaxIteration}");
		_logger.LogInformation($"Tolerance: {tolerance}");
		if (regularized)
		{
			_logger.LogInformation($"Reg. param: {slack}");
		}
		else
		{
			_logger.LogInformation("Regularization: No");
		}
	}

	public override string Name => "Coordinate Ascent";

	// Private helper methods
	private void UpdateCached()
	{
		for (var j = 0; j < Samples.Count; j++)
		{
			var rl = Samples[j];
			for (var i = 0; i < rl.Count; i++)
			{
				var score = rl[i].Cached + weightChange * rl[i].GetFeatureValue(Features[currentFeature]);
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
			{
				rl[i].Cached = rl[i].Cached / sum;
			}
		}
	}

	private int[] GetShuffledFeatures()
	{
		var indices = Enumerable.Range(0, Features.Length).ToList();
		indices = indices.OrderBy(x => Guid.NewGuid()).ToList(); // Shuffle
		return indices.ToArray();
	}

	private double GetDistance(double[] w1, double[] w2)
	{
		var s1 = w1.Sum(Math.Abs);
		var s2 = w2.Sum(Math.Abs);
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
			{
				weights[i] /= sum;
			}
		}
		else
		{
			sum = 1;
			for (var i = 0; i < weights.Length; i++)
			{
				weights[i] = 1.0 / weights.Length;
			}
		}
		return sum;
	}

	public void CopyModel(CoorAscent ranker)
	{
		weight = new double[Features.Length];
		if (ranker.weight.Length != weight.Length)
		{
			throw RankLibException.Create("These two models use different feature set!!");
		}
		Array.Copy(ranker.weight, weight, ranker.weight.Length);
		_logger.LogInformation("Model loaded.");
	}

	public double Distance(CoorAscent ca) => GetDistance(weight, ca.weight);
}
