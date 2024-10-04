using System.Diagnostics;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning;

public class RankerTrainer
{
	private static readonly ILogger<RankerTrainer> logger = NullLogger<RankerTrainer>.Instance;

	protected RankerFactory rf = new();
	protected double trainingTime = 0;

	public Ranker Train(RankerType type, List<RankList> train, int[] features, MetricScorer scorer)
	{
		var ranker = rf.CreateRanker(type, train, features, scorer);
		var stopwatch = Stopwatch.StartNew();
		ranker.Init();
		ranker.Learn();
		stopwatch.Stop();
		trainingTime = stopwatch.Elapsed.TotalMilliseconds * 1e6; // Convert to nanoseconds
		return ranker;
	}

	public Ranker Train(RankerType type, List<RankList> train, List<RankList> validation, int[] features, MetricScorer scorer)
	{
		var ranker = rf.CreateRanker(type, train, features, scorer);
		ranker.SetValidationSet(validation);
		var stopwatch = Stopwatch.StartNew();
		ranker.Init();
		ranker.Learn();
		stopwatch.Stop();
		trainingTime = stopwatch.Elapsed.TotalMilliseconds * 1e6; // Convert to nanoseconds
		return ranker;
	}

	public double GetTrainingTime() => trainingTime;

	public void PrintTrainingTime() => logger.LogInformation($"Training time: {SimpleMath.Round((trainingTime) / 1e9, 2)} seconds");
}
