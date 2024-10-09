using System.Diagnostics;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning;

public class RankerTrainer
{
	private readonly ILogger<RankerTrainer> _logger;
	private readonly RankerFactory _rankerFactory;
	public double TrainingTime { get; protected set; }

	public RankerTrainer(RankerFactory rankerFactory, ILogger<RankerTrainer>? logger = null)
	{
		_logger = logger ?? NullLogger<RankerTrainer>.Instance;
		_rankerFactory = rankerFactory;
	}

	public Ranker Train(RankerType type, List<RankList> train, int[] features, MetricScorer scorer)
	{
		var ranker = _rankerFactory.CreateRanker(type, train, features, scorer);
		var stopwatch = Stopwatch.StartNew();
		ranker.Init();
		ranker.Learn();
		stopwatch.Stop();
		TrainingTime = stopwatch.Elapsed.TotalMilliseconds * 1e6; // Convert to nanoseconds
		return ranker;
	}

	public Ranker Train(RankerType type, List<RankList> train, List<RankList> validation, int[] features, MetricScorer scorer)
	{
		var ranker = _rankerFactory.CreateRanker(type, train, features, scorer);
		ranker.SetValidationSet(validation);
		var stopwatch = Stopwatch.StartNew();
		ranker.Init();
		ranker.Learn();
		stopwatch.Stop();
		TrainingTime = stopwatch.Elapsed.TotalMilliseconds * 1e6; // Convert to nanoseconds
		return ranker;
	}

	public void PrintTrainingTime() => _logger.LogInformation($"Training time: {SimpleMath.Round((TrainingTime) / 1e9, 2)} seconds");
}
