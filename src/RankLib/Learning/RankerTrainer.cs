using System.Diagnostics;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning;

public class RankerTrainer
{
	private readonly ILoggerFactory _loggerFactory;
	private readonly ILogger<RankerTrainer> _logger;

	protected RankerFactory rf;
	protected double trainingTime = 0;

	public RankerTrainer(ILoggerFactory? loggerFactory = null)
	{
		_loggerFactory = loggerFactory ?? NullLoggerFactory.Instance;
		_logger = _loggerFactory.CreateLogger<RankerTrainer>();
		rf = new RankerFactory(_loggerFactory);
	}

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

	public void PrintTrainingTime() => _logger.LogInformation($"Training time: {SimpleMath.Round((trainingTime) / 1e9, 2)} seconds");
}
