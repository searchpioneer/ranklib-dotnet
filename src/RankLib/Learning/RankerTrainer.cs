using System.Diagnostics;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning;

/// <summary>
/// Trains a ranker using the provided training samples, and validates training using the validation samples.
/// </summary>
public class RankerTrainer
{
	private readonly RankerFactory _rankerFactory;
	private readonly ILogger<RankerTrainer> _logger;

	public RankerTrainer(RankerFactory rankerFactory, ILogger<RankerTrainer> logger)
	{
		_rankerFactory = rankerFactory;
		_logger = logger;
	}

	public async Task<(IRanker ranker, TimeSpan trainingTime)> TrainAsync(
		Type rankerType,
		List<RankList> trainingSamples,
		List<RankList>? validationSamples,
		int[] features,
		MetricScorer scorer,
		IRankerParameters? parameters = default)
	{
		var ranker = _rankerFactory.CreateRanker(rankerType, trainingSamples, features, scorer, parameters);
		ranker.ValidationSamples = validationSamples;
		var stopwatch = Stopwatch.StartNew();
		await ranker.InitAsync();
		await ranker.LearnAsync();
		stopwatch.Stop();
		return (ranker, stopwatch.Elapsed);
	}

	public async Task<(TRanker ranker, TimeSpan trainingTime)> TrainAsync<TRanker, TRankerParameters>(
		List<RankList> trainingSamples,
		List<RankList>? validationSamples,
		int[] features,
		MetricScorer scorer,
		TRankerParameters? parameters = default)
		where TRanker : IRanker<TRankerParameters>
		where TRankerParameters : IRankerParameters
	{
		var ranker = _rankerFactory.CreateRanker<TRanker, TRankerParameters>(trainingSamples, features, scorer, parameters);
		ranker.ValidationSamples = validationSamples;
		var stopwatch = Stopwatch.StartNew();
		await ranker.InitAsync();
		await ranker.LearnAsync();
		stopwatch.Stop();
		return (ranker, stopwatch.Elapsed);
	}
}
