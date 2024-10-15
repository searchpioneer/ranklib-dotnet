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

	public RankerTrainer(RankerFactory rankerFactory) => _rankerFactory = rankerFactory;

	public (IRanker ranker, TimeSpan trainingTime) Train(
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
		ranker.Init();
		ranker.Learn();
		stopwatch.Stop();
		return (ranker, stopwatch.Elapsed);
	}

	public (TRanker ranker, TimeSpan trainingTime) Train<TRanker, TRankerParameters>(
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
		ranker.Init();
		ranker.Learn();
		stopwatch.Stop();
		return (ranker, stopwatch.Elapsed);
	}
}
