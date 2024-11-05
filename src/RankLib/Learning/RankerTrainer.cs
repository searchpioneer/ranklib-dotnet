using System.Diagnostics;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning;

/// <summary>
/// Trains <see cref="IRanker"/> instances.
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

	/// <summary>
	/// Trains a ranker using the provided training samples, and validates training using the validation samples.
	/// </summary>
	/// <param name="rankerType">The type of ranker to train.</param>
	/// <param name="trainingSamples">The training samples.</param>
	/// <param name="validationSamples">The validation samples.</param>
	/// <param name="features">The features</param>
	/// <param name="scorer"></param>
	/// <param name="parameters"></param>
	/// <returns></returns>
	public async Task<IRanker> TrainAsync(
		Type rankerType,
		List<RankList> trainingSamples,
		List<RankList>? validationSamples,
		int[] features,
		MetricScorer scorer,
		IRankerParameters? parameters = default)
	{
		var ranker = _rankerFactory.CreateRanker(rankerType, trainingSamples, features, scorer, parameters);
		ranker.ValidationSamples = validationSamples;
		await ranker.InitAsync().ConfigureAwait(false);
		await ranker.LearnAsync().ConfigureAwait(false);
		return ranker;
	}

	public async Task<TRanker> TrainAsync<TRanker, TRankerParameters>(
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
		await ranker.InitAsync().ConfigureAwait(false);
		await ranker.LearnAsync().ConfigureAwait(false);
		return ranker;
	}
}
