using RankLib.Metric;

namespace RankLib.Learning;

/// <summary>
/// Trains <see cref="IRanker"/> instances.
/// </summary>
public class RankerTrainer
{
	private readonly RankerFactory _rankerFactory;

	/// <summary>
	/// Instantiates a new instance of <see cref="RankerTrainer"/>
	/// </summary>
	/// <param name="rankerFactory">The ranker factory used to create rankers</param>
	public RankerTrainer(RankerFactory rankerFactory) => _rankerFactory = rankerFactory;

	/// <summary>
	/// Trains a ranker using the provided training samples, and validates training using the validation samples.
	/// </summary>
	/// <param name="rankerType">The type of ranker to train.</param>
	/// <param name="trainingSamples">The training samples.</param>
	/// <param name="validationSamples">The validation samples.</param>
	/// <param name="features">The features</param>
	/// <param name="scorer">the scorer used to measure the effectiveness of the ranker</param>
	/// <param name="parameters">The ranking parameters</param>
	/// <param name="cancellationToken">Token that can be used to cancel the operation</param>
	/// <returns>A new instance of a trained <see cref="IRanker"/></returns>
	public async Task<IRanker> TrainAsync(
		Type rankerType,
		List<RankList> trainingSamples,
		List<RankList>? validationSamples,
		int[] features,
		MetricScorer scorer,
		IRankerParameters? parameters = default,
		CancellationToken cancellationToken = default)
	{
		var ranker = _rankerFactory.CreateRanker(rankerType, trainingSamples, features, scorer, parameters);
		ranker.ValidationSamples = validationSamples;
		await ranker.InitAsync(cancellationToken).ConfigureAwait(false);
		await ranker.LearnAsync(cancellationToken).ConfigureAwait(false);
		return ranker;
	}

	/// <summary>
	/// Trains a ranker using the provided training samples, and validates training using the validation samples.
	/// </summary>
	/// <param name="trainingSamples">The training samples.</param>
	/// <param name="validationSamples">The validation samples.</param>
	/// <param name="features">The features</param>
	/// <param name="scorer">the scorer used to measure the effectiveness of the ranker</param>
	/// <param name="parameters">The ranking parameters</param>
	/// <param name="cancellationToken">Token that can be used to cancel the operation</param>
	/// <returns>A new instance of a trained <see cref="IRanker"/></returns>
	/// <typeparam name="TRanker">The type of ranker</typeparam>
	/// <typeparam name="TRankerParameters">The type of ranker parameters</typeparam>
	/// <returns>A new instance of a trained ranker</returns>
	public async Task<TRanker> TrainAsync<TRanker, TRankerParameters>(
		List<RankList> trainingSamples,
		List<RankList>? validationSamples,
		int[] features,
		MetricScorer scorer,
		TRankerParameters? parameters = default,
		CancellationToken cancellationToken = default)
		where TRanker : IRanker<TRankerParameters>
		where TRankerParameters : IRankerParameters
	{
		var ranker = _rankerFactory.CreateRanker<TRanker, TRankerParameters>(trainingSamples, features, scorer, parameters);
		ranker.ValidationSamples = validationSamples;
		await ranker.InitAsync(cancellationToken).ConfigureAwait(false);
		await ranker.LearnAsync(cancellationToken).ConfigureAwait(false);
		return ranker;
	}
}
