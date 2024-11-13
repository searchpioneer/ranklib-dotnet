using RankLib.Learning;
using RankLib.Utilities;

namespace RankLib.Metric;

/// <summary>
/// Discounted Cumulative Gain (DCG) scorer.
/// DCG considers the position of relevant results in the ranking. Highly relevant documents appearing lower in
/// a list should be penalized, as the graded relevance value is reduced logarithmically proportional to the position
/// of the result.
/// </summary>
/// <remarks>
/// <a href="https://en.wikipedia.org/wiki/Discounted_cumulative_gain">
/// Wikipedia article on DCG.
/// </a>
/// </remarks>
public class DCGScorer : MetricScorer
{
	public const int DefaultK = 10;

	private static volatile double[]? DiscountCache;
	private static volatile double[]? GainCache;
	private static readonly object DiscountLock = new();
	private static readonly object GainLock = new();

	private static readonly Lazy<double[]> LazyDiscountCache = new(() =>
	{
		var discount = new double[5000];
		for (var i = 0; i < discount.Length; i++)
			discount[i] = 1.0 / SimpleMath.LogBase2(i + 2);

		DiscountCache = discount;
		return discount;
	});

	private static readonly Lazy<double[]> LazyGainCache = new(() =>
	{
		var gain = new double[6];
		for (var i = 0; i < gain.Length; i++)
			gain[i] = (1 << i) - 1;

		GainCache = gain;
		return gain;
	});


	public DCGScorer() : this(DefaultK)
	{
	}

	public DCGScorer(int k) => K = k;

	/// <summary>
	/// Compute DCG at k.
	/// </summary>
	public override double Score(RankList rankList)
	{
		if (rankList.Count == 0)
			return 0;

		var topK = K > rankList.Count || K <= 0
			? rankList.Count
			: K;

		var rel = GetRelevanceLabels(rankList);
		return GetDCG(rel, topK);
	}

	public override double[][] SwapChange(RankList rankList)
	{
		var rel = GetRelevanceLabels(rankList);
		var size = (rankList.Count > K) ? K : rankList.Count;
		var changes = new double[rankList.Count][];
		for (var i = 0; i < rankList.Count; i++)
			changes[i] = new double[rankList.Count];

		for (var i = 0; i < size; i++)
		{
			for (var j = i + 1; j < rankList.Count; j++)
				changes[j][i] = changes[i][j] = (Discount(i) - Discount(j)) * (Gain(rel[i]) - Gain(rel[j]));
		}

		return changes;
	}

	public override string Name => $"DCG@{K}";

	protected double GetDCG(int[] rel, int topK)
	{
		double dcg = 0;
		for (var i = 0; i < topK; i++)
			dcg += Gain(rel[i]) * Discount(i);

		return dcg;
	}

	protected double Discount(int index)
	{
		var currentCache = DiscountCache ?? LazyDiscountCache.Value;

		if (index < currentCache.Length)
			return currentCache[index];

		lock (DiscountLock)
		{
			currentCache = DiscountCache!;
			if (index < currentCache.Length)
				return currentCache[index];

			var cacheSize = Math.Max(currentCache.Length + 1000, index + 1000);
			var newCache = new double[cacheSize];
			Array.Copy(currentCache, newCache, currentCache.Length);

			for (var i = currentCache.Length; i < newCache.Length; i++)
				newCache[i] = 1.0 / SimpleMath.LogBase2(i + 2);

			Thread.MemoryBarrier();
			DiscountCache = newCache;
			return newCache[index];
		}
	}

	protected double Gain(int rel)
	{
		var currentCache = GainCache ?? LazyGainCache.Value;
		if (rel < currentCache.Length)
			return currentCache[rel];

		lock (GainLock)
		{
			currentCache = GainCache!;
			if (rel < currentCache.Length)
				return currentCache[rel];

			var cacheSize = Math.Max(currentCache.Length + 10, rel + 10);
			var newCache = new double[cacheSize];
			Array.Copy(currentCache, newCache, currentCache.Length);

			for (var i = currentCache.Length; i < newCache.Length; i++)
				newCache[i] = (1 << i) - 1;

			Thread.MemoryBarrier();
			GainCache = newCache;
			return newCache[rel];
		}
	}
}



