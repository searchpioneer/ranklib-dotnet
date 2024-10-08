using RankLib.Learning;
using RankLib.Utilities;

namespace RankLib.Metric;

/// <summary>
/// Discounted Cumulative Gain scorer
/// </summary>
/// <remarks>
/// https://en.wikipedia.org/wiki/Discounted_cumulative_gain
/// </remarks>
public class DCGScorer : MetricScorer
{
	protected static readonly Lazy<double[]> DiscountCache = new(() =>
	{
		var discount = new double[5000];
		for (var i = 0; i < discount.Length; i++)
		{
			discount[i] = 1.0 / SimpleMath.LogBase2(i + 2);
		}
		return discount;
	});

	protected static readonly Lazy<double[]> GainCache = new(() =>
	{
		var gain = new double[6];
		for (var i = 0; i < gain.Length; i++)
		{
			gain[i] = (1 << i) - 1;
		}
		return gain;
	});


	public DCGScorer() : this(10)
	{
	}

	public DCGScorer(int k) => K = k;

	public override MetricScorer Copy() => new DCGScorer();

	/// <summary>
	/// Compute DCG at k.
	/// </summary>
	public override double Score(RankList rankList)
	{
		if (rankList.Count == 0)
		{
			return 0;
		}

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
		{
			changes[i] = new double[rankList.Count];
		}

		for (var i = 0; i < size; i++)
		{
			for (var j = i + 1; j < rankList.Count; j++)
			{
				changes[j][i] = changes[i][j] = (Discount(i) - Discount(j)) * (Gain(rel[i]) - Gain(rel[j]));
			}
		}

		return changes;
	}

	public override string Name => $"DCG@{K}";

	protected double GetDCG(int[] rel, int topK)
	{
		double dcg = 0;
		for (var i = 0; i < topK; i++)
		{
			dcg += Gain(rel[i]) * Discount(i);
		}
		return dcg;
	}

	// Lazy caching for discount
	protected double Discount(int index)
	{
		var discount = DiscountCache.Value;

		if (index < discount.Length)
		{
			return discount[index];
		}

		// We need to expand our cache
		var cacheSize = discount.Length + 1000;
		while (cacheSize <= index)
		{
			cacheSize += 1000;
		}
		var tmp = new double[cacheSize];
		Array.Copy(discount, tmp, discount.Length);
		for (var i = discount.Length; i < tmp.Length; i++)
		{
			tmp[i] = 1.0 / SimpleMath.LogBase2(i + 2);
		}
		discount = tmp;
		return discount[index];
	}

	// Lazy caching for gain
	protected double Gain(int rel)
	{
		var gain = GainCache.Value;

		if (rel < gain.Length)
		{
			return gain[rel];
		}

		// We need to expand our cache
		var cacheSize = gain.Length + 10;
		while (cacheSize <= rel)
		{
			cacheSize += 10;
		}
		var tmp = new double[cacheSize];
		Array.Copy(gain, tmp, gain.Length);
		for (var i = gain.Length; i < tmp.Length; i++)
		{
			tmp[i] = (1 << i) - 1; // 2^i - 1
		}
		gain = tmp;
		return gain[rel];
	}
}
