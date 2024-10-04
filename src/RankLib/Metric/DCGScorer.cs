using RankLib.Learning;
using RankLib.Utilities;

namespace RankLib.Metric;

public class DCGScorer : MetricScorer
{
	protected static double[]? discount = null; // Cache
	protected static double[]? gain = null; // Cache

	public DCGScorer()
	{
		_k = 10;
		InitCache();
	}

	public DCGScorer(int k)
	{
		_k = k;
		InitCache();
	}

	private void InitCache()
	{
		if (discount == null)
		{
			discount = new double[5000];
			for (var i = 0; i < discount.Length; i++)
			{
				discount[i] = 1.0 / SimpleMath.LogBase2(i + 2);
			}
			gain = new double[6];
			for (var i = 0; i < 6; i++)
			{
				gain[i] = (1 << i) - 1; // 2^i - 1
			}
		}
	}

	public override MetricScorer Copy() => new DCGScorer();

	/// <summary>
	/// Compute DCG at k.
	/// </summary>
	public override double Score(RankList rl)
	{
		if (rl.Count == 0)
		{
			return 0;
		}

		var size = _k;
		if (_k > rl.Count || _k <= 0)
		{
			size = rl.Count;
		}

		var rel = GetRelevanceLabels(rl);
		return GetDCG(rel, size);
	}

	public override double[][] SwapChange(RankList rl)
	{
		var rel = GetRelevanceLabels(rl);
		var size = (rl.Count > _k) ? _k : rl.Count;
		var changes = new double[rl.Count][];
		for (var i = 0; i < rl.Count; i++)
		{
			changes[i] = new double[rl.Count];
		}

		for (var i = 0; i < size; i++)
		{
			for (var j = i + 1; j < rl.Count; j++)
			{
				changes[j][i] = changes[i][j] = (Discount(i) - Discount(j)) * (Gain(rel[i]) - Gain(rel[j]));
			}
		}

		return changes;
	}

	public override string Name() => "DCG@" + _k;

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
