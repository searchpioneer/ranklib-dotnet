using RankLib.Learning;

namespace RankLib.Metric;

/// <summary>
/// Best at K scorer
/// </summary>
public class BestAtKScorer : MetricScorer
{
	public const int DefaultK = 10;

	public BestAtKScorer() : this(DefaultK)
	{
	}

	public BestAtKScorer(int k) => K = k;

	public override double Score(RankList rankList)
	{
		var k = MaxToK(rankList, K - 1);
		return rankList[k].Label;
	}

	/// <summary>
	/// Return the index of the best item (e.g., docs with highest degree of relevance) among items in the range [0..k].
	/// NOTE: If you want best-at-k (i.e., best among top-k), you need MaxToK(rl, k-1).
	/// </summary>
	/// <param name="rl">The rank list.</param>
	/// <param name="k">The last position of the range.</param>
	/// <returns>The index of the best object in the specified range.</returns>
	public int MaxToK(RankList rl, int k)
	{
		var size = k;
		if (size < 0 || size > rl.Count - 1)
			size = rl.Count - 1;

		var max = -1.0;
		var maxI = 0;

		for (var i = 0; i <= size; i++)
		{
			if (max < rl[i].Label)
			{
				max = rl[i].Label;
				maxI = i;
			}
		}
		return maxI;
	}

	public override string Name => $"Best@{K}";

	public override double[][] SwapChange(RankList rankList)
	{
		// TODO: FIXME: Not sure if this implementation is correct!
		var labels = new int[rankList.Count];
		var best = new int[rankList.Count];
		var max = -1;
		var maxVal = -1;
		var secondMaxVal = -1; // within top-K
		var maxCount = 0; // within top-K

		for (var i = 0; i < rankList.Count; i++)
		{
			var v = (int)rankList[i].Label;
			labels[i] = v;

			if (maxVal < v)
			{
				if (i < K)
				{
					secondMaxVal = maxVal;
					maxCount = 0;
				}
				maxVal = v;
				max = i;
			}
			else if (maxVal == v && i < K)
			{
				maxCount++;
			}
			best[i] = max;
		}

		if (secondMaxVal == -1)
			secondMaxVal = 0;

		var changes = new double[rankList.Count][];
		for (var i = 0; i < rankList.Count; i++)
		{
			changes[i] = new double[rankList.Count];
			Array.Fill(changes[i], 0);
		}

		// TODO: FIXME: THIS IS VERY *INEFFICIENT*
		for (var i = 0; i < rankList.Count - 1; i++)
		{
			for (var j = i + 1; j < rankList.Count; j++)
			{
				double change = 0;
				if (j < K || i >= K)
					change = 0;
				else if (labels[i] == labels[j] || labels[j] == labels[best[K - 1]])
					change = 0;
				else if (labels[j] > labels[best[K - 1]])
					change = labels[j] - labels[best[i]];
				else if (labels[i] < labels[best[K - 1]] || maxCount > 1)
					change = 0;
				else
					change = maxVal - Math.Max(secondMaxVal, labels[j]);
				changes[i][j] = changes[j][i] = change;
			}
		}
		return changes;
	}
}
