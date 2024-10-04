using RankLib.Learning;

namespace RankLib.Metric;

public class BestAtKScorer : MetricScorer
{
	public BestAtKScorer() => _k = 10;

	public BestAtKScorer(int k) => _k = k;

	public override double Score(RankList rl)
	{
		var k = MaxToK(rl, _k - 1);
		return rl[k].Label;
	}

	public override MetricScorer Copy() => new BestAtKScorer();

	/// <summary>
	/// Return the position of the best object (e.g., docs with highest degree of relevance) among objects in the range [0..k].
	/// NOTE: If you want best-at-k (i.e., best among top-k), you need MaxToK(rl, k-1).
	/// </summary>
	/// <param name="rl">The rank list.</param>
	/// <param name="k">The last position of the range.</param>
	/// <returns>The index of the best object in the specified range.</returns>
	public int MaxToK(RankList rl, int k)
	{
		var size = k;
		if (size < 0 || size > rl.Count - 1)
		{
			size = rl.Count - 1;
		}

		var max = -1.0;
		var max_i = 0;

		for (var i = 0; i <= size; i++)
		{
			if (max < rl[i].Label)
			{
				max = rl[i].Label;
				max_i = i;
			}
		}
		return max_i;
	}

	public override string Name() => "Best@" + _k;

	public override double[][] SwapChange(RankList rl)
	{
		//FIXME: Not sure if this implementation is correct!
		var labels = new int[rl.Count];
		var best = new int[rl.Count];
		var max = -1;
		var maxVal = -1;
		var secondMaxVal = -1; // within top-K
		var maxCount = 0; // within top-K

		for (var i = 0; i < rl.Count; i++)
		{
			var v = (int)rl[i].Label;
			labels[i] = v;

			if (maxVal < v)
			{
				if (i < _k)
				{
					secondMaxVal = maxVal;
					maxCount = 0;
				}
				maxVal = v;
				max = i;
			}
			else if (maxVal == v && i < _k)
			{
				maxCount++;
			}
			best[i] = max;
		}

		if (secondMaxVal == -1)
		{
			secondMaxVal = 0;
		}

		var changes = new double[rl.Count][];
		for (var i = 0; i < rl.Count; i++)
		{
			changes[i] = new double[rl.Count];
			Array.Fill(changes[i], 0);
		}

		//FIXME: THIS IS VERY *INEFFICIENT*
		for (var i = 0; i < rl.Count - 1; i++)
		{
			for (var j = i + 1; j < rl.Count; j++)
			{
				double change = 0;
				if (j < _k || i >= _k)
				{
					change = 0;
				}
				else if (labels[i] == labels[j] || labels[j] == labels[best[_k - 1]])
				{
					change = 0;
				}
				else if (labels[j] > labels[best[_k - 1]])
				{
					change = labels[j] - labels[best[i]];
				}
				else if (labels[i] < labels[best[_k - 1]] || maxCount > 1)
				{
					change = 0;
				}
				else
				{
					change = maxVal - Math.Max(secondMaxVal, labels[j]);
				}
				changes[i][j] = changes[j][i] = change;
			}
		}
		return changes;
	}
}
