using RankLib.Learning;

namespace RankLib.Metric;

public class ERRScorer : MetricScorer
{
	public static double MAX = 16; // By default, we assume the relevance scale of {0, 1, 2, 3, 4} => g_max = 4 => 2^g_max = 16

	public ERRScorer() => K = 10;

	public ERRScorer(int k) => K = k;

	public override MetricScorer Copy() => new ERRScorer();

	/// <summary>
	/// Compute ERR at k.
	/// </summary>
	public override double Score(RankList rl)
	{
		var size = K;
		if (K > rl.Count || K <= 0)
		{
			size = rl.Count;
		}

		var rel = new List<int>();
		for (var i = 0; i < rl.Count; i++)
		{
			rel.Add((int)rl[i].Label);
		}

		var s = 0.0;
		var p = 1.0;
		for (var i = 1; i <= size; i++)
		{
			var r = R(rel[i - 1]);
			s += p * r / i;
			p *= (1.0 - r);
		}
		return s;
	}

	public override string Name => "ERR@" + K;

	private double R(int rel) => ((1 << rel) - 1) / MAX; // (2^rel - 1)/MAX

	public override double[][] SwapChange(RankList rl)
	{
		var size = (rl.Count > K) ? K : rl.Count;
		var labels = new int[rl.Count];
		var r = new double[rl.Count];
		var np = new double[rl.Count]; // p[i] = (1 - p[0])(1 - p[1])...(1 - p[i - 1])
		var p = 1.0;

		for (var i = 0; i < size; i++)
		{
			labels[i] = (int)rl[i].Label;
			r[i] = R(labels[i]);
			np[i] = p * (1.0 - r[i]);
			p *= np[i];
		}

		var changes = new double[rl.Count][];
		for (var i = 0; i < rl.Count; i++)
		{
			changes[i] = new double[rl.Count];
			Array.Fill(changes[i], 0);
		}

		for (var i = 0; i < size; i++)
		{
			var v1 = 1.0 / (i + 1) * (i == 0 ? 1 : np[i - 1]);
			double change = 0;
			for (var j = i + 1; j < rl.Count; j++)
			{
				if (labels[i] == labels[j])
				{
					change = 0;
				}
				else
				{
					change = v1 * (r[j] - r[i]);
					p = (i == 0 ? 1 : np[i - 1]) * (r[i] - r[j]);
					for (var k = i + 1; k < j; k++)
					{
						change += p * r[k] / (1 + k);
						p *= 1.0 - r[k];
					}
					change += (np[j - 1] * (1.0 - r[j]) * r[i] / (1.0 - r[i]) - np[j - 1] * r[j]) / (j + 1);
				}
				changes[j][i] = changes[i][j] = change;
			}
		}
		return changes;
	}
}
