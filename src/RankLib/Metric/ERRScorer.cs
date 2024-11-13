using RankLib.Learning;

namespace RankLib.Metric;

/// <summary>
/// Expected Reciprocal Rank.
/// Provides a good signal whether the top result is relevant.
/// </summary>
/// <remarks>
/// <a href="https://doi.org/10.1145/1645953.1646033">
/// Olivier Chapelle, Donald Metlzer, Ya Zhang, and Pierre Grinspan. (2009).
/// Expected reciprocal rank for graded relevance.
/// In Proceedings of the 18th ACM conference on Information and knowledge management (CIKM '09).
/// </a>
/// </remarks>
public class ERRScorer : MetricScorer
{
	/// <summary>
	/// Highest judged relevance label. By default, a 5 point scale is assumed [0,1,2,3,4]. The calculated
	/// max value is <c>Math.Pow(2, Max)</c>
	/// </summary>
	public const double DefaultMax = 4;

	public const int DefaultK = 10;

	public double Max { get; internal set; }

	public ERRScorer() : this(DefaultK)
	{
	}

	public ERRScorer(int k) : this(k, DefaultMax)
	{
	}

	public ERRScorer(int k, double max)
	{
		Max = Math.Pow(2, max);
		K = k;
	}

	/// <summary>
	/// Compute ERR at k.
	/// </summary>
	public override double Score(RankList rankList)
	{
		var size = K;
		if (K > rankList.Count || K <= 0)
			size = rankList.Count;

		var rel = new List<int>();
		for (var i = 0; i < rankList.Count; i++)
			rel.Add((int)rankList[i].Label);

		var s = 0.0d;
		var p = 1.0d;
		for (var i = 1; i <= size; i++)
		{
			var r = R(rel[i - 1]);
			s += p * r / i;
			p *= 1.0 - r;
		}
		return s;
	}

	public override string Name => $"ERR@{K}";

	private double R(int rel) => ((1 << rel) - 1) / Max; // (2^rel - 1)/MAX

	public override double[][] SwapChange(RankList rankList)
	{
		var size = rankList.Count > K ? K : rankList.Count;
		var labels = new int[rankList.Count];
		var r = new double[rankList.Count];
		var np = new double[rankList.Count]; // p[i] = (1 - p[0])(1 - p[1])...(1 - p[i - 1])
		var p = 1.0;

		for (var i = 0; i < size; i++)
		{
			labels[i] = (int)rankList[i].Label;
			r[i] = R(labels[i]);
			np[i] = p * (1.0 - r[i]);
			p *= np[i];
		}

		var changes = new double[rankList.Count][];
		for (var i = 0; i < rankList.Count; i++)
		{
			changes[i] = new double[rankList.Count];
			Array.Fill(changes[i], 0);
		}

		for (var i = 0; i < size; i++)
		{
			var v1 = 1.0 / (i + 1) * (i == 0 ? 1 : np[i - 1]);
			for (var j = i + 1; j < rankList.Count; j++)
			{
				double change;
				if (labels[i] == labels[j])
					change = 0;
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
