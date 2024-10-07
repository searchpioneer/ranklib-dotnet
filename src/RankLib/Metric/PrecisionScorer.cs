using RankLib.Learning;

namespace RankLib.Metric;

public class PrecisionScorer : MetricScorer
{
	public PrecisionScorer() => K = 10;

	public PrecisionScorer(int k) => K = k;

	public override double Score(RankList rl)
	{
		var count = 0;
		var size = K;

		if (K > rl.Count || K <= 0)
		{
			size = rl.Count;
		}

		for (var i = 0; i < size; i++)
		{
			if (rl[i].Label > 0.0)
			{
				count++;
			}
		}

		return (double)count / size;
	}

	public override MetricScorer Copy() => new PrecisionScorer(K);

	public override string Name => $"P@{K}";

	public override double[][] SwapChange(RankList rl)
	{
		var size = (rl.Count > K) ? K : rl.Count;

		var changes = new double[rl.Count][];
		for (var i = 0; i < rl.Count; i++)
		{
			changes[i] = new double[rl.Count];
			Array.Fill(changes[i], 0);
		}

		for (var i = 0; i < size; i++)
		{
			for (var j = size; j < rl.Count; j++)
			{
				var c = GetBinaryRelevance(rl[j].Label) - GetBinaryRelevance(rl[i].Label);
				changes[i][j] = changes[j][i] = (double)c / size;
			}
		}

		return changes;
	}

	private int GetBinaryRelevance(float label) => label > 0.0 ? 1 : 0;
}
