using RankLib.Learning;

namespace RankLib.Metric;

public class PrecisionScorer : MetricScorer
{
	public PrecisionScorer() => K = 10;

	public PrecisionScorer(int k) => K = k;

	public override double Score(RankList rankList)
	{
		var count = 0;
		var size = K;

		if (K > rankList.Count || K <= 0)
		{
			size = rankList.Count;
		}

		for (var i = 0; i < size; i++)
		{
			if (rankList[i].Label > 0.0)
			{
				count++;
			}
		}

		return (double)count / size;
	}

	public override string Name => $"P@{K}";

	public override double[][] SwapChange(RankList rankList)
	{
		var size = (rankList.Count > K) ? K : rankList.Count;

		var changes = new double[rankList.Count][];
		for (var i = 0; i < rankList.Count; i++)
		{
			changes[i] = new double[rankList.Count];
			Array.Fill(changes[i], 0);
		}

		for (var i = 0; i < size; i++)
		{
			for (var j = size; j < rankList.Count; j++)
			{
				var c = GetBinaryRelevance(rankList[j].Label) - GetBinaryRelevance(rankList[i].Label);
				changes[i][j] = changes[j][i] = (double)c / size;
			}
		}

		return changes;
	}

	private int GetBinaryRelevance(float label) => label > 0.0 ? 1 : 0;
}
