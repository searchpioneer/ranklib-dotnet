using RankLib.Learning;

namespace RankLib.Metric;

public class PrecisionScorer : MetricScorer
{
	public PrecisionScorer() : base() => _k = 10;

	public PrecisionScorer(int k) => _k = k;

	public override double Score(RankList rl)
	{
		var count = 0;
		var size = _k;

		if (_k > rl.Count || _k <= 0)
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

	public override MetricScorer Copy() => new PrecisionScorer(_k);

	public override string Name() => $"P@{_k}";

	public override double[][] SwapChange(RankList rl)
	{
		var size = (rl.Count > _k) ? _k : rl.Count;

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
