using RankLib.Learning;

namespace RankLib.Metric;

public class ReciprocalRankScorer : MetricScorer
{
	public ReciprocalRankScorer() => _k = 0; // consider the whole list

	public override double Score(RankList rl)
	{
		var size = (rl.Count > _k) ? _k : rl.Count;
		var firstRank = -1;

		for (var i = 0; i < size && firstRank == -1; i++)
		{
			if (rl[i].Label > 0.0)
			{
				firstRank = i + 1;
			}
		}

		return (firstRank == -1) ? 0 : (1.0 / firstRank);
	}

	public override MetricScorer Copy() => new ReciprocalRankScorer();

	public override string Name() => $"RR@{_k}";

	public override double[][] SwapChange(RankList rl)
	{
		var firstRank = -1;
		var secondRank = -1;
		var size = (rl.Count > _k) ? _k : rl.Count;

		for (var i = 0; i < size; i++)
		{
			if (rl[i].Label > 0.0)
			{
				if (firstRank == -1)
				{
					firstRank = i;
				}
				else if (secondRank == -1)
				{
					secondRank = i;
				}
			}
		}

		// Initialize changes array
		var changes = new double[rl.Count][];
		for (var i = 0; i < rl.Count; i++)
		{
			changes[i] = new double[rl.Count];
			Array.Fill(changes[i], 0);
		}

		// Calculate change in Reciprocal Rank (RR)
		var rr = 0.0;
		if (firstRank != -1)
		{
			rr = 1.0 / (firstRank + 1);
			for (var j = firstRank + 1; j < size; j++)
			{
				if (rl[j].Label == 0)
				{
					if (secondRank == -1 || j < secondRank)
					{
						changes[firstRank][j] = changes[j][firstRank] = 1.0 / (j + 1) - rr;
					}
					else
					{
						changes[firstRank][j] = changes[j][firstRank] = 1.0 / (secondRank + 1) - rr;
					}
				}
			}

			for (var j = size; j < rl.Count; j++)
			{
				if (rl[j].Label == 0)
				{
					if (secondRank == -1)
					{
						changes[firstRank][j] = changes[j][firstRank] = -rr;
					}
					else
					{
						changes[firstRank][j] = changes[j][firstRank] = 1.0 / (secondRank + 1) - rr;
					}
				}
			}
		}
		else
		{
			firstRank = size;
		}

		// Consider swapping documents below firstRank with those earlier in the list
		for (var i = 0; i < firstRank; i++)
		{
			for (var j = firstRank; j < rl.Count; j++)
			{
				if (rl[j].Label > 0)
				{
					changes[i][j] = changes[j][i] = 1.0 / (i + 1) - rr;
				}
			}
		}

		return changes;
	}
}
