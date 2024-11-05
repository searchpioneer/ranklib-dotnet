using RankLib.Learning;

namespace RankLib.Metric;

public class ReciprocalRankScorer : MetricScorer
{
	public ReciprocalRankScorer() => K = 0; // consider the whole list

	public override double Score(RankList rankList)
	{
		var size = rankList.Count > K ? K : rankList.Count;
		var firstRank = -1;

		for (var i = 0; i < size && firstRank == -1; i++)
		{
			if (rankList[i].Label > 0)
				firstRank = i + 1;
		}

		return firstRank == -1
			? 0
			: 1.0 / firstRank;
	}

	public override string Name => $"RR@{K}";

	public override double[][] SwapChange(RankList rankList)
	{
		var firstRank = -1;
		var secondRank = -1;
		var size = rankList.Count > K ? K : rankList.Count;

		for (var i = 0; i < size; i++)
		{
			if (rankList[i].Label > 0)
			{
				if (firstRank == -1)
					firstRank = i;
				else if (secondRank == -1)
					secondRank = i;
			}
		}

		// Initialize changes array
		var changes = new double[rankList.Count][];
		for (var i = 0; i < rankList.Count; i++)
		{
			changes[i] = new double[rankList.Count];
			Array.Fill(changes[i], 0);
		}

		// Calculate change in Reciprocal Rank (RR)
		var rr = 0.0;
		if (firstRank != -1)
		{
			rr = 1.0 / (firstRank + 1);
			for (var j = firstRank + 1; j < size; j++)
			{
				if (rankList[j].Label == 0)
				{
					if (secondRank == -1 || j < secondRank)
						changes[firstRank][j] = changes[j][firstRank] = 1.0 / (j + 1) - rr;
					else
						changes[firstRank][j] = changes[j][firstRank] = 1.0 / (secondRank + 1) - rr;
				}
			}

			for (var j = size; j < rankList.Count; j++)
			{
				if (rankList[j].Label == 0)
				{
					if (secondRank == -1)
						changes[firstRank][j] = changes[j][firstRank] = -rr;
					else
						changes[firstRank][j] = changes[j][firstRank] = 1.0 / (secondRank + 1) - rr;
				}
			}
		}
		else
			firstRank = size;

		// Consider swapping documents below firstRank with those earlier in the list
		for (var i = 0; i < firstRank; i++)
		{
			for (var j = firstRank; j < rankList.Count; j++)
			{
				if (rankList[j].Label > 0)
					changes[i][j] = changes[j][i] = 1.0 / (i + 1) - rr;
			}
		}

		return changes;
	}
}
