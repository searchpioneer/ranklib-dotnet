using RankLib.Learning;

namespace RankLib.Features;

public class SumNormalizer : Normalizer
{
	public static readonly SumNormalizer Instance = new();

	public override void Normalize(RankList rankList)
	{
		if (rankList.Count == 0)
			throw new ArgumentException("The rank list is empty", nameof(rankList));

		var featureCount = rankList.FeatureCount;
		var norm = new double[featureCount];
		Array.Fill(norm, 0);

		for (var i = 0; i < rankList.Count; i++)
		{
			var dataPoint = rankList[i];
			for (var j = 1; j <= featureCount; j++)
			{
				norm[j - 1] += Math.Abs(dataPoint.GetFeatureValue(j));
			}
		}

		for (var i = 0; i < rankList.Count; i++)
		{
			var dataPoint = rankList[i];
			for (var j = 1; j <= featureCount; j++)
			{
				if (norm[j - 1] > 0)
					dataPoint.SetFeatureValue(j, (float)(dataPoint.GetFeatureValue(j) / norm[j - 1]));
			}
		}
	}

	public override void Normalize(RankList rankList, int[] fids)
	{
		if (rankList.Count == 0)
			throw new ArgumentException("The rank list is empty", nameof(rankList));

		// Remove duplicate features from the input fids to avoid normalizing the same features multiple times
		fids = RemoveDuplicateFeatures(fids);
		var norm = new double[fids.Length];
		Array.Fill(norm, 0);

		for (var i = 0; i < rankList.Count; i++)
		{
			var dataPoint = rankList[i];
			for (var j = 0; j < fids.Length; j++)
			{
				norm[j] += Math.Abs(dataPoint.GetFeatureValue(fids[j]));
			}
		}

		for (var i = 0; i < rankList.Count; i++)
		{
			var dataPoint = rankList[i];
			for (var j = 0; j < fids.Length; j++)
			{
				if (norm[j] > 0)
					dataPoint.SetFeatureValue(fids[j], Convert.ToSingle(dataPoint.GetFeatureValue(fids[j]) / norm[j]));
			}
		}
	}

	public override string Name => "sum";
}
