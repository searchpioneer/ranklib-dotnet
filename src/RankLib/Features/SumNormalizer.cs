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

	public override void Normalize(RankList rankList, int[] featureIds)
	{
		if (rankList.Count == 0)
			throw new ArgumentException("The rank list is empty", nameof(rankList));

		// Remove duplicate features from the input featureIds to avoid normalizing the same features multiple times
		featureIds = RemoveDuplicateFeatures(featureIds);
		var norm = new double[featureIds.Length];
		Array.Fill(norm, 0);

		for (var i = 0; i < rankList.Count; i++)
		{
			var dataPoint = rankList[i];
			for (var j = 0; j < featureIds.Length; j++)
			{
				norm[j] += Math.Abs(dataPoint.GetFeatureValue(featureIds[j]));
			}
		}

		for (var i = 0; i < rankList.Count; i++)
		{
			var dataPoint = rankList[i];
			for (var j = 0; j < featureIds.Length; j++)
			{
				if (norm[j] > 0)
					dataPoint.SetFeatureValue(featureIds[j], (float)(dataPoint.GetFeatureValue(featureIds[j]) / norm[j]));
			}
		}
	}

	public override string Name => "sum";
}
