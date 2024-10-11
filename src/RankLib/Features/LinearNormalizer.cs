using RankLib.Learning;

namespace RankLib.Features;

public class LinearNormalizer : Normalizer
{
	public override void Normalize(RankList rankList)
	{
		if (rankList.Count == 0)
			throw new ArgumentException("rank list is empty", nameof(rankList));

		var featureCount = rankList.FeatureCount;
		var fids = new int[featureCount];
		for (var i = 1; i <= featureCount; i++)
		{
			fids[i - 1] = i;
		}

		Normalize(rankList, fids);
	}

	public override void Normalize(RankList rankList, int[] fids)
	{
		if (rankList.Count == 0)
			throw new ArgumentException("rank list is empty", nameof(rankList));

		// Remove duplicate features from the input fids to avoid normalizing the same features multiple times
		fids = RemoveDuplicateFeatures(fids);

		var min = new float[fids.Length];
		var max = new float[fids.Length];
		Array.Fill(min, float.MaxValue);
		Array.Fill(max, float.MinValue);

		for (var i = 0; i < rankList.Count; i++)
		{
			var dataPoint = rankList[i];
			for (var j = 0; j < fids.Length; j++)
			{
				min[j] = Math.Min(min[j], dataPoint.GetFeatureValue(fids[j]));
				max[j] = Math.Max(max[j], dataPoint.GetFeatureValue(fids[j]));
			}
		}

		for (var i = 0; i < rankList.Count; i++)
		{
			var dataPoint = rankList[i];
			for (var j = 0; j < fids.Length; j++)
			{
				if (max[j] > min[j])
				{
					var value = (dataPoint.GetFeatureValue(fids[j]) - min[j]) / (max[j] - min[j]);
					dataPoint.SetFeatureValue(fids[j], value);
				}
				else
				{
					dataPoint.SetFeatureValue(fids[j], 0);
				}
			}
		}
	}

	public override string Name => "linear";
}
