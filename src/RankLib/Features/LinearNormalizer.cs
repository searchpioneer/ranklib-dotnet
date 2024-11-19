using RankLib.Learning;

namespace RankLib.Features;

public class LinearNormalizer : Normalizer
{
	/// <inheritdoc />
	public override void Normalize(RankList rankList)
	{
		if (rankList.Count == 0)
			throw new ArgumentException("rank list is empty", nameof(rankList));

		var featureCount = rankList.FeatureCount;
		var featureIds = new int[featureCount];
		for (var i = 1; i <= featureCount; i++)
			featureIds[i - 1] = i;

		Normalize(rankList, featureIds);
	}

	/// <inheritdoc />
	public override void Normalize(RankList rankList, int[] featureIds)
	{
		if (rankList.Count == 0)
			throw new ArgumentException("rank list is empty", nameof(rankList));

		// Remove duplicate features from the input featureIds to avoid normalizing the same features multiple times
		featureIds = RemoveDuplicateFeatures(featureIds);

		var min = new float[featureIds.Length];
		var max = new float[featureIds.Length];
		Array.Fill(min, float.MaxValue);
		Array.Fill(max, float.MinValue);

		for (var i = 0; i < rankList.Count; i++)
		{
			var dataPoint = rankList[i];
			for (var j = 0; j < featureIds.Length; j++)
			{
				min[j] = Math.Min(min[j], dataPoint.GetFeatureValue(featureIds[j]));
				max[j] = Math.Max(max[j], dataPoint.GetFeatureValue(featureIds[j]));
			}
		}

		for (var i = 0; i < rankList.Count; i++)
		{
			var dataPoint = rankList[i];
			for (var j = 0; j < featureIds.Length; j++)
			{
				if (max[j] > min[j])
				{
					var value = (dataPoint.GetFeatureValue(featureIds[j]) - min[j]) / (max[j] - min[j]);
					dataPoint.SetFeatureValue(featureIds[j], value);
				}
				else
					dataPoint.SetFeatureValue(featureIds[j], 0);
			}
		}
	}

	/// <inheritdoc />
	public override string Name => "linear";
}
