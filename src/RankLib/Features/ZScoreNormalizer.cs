using RankLib.Learning;

namespace RankLib.Features;

public class ZScoreNormalizer : Normalizer
{
	/// <inheritdoc />
	public override void Normalize(RankList rankList)
	{
		if (rankList.Count == 0)
			throw new ArgumentException("The rank list is empty", nameof(rankList));

		var nFeature = rankList.FeatureCount;
		var means = new double[nFeature];
		Array.Fill(means, 0);

		for (var i = 0; i < rankList.Count; i++)
		{
			var dp = rankList[i];
			for (var j = 1; j <= nFeature; j++)
				means[j - 1] += dp.GetFeatureValue(j);
		}

		for (var j = 1; j <= nFeature; j++)
		{
			means[j - 1] /= rankList.Count;
			double std = 0;

			for (var i = 0; i < rankList.Count; i++)
			{
				var p = rankList[i];
				var x = p.GetFeatureValue(j) - means[j - 1];
				std += x * x;
			}

			std = Math.Sqrt(std / (rankList.Count - 1));

			if (std > 0)
			{
				for (var i = 0; i < rankList.Count; i++)
				{
					var p = rankList[i];
					var x = (p.GetFeatureValue(j) - means[j - 1]) / std; // standard normal (0, 1)
					p.SetFeatureValue(j, (float)x);
				}
			}
		}
	}

	/// <inheritdoc />
	public override void Normalize(RankList rankList, int[] featureIds)
	{
		if (rankList.Count == 0)
			throw new ArgumentException("The rank list is empty", nameof(rankList));

		// Remove duplicate features from the input featureIds to avoid normalizing the same features multiple times
		featureIds = RemoveDuplicateFeatures(featureIds);

		var means = new double[featureIds.Length];
		Array.Fill(means, 0);

		for (var i = 0; i < rankList.Count; i++)
		{
			var dataPoint = rankList[i];
			for (var j = 0; j < featureIds.Length; j++)
				means[j] += dataPoint.GetFeatureValue(featureIds[j]);
		}

		for (var j = 0; j < featureIds.Length; j++)
		{
			means[j] /= rankList.Count;
			double std = 0;

			for (var i = 0; i < rankList.Count; i++)
			{
				var dataPoint = rankList[i];
				var x = dataPoint.GetFeatureValue(featureIds[j]) - means[j];
				std += x * x;
			}

			std = Math.Sqrt(std / (rankList.Count - 1));

			if (std > 0.0)
			{
				for (var i = 0; i < rankList.Count; i++)
				{
					var p = rankList[i];
					var x = (p.GetFeatureValue(featureIds[j]) - means[j]) / std; // standard normal (0, 1)
					p.SetFeatureValue(featureIds[j], (float)x);
				}
			}
		}
	}

	/// <inheritdoc />
	public override string Name => "zscore";
}
