using RankLib.Learning;

namespace RankLib.Features;

public class ZScoreNormalizer : Normalizer
{
	public override void Normalize(RankList rl)
	{
		if (rl.Count == 0)
		{
			throw new InvalidOperationException("Error in ZScoreNormalizer::Normalize(): The input ranked list is empty");
		}

		var nFeature = rl.FeatureCount;
		var means = new double[nFeature];
		Array.Fill(means, 0);

		for (var i = 0; i < rl.Count; i++)
		{
			var dp = rl[i];
			for (var j = 1; j <= nFeature; j++)
			{
				means[j - 1] += dp.GetFeatureValue(j);
			}
		}

		for (var j = 1; j <= nFeature; j++)
		{
			means[j - 1] /= rl.Count;
			double std = 0;

			for (var i = 0; i < rl.Count; i++)
			{
				var p = rl[i];
				var x = p.GetFeatureValue(j) - means[j - 1];
				std += x * x;
			}

			std = Math.Sqrt(std / (rl.Count - 1));

			if (std > 0)
			{
				for (var i = 0; i < rl.Count; i++)
				{
					var p = rl[i];
					var x = (p.GetFeatureValue(j) - means[j - 1]) / std; // standard normal (0, 1)
					p.SetFeatureValue(j, (float)x);
				}
			}
		}
	}

	public override void Normalize(RankList rl, int[] fids)
	{
		if (rl.Count == 0)
		{
			throw new InvalidOperationException("Error in ZScoreNormalizer::Normalize(): The input ranked list is empty");
		}

		// Remove duplicate features from the input fids to avoid normalizing the same features multiple times
		fids = RemoveDuplicateFeatures(fids);

		var means = new double[fids.Length];
		Array.Fill(means, 0);

		for (var i = 0; i < rl.Count; i++)
		{
			var dp = rl[i];
			for (var j = 0; j < fids.Length; j++)
			{
				means[j] += dp.GetFeatureValue(fids[j]);
			}
		}

		for (var j = 0; j < fids.Length; j++)
		{
			means[j] /= rl.Count;
			double std = 0;

			for (var i = 0; i < rl.Count; i++)
			{
				var p = rl[i];
				var x = p.GetFeatureValue(fids[j]) - means[j];
				std += x * x;
			}

			std = Math.Sqrt(std / (rl.Count - 1));

			if (std > 0.0)
			{
				for (var i = 0; i < rl.Count; i++)
				{
					var p = rl[i];
					var x = (p.GetFeatureValue(fids[j]) - means[j]) / std; // standard normal (0, 1)
					p.SetFeatureValue(fids[j], (float)x);
				}
			}
		}
	}

	public override string Name => "zscore";
}
