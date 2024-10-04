using RankLib.Learning;

namespace RankLib.Features;

public class SumNormalizer : Normalizer
{
	public override void Normalize(RankList rl)
	{
		if (rl.Count == 0)
		{
			throw new InvalidOperationException("Error in SumNormalizer::Normalize(): The input ranked list is empty");
		}

		var nFeature = rl.GetFeatureCount();
		var norm = new double[nFeature];
		Array.Fill(norm, 0);

		for (var i = 0; i < rl.Count; i++)
		{
			var dp = rl[i];
			for (var j = 1; j <= nFeature; j++)
			{
				norm[j - 1] += Math.Abs(dp.GetFeatureValue(j));
			}
		}

		for (var i = 0; i < rl.Count; i++)
		{
			var dp = rl[i];
			for (var j = 1; j <= nFeature; j++)
			{
				if (norm[j - 1] > 0)
				{
					dp.SetFeatureValue(j, (float)(dp.GetFeatureValue(j) / norm[j - 1]));
				}
			}
		}
	}

	public override void Normalize(RankList rl, int[] fids)
	{
		if (rl.Count == 0)
		{
			throw new InvalidOperationException("Error in SumNormalizer::Normalize(): The input ranked list is empty");
		}

		// Remove duplicate features from the input fids to avoid normalizing the same features multiple times
		fids = RemoveDuplicateFeatures(fids);

		var norm = new double[fids.Length];
		Array.Fill(norm, 0);

		for (var i = 0; i < rl.Count; i++)
		{
			var dp = rl[i];
			for (var j = 0; j < fids.Length; j++)
			{
				norm[j] += Math.Abs(dp.GetFeatureValue(fids[j]));
			}
		}

		for (var i = 0; i < rl.Count; i++)
		{
			var dp = rl[i];
			for (var j = 0; j < fids.Length; j++)
			{
				if (norm[j] > 0)
				{
					dp.SetFeatureValue(fids[j], (float)(dp.GetFeatureValue(fids[j]) / norm[j]));
				}
			}
		}
	}

	public override string Name => "sum";
}
