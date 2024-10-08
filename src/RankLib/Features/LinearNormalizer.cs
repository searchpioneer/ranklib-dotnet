using RankLib.Learning;

namespace RankLib.Features;

public class LinearNormalizer : Normalizer
{
	public override void Normalize(RankList rl)
	{
		if (rl.Count == 0)
		{
			throw new InvalidOperationException("The input ranked list is empty");
		}

		var featureCount = rl.FeatureCount;
		var fids = new int[featureCount];
		for (var i = 1; i <= featureCount; i++)
		{
			fids[i - 1] = i;
		}

		Normalize(rl, fids);
	}

	public override void Normalize(RankList rl, int[] fids)
	{
		if (rl.Count == 0)
		{
			throw new InvalidOperationException("The input ranked list is empty");
		}

		// Remove duplicate features from the input fids to avoid normalizing the same features multiple times
		fids = RemoveDuplicateFeatures(fids);

		var min = new float[fids.Length];
		var max = new float[fids.Length];
		Array.Fill(min, float.MaxValue);
		Array.Fill(max, float.MinValue);

		for (var i = 0; i < rl.Count; i++)
		{
			var dp = rl[i];
			for (var j = 0; j < fids.Length; j++)
			{
				min[j] = Math.Min(min[j], dp.GetFeatureValue(fids[j]));
				max[j] = Math.Max(max[j], dp.GetFeatureValue(fids[j]));
			}
		}

		for (var i = 0; i < rl.Count; i++)
		{
			var dp = rl[i];
			for (var j = 0; j < fids.Length; j++)
			{
				if (max[j] > min[j])
				{
					var value = (dp.GetFeatureValue(fids[j]) - min[j]) / (max[j] - min[j]);
					dp.SetFeatureValue(fids[j], value);
				}
				else
				{
					dp.SetFeatureValue(fids[j], 0);
				}
			}
		}
	}

	public override string Name => "linear";
}
