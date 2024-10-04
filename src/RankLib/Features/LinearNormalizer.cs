using RankLib.Learning;

namespace RankLib.Features;

public class LinearNormalizer : Normalizer
{
    public override void Normalize(RankList rl)
    {
        if (rl.Size() == 0)
        {
            throw new InvalidOperationException("Error in LinearNormalizer::Normalize(): The input ranked list is empty");
        }

        int nFeature = rl.GetFeatureCount();
        int[] fids = new int[nFeature];
        for (int i = 1; i <= nFeature; i++)
        {
            fids[i - 1] = i;
        }

        Normalize(rl, fids);
    }

    public override void Normalize(RankList rl, int[] fids)
    {
        if (rl.Size() == 0)
        {
            throw new InvalidOperationException("Error in LinearNormalizer::Normalize(): The input ranked list is empty");
        }

        // Remove duplicate features from the input fids to avoid normalizing the same features multiple times
        fids = RemoveDuplicateFeatures(fids);

        float[] min = new float[fids.Length];
        float[] max = new float[fids.Length];
        Array.Fill(min, float.MaxValue);
        Array.Fill(max, float.MinValue);

        for (int i = 0; i < rl.Size(); i++)
        {
            DataPoint dp = rl.Get(i);
            for (int j = 0; j < fids.Length; j++)
            {
                min[j] = Math.Min(min[j], dp.GetFeatureValue(fids[j]));
                max[j] = Math.Max(max[j], dp.GetFeatureValue(fids[j]));
            }
        }

        for (int i = 0; i < rl.Size(); i++)
        {
            DataPoint dp = rl.Get(i);
            for (int j = 0; j < fids.Length; j++)
            {
                if (max[j] > min[j])
                {
                    float value = (dp.GetFeatureValue(fids[j]) - min[j]) / (max[j] - min[j]);
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