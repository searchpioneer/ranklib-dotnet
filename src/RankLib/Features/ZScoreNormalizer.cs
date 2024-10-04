using RankLib.Learning;

namespace RankLib.Features;

public class ZScoreNormalizer : Normalizer
{
    public override void Normalize(RankList rl)
    {
        if (rl.Size() == 0)
        {
            throw new InvalidOperationException("Error in ZScoreNormalizer::Normalize(): The input ranked list is empty");
        }

        int nFeature = rl.GetFeatureCount();
        double[] means = new double[nFeature];
        Array.Fill(means, 0);

        for (int i = 0; i < rl.Size(); i++)
        {
            DataPoint dp = rl.Get(i);
            for (int j = 1; j <= nFeature; j++)
            {
                means[j - 1] += dp.GetFeatureValue(j);
            }
        }

        for (int j = 1; j <= nFeature; j++)
        {
            means[j - 1] /= rl.Size();
            double std = 0;

            for (int i = 0; i < rl.Size(); i++)
            {
                DataPoint p = rl.Get(i);
                double x = p.GetFeatureValue(j) - means[j - 1];
                std += x * x;
            }

            std = Math.Sqrt(std / (rl.Size() - 1));

            if (std > 0)
            {
                for (int i = 0; i < rl.Size(); i++)
                {
                    DataPoint p = rl.Get(i);
                    double x = (p.GetFeatureValue(j) - means[j - 1]) / std; // standard normal (0, 1)
                    p.SetFeatureValue(j, (float)x);
                }
            }
        }
    }

    public override void Normalize(RankList rl, int[] fids)
    {
        if (rl.Size() == 0)
        {
            throw new InvalidOperationException("Error in ZScoreNormalizer::Normalize(): The input ranked list is empty");
        }

        // Remove duplicate features from the input fids to avoid normalizing the same features multiple times
        fids = RemoveDuplicateFeatures(fids);

        double[] means = new double[fids.Length];
        Array.Fill(means, 0);

        for (int i = 0; i < rl.Size(); i++)
        {
            DataPoint dp = rl.Get(i);
            for (int j = 0; j < fids.Length; j++)
            {
                means[j] += dp.GetFeatureValue(fids[j]);
            }
        }

        for (int j = 0; j < fids.Length; j++)
        {
            means[j] /= rl.Size();
            double std = 0;

            for (int i = 0; i < rl.Size(); i++)
            {
                DataPoint p = rl.Get(i);
                double x = p.GetFeatureValue(fids[j]) - means[j];
                std += x * x;
            }

            std = Math.Sqrt(std / (rl.Size() - 1));

            if (std > 0.0)
            {
                for (int i = 0; i < rl.Size(); i++)
                {
                    DataPoint p = rl.Get(i);
                    double x = (p.GetFeatureValue(fids[j]) - means[j]) / std; // standard normal (0, 1)
                    p.SetFeatureValue(fids[j], (float)x);
                }
            }
        }
    }

    public override string Name => "zscore";
}