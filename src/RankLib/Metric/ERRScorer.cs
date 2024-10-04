using RankLib.Learning;

namespace RankLib.Metric;

public class ERRScorer : MetricScorer
{
    public static double MAX = 16; // By default, we assume the relevance scale of {0, 1, 2, 3, 4} => g_max = 4 => 2^g_max = 16

    public ERRScorer()
    {
        _k = 10;
    }

    public ERRScorer(int k)
    {
        _k = k;
    }

    public override MetricScorer Copy()
    {
        return new ERRScorer();
    }

    /// <summary>
    /// Compute ERR at k.
    /// </summary>
    public override double Score(RankList rl)
    {
        int size = _k;
        if (_k > rl.Size() || _k <= 0)
        {
            size = rl.Size();
        }

        var rel = new List<int>();
        for (int i = 0; i < rl.Size(); i++)
        {
            rel.Add((int)rl.Get(i).GetLabel());
        }

        double s = 0.0;
        double p = 1.0;
        for (int i = 1; i <= size; i++)
        {
            double r = R(rel[i - 1]);
            s += p * r / i;
            p *= (1.0 - r);
        }
        return s;
    }

    public override string Name()
    {
        return "ERR@" + _k;
    }

    private double R(int rel)
    {
        return ((1 << rel) - 1) / MAX; // (2^rel - 1)/MAX
    }

    public override double[][] SwapChange(RankList rl)
    {
        int size = (rl.Size() > _k) ? _k : rl.Size();
        int[] labels = new int[rl.Size()];
        double[] r = new double[rl.Size()];
        double[] np = new double[rl.Size()]; // p[i] = (1 - p[0])(1 - p[1])...(1 - p[i - 1])
        double p = 1.0;

        for (int i = 0; i < size; i++)
        {
            labels[i] = (int)rl.Get(i).GetLabel();
            r[i] = R(labels[i]);
            np[i] = p * (1.0 - r[i]);
            p *= np[i];
        }

        var changes = new double[rl.Size()][];
        for (int i = 0; i < rl.Size(); i++)
        {
            changes[i] = new double[rl.Size()];
            Array.Fill(changes[i], 0);
        }

        for (int i = 0; i < size; i++)
        {
            double v1 = 1.0 / (i + 1) * (i == 0 ? 1 : np[i - 1]);
            double change = 0;
            for (int j = i + 1; j < rl.Size(); j++)
            {
                if (labels[i] == labels[j])
                {
                    change = 0;
                }
                else
                {
                    change = v1 * (r[j] - r[i]);
                    p = (i == 0 ? 1 : np[i - 1]) * (r[i] - r[j]);
                    for (int k = i + 1; k < j; k++)
                    {
                        change += p * r[k] / (1 + k);
                        p *= 1.0 - r[k];
                    }
                    change += (np[j - 1] * (1.0 - r[j]) * r[i] / (1.0 - r[i]) - np[j - 1] * r[j]) / (j + 1);
                }
                changes[j][i] = changes[i][j] = change;
            }
        }
        return changes;
    }
}
