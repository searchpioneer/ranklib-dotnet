using RankLib.Learning;

namespace RankLib.Metric;

public class BestAtKScorer : MetricScorer
{
    public BestAtKScorer()
    {
        _k = 10;
    }

    public BestAtKScorer(int k)
    {
        _k = k;
    }

    public override double Score(RankList rl)
    {
        return rl.Get(MaxToK(rl, _k - 1)).GetLabel();
    }

    public override MetricScorer Copy()
    {
        return new BestAtKScorer();
    }

    /// <summary>
    /// Return the position of the best object (e.g., docs with highest degree of relevance) among objects in the range [0..k].
    /// NOTE: If you want best-at-k (i.e., best among top-k), you need MaxToK(rl, k-1).
    /// </summary>
    /// <param name="rl">The rank list.</param>
    /// <param name="k">The last position of the range.</param>
    /// <returns>The index of the best object in the specified range.</returns>
    public int MaxToK(RankList rl, int k)
    {
        int size = k;
        if (size < 0 || size > rl.Size() - 1)
        {
            size = rl.Size() - 1;
        }

        double max = -1.0;
        int max_i = 0;

        for (int i = 0; i <= size; i++)
        {
            if (max < rl.Get(i).GetLabel())
            {
                max = rl.Get(i).GetLabel();
                max_i = i;
            }
        }
        return max_i;
    }

    public override string Name()
    {
        return "Best@" + _k;
    }

    public override double[][] SwapChange(RankList rl)
    {
        //FIXME: Not sure if this implementation is correct!
        int[] labels = new int[rl.Size()];
        int[] best = new int[rl.Size()];
        int max = -1;
        int maxVal = -1;
        int secondMaxVal = -1; // within top-K
        int maxCount = 0; // within top-K

        for (int i = 0; i < rl.Size(); i++)
        {
            int v = (int)rl.Get(i).GetLabel();
            labels[i] = v;

            if (maxVal < v)
            {
                if (i < _k)
                {
                    secondMaxVal = maxVal;
                    maxCount = 0;
                }
                maxVal = v;
                max = i;
            }
            else if (maxVal == v && i < _k)
            {
                maxCount++;
            }
            best[i] = max;
        }

        if (secondMaxVal == -1)
        {
            secondMaxVal = 0;
        }

        double[][] changes = new double[rl.Size()][];
        for (int i = 0; i < rl.Size(); i++)
        {
            changes[i] = new double[rl.Size()];
            Array.Fill(changes[i], 0);
        }

        //FIXME: THIS IS VERY *INEFFICIENT*
        for (int i = 0; i < rl.Size() - 1; i++)
        {
            for (int j = i + 1; j < rl.Size(); j++)
            {
                double change = 0;
                if (j < _k || i >= _k)
                {
                    change = 0;
                }
                else if (labels[i] == labels[j] || labels[j] == labels[best[_k - 1]])
                {
                    change = 0;
                }
                else if (labels[j] > labels[best[_k - 1]])
                {
                    change = labels[j] - labels[best[i]];
                }
                else if (labels[i] < labels[best[_k - 1]] || maxCount > 1)
                {
                    change = 0;
                }
                else
                {
                    change = maxVal - Math.Max(secondMaxVal, labels[j]);
                }
                changes[i][j] = changes[j][i] = change;
            }
        }
        return changes;
    }
}
