using RankLib.Learning;

namespace RankLib.Metric;

public class PrecisionScorer : MetricScorer
{
    public PrecisionScorer() : base()
    {
        this._k = 10;
    }

    public PrecisionScorer(int k)
    {
        this._k = k;
    }

    public override double Score(RankList rl)
    {
        int count = 0;
        int size = _k;

        if (_k > rl.Size() || _k <= 0)
        {
            size = rl.Size();
        }

        for (int i = 0; i < size; i++)
        {
            if (rl.Get(i).GetLabel() > 0.0)
            {
                count++;
            }
        }

        return (double)count / size;
    }

    public override MetricScorer Copy()
    {
        return new PrecisionScorer(_k);
    }

    public override string Name()
    {
        return $"P@{_k}";
    }

    public override double[][] SwapChange(RankList rl)
    {
        int size = (rl.Size() > _k) ? _k : rl.Size();

        var changes = new double[rl.Size()][];
        for (int i = 0; i < rl.Size(); i++)
        {
            changes[i] = new double[rl.Size()];
            Array.Fill(changes[i], 0);
        }

        for (int i = 0; i < size; i++)
        {
            for (int j = size; j < rl.Size(); j++)
            {
                int c = GetBinaryRelevance(rl.Get(j).GetLabel()) - GetBinaryRelevance(rl.Get(i).GetLabel());
                changes[i][j] = changes[j][i] = (double)c / size;
            }
        }

        return changes;
    }

    private int GetBinaryRelevance(float label)
    {
        return label > 0.0 ? 1 : 0;
    }
}