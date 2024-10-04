using RankLib.Learning;

namespace RankLib.Metric;

public class ReciprocalRankScorer : MetricScorer
{
    public ReciprocalRankScorer()
    {
        this._k = 0; // consider the whole list
    }

    public override double Score(RankList rl)
    {
        int size = (rl.Size() > _k) ? _k : rl.Size();
        int firstRank = -1;

        for (int i = 0; i < size && firstRank == -1; i++)
        {
            if (rl.Get(i).GetLabel() > 0.0)
            {
                firstRank = i + 1;
            }
        }

        return (firstRank == -1) ? 0 : (1.0 / firstRank);
    }

    public override MetricScorer Copy()
    {
        return new ReciprocalRankScorer();
    }

    public override string Name()
    {
        return $"RR@{_k}";
    }

    public override double[][] SwapChange(RankList rl)
    {
        int firstRank = -1;
        int secondRank = -1;
        int size = (rl.Size() > _k) ? _k : rl.Size();

        for (int i = 0; i < size; i++)
        {
            if (rl.Get(i).GetLabel() > 0.0)
            {
                if (firstRank == -1)
                {
                    firstRank = i;
                }
                else if (secondRank == -1)
                {
                    secondRank = i;
                }
            }
        }

        // Initialize changes array
        var changes = new double[rl.Size()][];
        for (int i = 0; i < rl.Size(); i++)
        {
            changes[i] = new double[rl.Size()];
            Array.Fill(changes[i], 0);
        }

        // Calculate change in Reciprocal Rank (RR)
        double rr = 0.0;
        if (firstRank != -1)
        {
            rr = 1.0 / (firstRank + 1);
            for (int j = firstRank + 1; j < size; j++)
            {
                if (rl.Get(j).GetLabel() == 0)
                {
                    if (secondRank == -1 || j < secondRank)
                    {
                        changes[firstRank][j] = changes[j][firstRank] = 1.0 / (j + 1) - rr;
                    }
                    else
                    {
                        changes[firstRank][j] = changes[j][firstRank] = 1.0 / (secondRank + 1) - rr;
                    }
                }
            }

            for (int j = size; j < rl.Size(); j++)
            {
                if (rl.Get(j).GetLabel() == 0)
                {
                    if (secondRank == -1)
                    {
                        changes[firstRank][j] = changes[j][firstRank] = -rr;
                    }
                    else
                    {
                        changes[firstRank][j] = changes[j][firstRank] = 1.0 / (secondRank + 1) - rr;
                    }
                }
            }
        }
        else
        {
            firstRank = size;
        }

        // Consider swapping documents below firstRank with those earlier in the list
        for (int i = 0; i < firstRank; i++)
        {
            for (int j = firstRank; j < rl.Size(); j++)
            {
                if (rl.Get(j).GetLabel() > 0)
                {
                    changes[i][j] = changes[j][i] = 1.0 / (i + 1) - rr;
                }
            }
        }

        return changes;
    }
}