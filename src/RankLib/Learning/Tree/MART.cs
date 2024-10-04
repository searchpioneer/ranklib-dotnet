using RankLib.Metric;

namespace RankLib.Learning.Tree;

public class MART : LambdaMART
{
    public MART()
    {
    }

    public MART(List<RankList> samples, int[] features, MetricScorer scorer)
        : base(samples, features, scorer)
    {
    }

    public override Ranker CreateNew()
    {
        return new MART();
    }

    public override string Name()
    {
        return "MART";
    }

    protected override void ComputePseudoResponses()
    {
        for (int i = 0; i < martSamples.Length; i++)
        {
            pseudoResponses[i] = martSamples[i].GetLabel() - modelScores[i];
        }
    }

    protected override void UpdateTreeOutput(RegressionTree rt)
    {
        var leaves = rt.Leaves();
        foreach (var s in leaves)
        {
            float s1 = 0.0F;
            int[] idx = s.GetSamples();
            foreach (int k in idx)
            {
                s1 += Convert.ToSingle(pseudoResponses[k]);
            }
            s.SetOutput(s1 / idx.Length);
        }
    }
}
