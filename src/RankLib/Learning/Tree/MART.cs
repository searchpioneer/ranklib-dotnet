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

	public override Ranker CreateNew() => new MART();

	public override string Name() => "MART";

	protected override void ComputePseudoResponses()
	{
		for (var i = 0; i < martSamples.Length; i++)
		{
			pseudoResponses[i] = martSamples[i].Label - modelScores[i];
		}
	}

	protected override void UpdateTreeOutput(RegressionTree rt)
	{
		var leaves = rt.Leaves();
		foreach (var s in leaves)
		{
			var s1 = 0.0F;
			var idx = s.GetSamples();
			foreach (var k in idx)
			{
				s1 += Convert.ToSingle(pseudoResponses[k]);
			}
			s.SetOutput(s1 / idx.Length);
		}
	}
}
