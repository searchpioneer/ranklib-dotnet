using Microsoft.Extensions.Logging;
using RankLib.Metric;

namespace RankLib.Learning.Tree;

public class MART : LambdaMART
{
	internal new const string RankerName = "MART";

	public MART(ILogger<MART>? logger = null) : base(logger)
	{
	}

	public MART(List<RankList> samples, int[] features, MetricScorer scorer, ILogger<MART>? logger = null)
		: base(samples, features, scorer, logger)
	{
	}

	public override string Name => RankerName;

	protected override Task ComputePseudoResponses()
	{
		for (var i = 0; i < MARTSamples.Length; i++)
		{
			PseudoResponses[i] = MARTSamples[i].Label - ModelScores[i];
		}

		return Task.CompletedTask;
	}

	protected override void UpdateTreeOutput(RegressionTree tree)
	{
		var leaves = tree.Leaves;
		foreach (var s in leaves)
		{
			var s1 = 0.0F;
			var idx = s.GetSamples();
			foreach (var k in idx)
			{
				s1 += Convert.ToSingle(PseudoResponses[k]);
			}
			s.SetOutput(s1 / idx.Length);
		}
	}
}
