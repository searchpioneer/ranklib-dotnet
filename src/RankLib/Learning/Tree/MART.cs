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
			PseudoResponses[i] = MARTSamples[i].Label - ModelScores[i];

		return Task.CompletedTask;
	}

	protected override void UpdateTreeOutput(RegressionTree rt)
	{
		var leaves = rt.Leaves;
		foreach (var s in leaves)
		{
			float s1 = 0;
			var idx = s.GetSamples();
			for (var i = 0; i < idx.Length; i++)
			{
				var k = idx[i];
				s1 = (float)(s1 + PseudoResponses[k]);
			}

			s.SetOutput(s1 / idx.Length);
		}
	}
}
