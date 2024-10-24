using Microsoft.Extensions.Logging;
using RankLib.Metric;

namespace RankLib.Learning.Tree;

/// <summary>
/// MART (Multiple Additive Regression Trees) is an ensemble learning method
/// that combines multiple regression trees to improve prediction accuracy,
/// typically used in boosting frameworks to iteratively correct errors from
/// previous trees and optimize (point-wise) ranking or regression tasks.
/// </summary>
/// <remarks>
/// <a href="https://jerryfriedman.su.domains/ftp/trebst.pdf">
/// J.H. Friedman. Greedy function approximation: A gradient boosting machine.
/// Technical Report, IMS Reitz Lecture, Stanford, 1999; see also Annals of Statistics, 2001.
/// </a>
/// </remarks>
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

	protected override void UpdateTreeOutput(RegressionTree tree)
	{
		foreach (var split in tree.Leaves)
		{
			float s1 = 0;
			var idx = split.GetSamples();
			for (var i = 0; i < idx.Length; i++)
			{
				var k = idx[i];
				s1 = (float)(s1 + PseudoResponses[k]);
			}

			split.Output = s1 / idx.Length;
		}
	}
}
