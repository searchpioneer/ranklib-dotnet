namespace RankLib.Metric;

public class MetricScorerFactory
{
	private static readonly MetricScorer[] mFactory =
	[
		new APScorer(),
		new NDCGScorer(),
		new DCGScorer(),
		new PrecisionScorer(),
		new ReciprocalRankScorer(),
		new BestAtKScorer(),
		new ERRScorer()
	];

	private static readonly Dictionary<string, MetricScorer> map = new();

	public MetricScorerFactory()
	{
		map["MAP"] = new APScorer();
		map["NDCG"] = new NDCGScorer();
		map["DCG"] = new DCGScorer();
		map["P"] = new PrecisionScorer();
		map["RR"] = new ReciprocalRankScorer();
		map["BEST"] = new BestAtKScorer();
		map["ERR"] = new ERRScorer();
	}

	public MetricScorer CreateScorer(Metric metric) => mFactory[metric - Metric.MAP].Copy();

	public MetricScorer CreateScorer(Metric metric, int k)
	{
		var scorer = mFactory[metric - Metric.MAP].Copy();
		scorer.SetK(k);
		return scorer;
	}

	public MetricScorer? CreateScorer(string metric)
	{
		MetricScorer? scorer;

		if (metric.Contains('@'))
		{
			var m = metric.Substring(0, metric.IndexOf('@'));
			var k = int.Parse(metric.Substring(metric.IndexOf('@') + 1));
			scorer = map[m.ToUpper()].Copy();
			scorer.SetK(k);
		}
		else
		{
			scorer = map[metric.ToUpper()].Copy();
		}

		return scorer;
	}
}
