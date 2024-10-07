using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace RankLib.Metric;

public class MetricScorerFactory
{
	private readonly ILoggerFactory? _loggerFactory;

	private readonly MetricScorer[] mFactory;

	private static readonly Dictionary<string, MetricScorer> map = new();

	public MetricScorerFactory(ILoggerFactory? loggerFactory = null)
	{
		_loggerFactory = loggerFactory ?? NullLoggerFactory.Instance;

		mFactory =
		[
			new APScorer(_loggerFactory.CreateLogger<APScorer>()),
			new NDCGScorer(_loggerFactory.CreateLogger<NDCGScorer>()),
			new DCGScorer(),
			new PrecisionScorer(),
			new ReciprocalRankScorer(),
			new BestAtKScorer(),
			new ERRScorer()
		];

		map["MAP"] = new APScorer(_loggerFactory.CreateLogger<APScorer>());
		map["NDCG"] = new NDCGScorer(_loggerFactory.CreateLogger<NDCGScorer>());
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
		scorer.K = k;
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
			scorer.K = k;
		}
		else
		{
			scorer = map[metric.ToUpper()].Copy();
		}

		return scorer;
	}
}
