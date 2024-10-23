using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Utilities;

namespace RankLib.Metric;

/// <summary>
/// Factory for creating <see cref="MetricScorer"/>
/// </summary>
public class MetricScorerFactory
{
	private static readonly Dictionary<string, Metric> MetricNames = new(StringComparer.OrdinalIgnoreCase)
	{
		["MAP"] = Metric.MAP,
		["NDCG"] = Metric.NDCG,
		["DCG"] = Metric.DCG,
		["P"] = Metric.Precision,
		["RR"] = Metric.Reciprocal,
		["BEST"] = Metric.Best,
		["ERR"] = Metric.ERR,
	};

	private readonly ILoggerFactory _loggerFactory;

	public MetricScorerFactory(ILoggerFactory? loggerFactory = null) =>
		_loggerFactory = loggerFactory ?? NullLoggerFactory.Instance;

	public MetricScorer CreateScorer(Metric metric) =>
		metric switch
		{
			Metric.MAP => new APScorer(_loggerFactory.CreateLogger<APScorer>()),
			Metric.NDCG => new NDCGScorer(_loggerFactory.CreateLogger<NDCGScorer>()),
			Metric.DCG => new DCGScorer(),
			Metric.Precision => new PrecisionScorer(),
			Metric.Reciprocal => new ReciprocalRankScorer(),
			Metric.Best => new BestAtKScorer(),
			Metric.ERR => new ERRScorer(),
			_ => throw new ArgumentOutOfRangeException(nameof(metric), metric, null)
		};

	public MetricScorer CreateScorer(Metric metric, int k)
	{
		var scorer = CreateScorer(metric);
		scorer.K = k;
		return scorer;
	}

	public MetricScorer CreateScorer(string metric)
	{
		MetricScorer scorer;
		var metricSpan = metric.AsSpan();
		if (metricSpan.Contains('@'))
		{
			var atIndex = metricSpan.IndexOf('@');
			var m = metricSpan.Slice(0, atIndex).ToString();
			var k = int.Parse(metricSpan.Slice(atIndex + 1));

			if (!MetricNames.TryGetValue(m, out var value))
				throw RankLibException.Create($"Could not create scorer for metric '{metric}'");

			scorer = CreateScorer(value);
			scorer.K = k;
		}
		else
		{
			if (!MetricNames.TryGetValue(metric, out var value))
				throw RankLibException.Create($"Could not create scorer for metric '{metric}'");

			scorer = CreateScorer(value);
		}

		return scorer;
	}
}
