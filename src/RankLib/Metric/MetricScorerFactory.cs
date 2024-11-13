using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Utilities;

namespace RankLib.Metric;

/// <summary>
/// Factory for creating <see cref="MetricScorer"/> instances.
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

	/// <summary>
	/// Initializes a new instance of <see cref="MetricScorerFactory"/>
	/// </summary>
	/// <param name="loggerFactory">The logger factory used to create loggers for created <see cref="MetricScorer"/></param>
	public MetricScorerFactory(ILoggerFactory? loggerFactory = null) =>
		_loggerFactory = loggerFactory ?? NullLoggerFactory.Instance;

	/// <summary>
	/// Creates a new instance of a <see cref="MetricScorer"/> from a metric and k value.
	/// </summary>
	/// <param name="metric">The metric</param>
	/// <returns>A new instance of a <see cref="MetricScorer"/></returns>
	/// <exception cref="ArgumentOutOfRangeException">The metric is not in the range of valid values.</exception>
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

	/// <summary>
	/// Creates a new instance of a <see cref="MetricScorer"/> from a metric and k value.
	/// </summary>
	/// <param name="metric">The metric</param>
	/// <param name="k">The k value</param>
	/// <returns>A new instance of a <see cref="MetricScorer"/></returns>
	public MetricScorer CreateScorer(Metric metric, int k)
	{
		var scorer = CreateScorer(metric);
		scorer.K = k;
		return scorer;
	}

	/// <summary>
	/// Creates a new instance of a <see cref="MetricScorer"/> from a metric string value.
	/// </summary>
	/// <param name="metric">The metric string value</param>
	/// <returns>A new instance of a <see cref="MetricScorer"/></returns>
	/// <exception cref="ArgumentException">The metric string value does not represent a known metric.</exception>
	public MetricScorer CreateScorer(string metric)
	{
		MetricScorer scorer;
		var metricSpan = metric.AsSpan();
		if (metricSpan.Contains('@'))
		{
			var atIndex = metricSpan.IndexOf('@');
			var metricName = metricSpan[..atIndex].ToString();
			var k = int.Parse(metricSpan[(atIndex + 1)..]);

			if (!MetricNames.TryGetValue(metricName, out var value))
				throw new ArgumentException($"Could not create scorer for metric '{metric}'", nameof(metric));

			scorer = CreateScorer(value);
			scorer.K = k;
		}
		else
		{
			if (!MetricNames.TryGetValue(metric, out var value))
				throw new ArgumentException($"Could not create scorer for metric '{metric}'", nameof(metric));

			scorer = CreateScorer(value);
		}

		return scorer;
	}
}
