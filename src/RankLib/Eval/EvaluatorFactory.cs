using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Features;
using RankLib.Learning;
using RankLib.Metric;

namespace RankLib.Eval;

/// <summary>
/// Factory for creating <see cref="Evaluator"/>
/// </summary>
public class EvaluatorFactory
{
	private readonly RankerFactory _rankerFactory;
	private readonly MetricScorerFactory _metricScorerFactory;
	private readonly FeatureManager _featureManager;
	private readonly ILoggerFactory _loggerFactory;

	public EvaluatorFactory(
		RankerFactory rankerFactory,
		MetricScorerFactory metricScorerFactory,
		FeatureManager featureManager,
		ILoggerFactory? loggerFactory = null)
	{
		_rankerFactory = rankerFactory;
		_metricScorerFactory = metricScorerFactory;
		_featureManager = featureManager;
		_loggerFactory = loggerFactory ?? NullLoggerFactory.Instance;
	}

	public Evaluator CreateEvaluator(RankerType rankerType, Metric.Metric trainMetric, Metric.Metric testMetric, Normalizer? normalizer = null, string? queryRelevanceFile = null)
	{
		var trainScorer = _metricScorerFactory.CreateScorer(trainMetric);
		var testScorer = _metricScorerFactory.CreateScorer(testMetric);

		if (queryRelevanceFile != null)
		{
			trainScorer.LoadExternalRelevanceJudgment(queryRelevanceFile);
			testScorer.LoadExternalRelevanceJudgment(queryRelevanceFile);
		}

		return new Evaluator(_rankerFactory, rankerType, _featureManager, trainScorer, testScorer, normalizer, _loggerFactory);
	}

	public Evaluator CreateEvaluator(RankerType rankerType, Metric.Metric trainMetric, int trainK, Metric.Metric testMetric, int testK, Normalizer? normalizer = null, string? queryRelevanceFile = null)
	{
		var trainScorer = _metricScorerFactory.CreateScorer(trainMetric, trainK);
		var testScorer = _metricScorerFactory.CreateScorer(testMetric, testK);

		if (queryRelevanceFile != null)
		{
			trainScorer.LoadExternalRelevanceJudgment(queryRelevanceFile);
			testScorer.LoadExternalRelevanceJudgment(queryRelevanceFile);
		}

		return new Evaluator(_rankerFactory, rankerType, _featureManager, trainScorer, testScorer, normalizer, _loggerFactory);
	}

	public Evaluator CreateEvaluator(RankerType rankerType, Metric.Metric trainMetric, Metric.Metric testMetric, int k, Normalizer? normalizer = null, string? queryRelevanceFile = null)
	{
		var trainScorer = _metricScorerFactory.CreateScorer(trainMetric, k);
		var testScorer = _metricScorerFactory.CreateScorer(testMetric, k);

		if (queryRelevanceFile != null)
		{
			trainScorer.LoadExternalRelevanceJudgment(queryRelevanceFile);
			testScorer.LoadExternalRelevanceJudgment(queryRelevanceFile);
		}

		return new Evaluator(_rankerFactory, rankerType, _featureManager, trainScorer, testScorer, normalizer, _loggerFactory);
	}

	public Evaluator CreateEvaluator(RankerType rankerType, Metric.Metric metric, int k, Normalizer? normalizer = null, string? queryRelevanceFile = null)
	{
		var scorer = _metricScorerFactory.CreateScorer(metric, k);

		if (queryRelevanceFile != null)
		{
			scorer.LoadExternalRelevanceJudgment(queryRelevanceFile);
		}

		return new Evaluator(_rankerFactory, rankerType, _featureManager, scorer, normalizer, _loggerFactory);
	}

	public Evaluator CreateEvaluator(RankerType rankerType, string trainMetric, string testMetric, Normalizer? normalizer = null, string? queryRelevanceFile = null)
	{
		var trainScorer = _metricScorerFactory.CreateScorer(trainMetric);
		var testScorer = _metricScorerFactory.CreateScorer(testMetric);

		if (queryRelevanceFile != null)
		{
			trainScorer.LoadExternalRelevanceJudgment(queryRelevanceFile);
			testScorer.LoadExternalRelevanceJudgment(queryRelevanceFile);
		}

		return new Evaluator(_rankerFactory, rankerType, _featureManager, trainScorer, testScorer, normalizer, _loggerFactory);
	}
}
