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
	private readonly RankerTrainer _trainer;
	private readonly ILoggerFactory _loggerFactory;

	public EvaluatorFactory(
		RankerFactory rankerFactory,
		MetricScorerFactory metricScorerFactory,
		FeatureManager featureManager,
		RankerTrainer trainer,
		ILoggerFactory? loggerFactory = null)
	{
		_rankerFactory = rankerFactory;
		_metricScorerFactory = metricScorerFactory;
		_featureManager = featureManager;
		_trainer = trainer;
		_loggerFactory = loggerFactory ?? NullLoggerFactory.Instance;
	}

	public Evaluator CreateEvaluator(
		Metric.Metric trainMetric,
		Metric.Metric testMetric,
		Normalizer? normalizer = null,
		bool mustHaveRelDoc = false,
		bool useSparseRepresentation = false,
		string? queryRelevanceFile = null)
	{
		var trainScorer = _metricScorerFactory.CreateScorer(trainMetric);
		var testScorer = _metricScorerFactory.CreateScorer(testMetric);

		if (queryRelevanceFile != null)
		{
			trainScorer.LoadExternalRelevanceJudgment(queryRelevanceFile);
			testScorer.LoadExternalRelevanceJudgment(queryRelevanceFile);
		}

		return new Evaluator(_rankerFactory, _featureManager, trainScorer, testScorer, _trainer, normalizer, mustHaveRelDoc, useSparseRepresentation, _loggerFactory);
	}

	public Evaluator CreateEvaluator(
		Metric.Metric trainMetric,
		int trainK,
		Metric.Metric testMetric,
		int testK,
		Normalizer? normalizer = null,
		bool mustHaveRelDoc = false,
		bool useSparseRepresentation = false,
		string? queryRelevanceFile = null)
	{
		var trainScorer = _metricScorerFactory.CreateScorer(trainMetric, trainK);
		var testScorer = _metricScorerFactory.CreateScorer(testMetric, testK);

		if (queryRelevanceFile != null)
		{
			trainScorer.LoadExternalRelevanceJudgment(queryRelevanceFile);
			testScorer.LoadExternalRelevanceJudgment(queryRelevanceFile);
		}

		return new Evaluator(_rankerFactory, _featureManager, trainScorer, testScorer, _trainer, normalizer, mustHaveRelDoc, useSparseRepresentation, _loggerFactory);
	}

	public Evaluator CreateEvaluator(
		Metric.Metric trainMetric,
		Metric.Metric testMetric,
		int k,
		Normalizer? normalizer = null,
		bool mustHaveRelDoc = false,
		bool useSparseRepresentation = false,
		string? queryRelevanceFile = null)
	{
		var trainScorer = _metricScorerFactory.CreateScorer(trainMetric, k);
		var testScorer = _metricScorerFactory.CreateScorer(testMetric, k);

		if (queryRelevanceFile != null)
		{
			trainScorer.LoadExternalRelevanceJudgment(queryRelevanceFile);
			testScorer.LoadExternalRelevanceJudgment(queryRelevanceFile);
		}

		return new Evaluator(_rankerFactory, _featureManager, trainScorer, testScorer, _trainer, normalizer, mustHaveRelDoc, useSparseRepresentation, _loggerFactory);
	}

	public Evaluator CreateEvaluator(
		Metric.Metric metric,
		int k,
		Normalizer? normalizer = null,
		bool mustHaveRelDoc = false,
		bool useSparseRepresentation = false,
		string? queryRelevanceFile = null)
	{
		var scorer = _metricScorerFactory.CreateScorer(metric, k);

		if (queryRelevanceFile != null)
		{
			scorer.LoadExternalRelevanceJudgment(queryRelevanceFile);
		}

		return new Evaluator(_rankerFactory, _featureManager, scorer, _trainer, normalizer, mustHaveRelDoc, useSparseRepresentation, _loggerFactory);
	}

	public Evaluator CreateEvaluator(
		string trainMetric,
		string testMetric,
		Normalizer? normalizer = null,
		bool mustHaveRelDoc = false,
		bool useSparseRepresentation = false,
		string? queryRelevanceFile = null)
	{
		var trainScorer = _metricScorerFactory.CreateScorer(trainMetric);
		var testScorer = _metricScorerFactory.CreateScorer(testMetric);

		if (queryRelevanceFile != null)
		{
			trainScorer.LoadExternalRelevanceJudgment(queryRelevanceFile);
			testScorer.LoadExternalRelevanceJudgment(queryRelevanceFile);
		}

		return new Evaluator(_rankerFactory, _featureManager, trainScorer, testScorer, _trainer, normalizer, mustHaveRelDoc, useSparseRepresentation, _loggerFactory);
	}
}
