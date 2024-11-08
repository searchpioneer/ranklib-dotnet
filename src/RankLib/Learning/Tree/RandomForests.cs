using System.Globalization;
using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Parsing;
using RankLib.Utilities;
#pragma warning disable CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider declaring as nullable.

namespace RankLib.Learning.Tree;

/// <summary>
/// Ranking parameters for <see cref="RandomForests"/>
/// </summary>
public class RandomForestsParameters : IRankerParameters
{
	private RankerType _rankerType = RankerType.MART;
	public const float DefaultFeatureSamplingRate = 0.3f;
	public const float DefaultSubSamplingRate = 1.0f;

	// Parameters
	// [a] general bagging parameters
	/// <summary>
	/// Number of bags
	/// </summary>
	public int BagCount { get; set; } = 300;

	/// <summary>
	/// Sampling of samples rate, with replacement
	/// </summary>
	public float SubSamplingRate { get; set; } = DefaultSubSamplingRate;

	/// <summary>
	/// Feature sampling rate, without replacement
	/// </summary>
	public float FeatureSamplingRate { get; set; } = DefaultFeatureSamplingRate;

	/// <summary>
	/// Ranking algorithm to use each bag. Only <see cref="RankerType.MART"/>
	/// and <see cref="RankerType.LambdaMART"/> accepted.
	/// </summary>
	/// <exception cref="ArgumentException">
	/// Thrown when ranker type is not one accepted.
	/// </exception>
	public RankerType RankerType
	{
		get => _rankerType;
		set
		{
			if (value is not (RankerType.MART or RankerType.LambdaMART))
				throw new ArgumentException($"value must be {RankerType.MART} or {RankerType.LambdaMART}", nameof(value));

			_rankerType = value;
		}
	}

	/// <summary>
	/// Number of trees in each bag
	/// </summary>
	public int TreeCount { get; set; } = 1;

	/// <summary>
	/// Number of leaves in each tree
	/// </summary>
	public int TreeLeavesCount { get; set; } = 100;

	/// <summary>
	/// The learning rate, or shrinkage, only matters if <see cref="TreeCount"/> > 1
	/// </summary>
	public float LearningRate { get; set; } = 0.1F;

	/// <summary>
	/// The number of threshold candidates.
	/// </summary>
	public int Threshold { get; set; } = 256;

	/// <summary>
	/// Minimum leaf support
	/// </summary>
	public int MinimumLeafSupport { get; set; } = 1;

	/// <summary>
	/// Gets or sets the maximum number of concurrent tasks allowed when splitting up workloads
	/// that can be run on multiple threads. If unspecified, uses the count of all available processors.
	/// </summary>
	public int MaxDegreeOfParallelism { get; set; } = Environment.ProcessorCount;

	public override string ToString()
	{
		var builder = new StringBuilder();
		builder.AppendLine($"No. of bags: {BagCount}");
		builder.AppendLine($"Sub-sampling: {SubSamplingRate}");
		builder.AppendLine($"Feature-sampling: {FeatureSamplingRate}");
		builder.AppendLine($"No. of trees: {TreeCount}");
		builder.AppendLine($"No. of leaves: {TreeLeavesCount}");
		builder.AppendLine($"No. of threshold candidates: {Threshold}");
		builder.AppendLine($"Learning rate: {LearningRate}");
		return builder.ToString();
	}
}

/// <summary>
/// Random Forests is an ensemble learning method that constructs multiple decision trees during training
/// and merges their results for more accurate and stable predictions.
/// </summary>
/// <remarks>
/// <a href="https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf">
/// L. Breiman. Random Forests. Machine Learning 45 (1): 5–32, 2001.
/// </a>
/// </remarks>
public class RandomForests : Ranker<RandomForestsParameters>
{
	internal const string RankerName = "Random Forests";

	private readonly ILoggerFactory _loggerFactory;
	private readonly ILogger<RandomForests> _logger;
	private LambdaMARTParameters _lambdaMARTParameters;

	public Ensemble[] Ensembles { get; private set; } = [];

	public override string Name => RankerName;

	public RandomForests(ILoggerFactory? loggerFactory = null)
	{
		_loggerFactory = loggerFactory ?? NullLoggerFactory.Instance;
		_logger = _loggerFactory.CreateLogger<RandomForests>();
	}

	public RandomForests(List<RankList> samples, int[] features, MetricScorer scorer, ILoggerFactory? loggerFactory = null)
		: base(samples, features, scorer)
	{
		_loggerFactory = loggerFactory ?? NullLoggerFactory.Instance;
		_logger = _loggerFactory.CreateLogger<RandomForests>();
	}

	public override Task InitAsync()
	{
		_logger.LogInformation("Initializing...");
		Ensembles = new Ensemble[Parameters.BagCount];
		_lambdaMARTParameters = new LambdaMARTParameters
		{
			TreeCount = Parameters.TreeCount,
			TreeLeavesCount = Parameters.TreeLeavesCount,
			LearningRate = Parameters.LearningRate,
			Threshold = Parameters.Threshold,
			MinimumLeafSupport = Parameters.MinimumLeafSupport,
			// no early stopping since we're doing bagging
			StopEarlyRoundCount = -1,
			// Turn on feature sampling
			SamplingRate = Parameters.FeatureSamplingRate,
			MaxDegreeOfParallelism = Parameters.MaxDegreeOfParallelism
		};

		return Task.CompletedTask;
	}

	public override async Task LearnAsync()
	{
		var rankerFactory = new RankerFactory(_loggerFactory);
		_logger.LogInformation("Training starts...");
		_logger.PrintLog([9, 9, 11], ["bag", Scorer.Name + "-B", Scorer.Name + "-OOB"]);

		double[]? impacts = null;

		// Start the bagging process
		for (var i = 0; i < Parameters.BagCount; i++)
		{
			// Create a "bag" of samples by random sampling from the training set
			var (bag, _) = Sampler.Sample(Samples, Parameters.SubSamplingRate, true);
			var ranker = (LambdaMART)rankerFactory.CreateRanker(Parameters.RankerType, bag, Features, Scorer);

			ranker.Parameters = _lambdaMARTParameters;
			await ranker.InitAsync().ConfigureAwait(false);
			await ranker.LearnAsync().ConfigureAwait(false);

			// Accumulate impacts
			if (impacts == null)
				impacts = ranker.Impacts;
			else
			{
				for (var ftr = 0; ftr < impacts.Length; ftr++)
					impacts[ftr] += ranker.Impacts[ftr];
			}
			_logger.PrintLog([9, 9], ["b[" + (i + 1) + "]", SimpleMath.Round(ranker.GetTrainingDataScore(), 4).ToString(CultureInfo.InvariantCulture)]);
			Ensembles[i] = ranker.Ensemble;
		}

		// Finishing up
		TrainingDataScore = Scorer.Score(Rank(Samples));
		_logger.LogInformation("Finished successfully.");
		_logger.LogInformation(Scorer.Name + " on training data: " + SimpleMath.Round(TrainingDataScore, 4));

		if (ValidationSamples != null)
		{
			ValidationDataScore = Scorer.Score(Rank(ValidationSamples));
			_logger.LogInformation(Scorer.Name + " on validation data: " + SimpleMath.Round(ValidationDataScore, 4));
		}

		// Print feature impacts
		_logger.LogInformation("-- FEATURE IMPACTS");
		if (_logger.IsEnabled(LogLevel.Information))
		{
			var ftrsSorted = MergeSorter.Sort(impacts!, false);
			foreach (var ftr in ftrsSorted)
				_logger.LogInformation(" Feature " + Features[ftr] + " reduced error " + impacts![ftr]);
		}
	}

	public override double Eval(DataPoint dataPoint)
	{
		double s = 0;
		foreach (var ensemble in Ensembles)
			s += ensemble.Eval(dataPoint);

		return s / Ensembles.Length;
	}

	public override string GetModel()
	{
		var output = new StringBuilder();
		output.Append($"## {Name}\n");
		output.Append($"## No. of bags = {Parameters.BagCount}\n");
		output.Append($"## Sub-sampling = {Parameters.SubSamplingRate}\n");
		output.Append($"## Feature-sampling = {Parameters.FeatureSamplingRate}\n");
		output.Append($"## No. of trees = {Parameters.TreeCount}\n");
		output.Append($"## No. of leaves = {Parameters.TreeLeavesCount}\n");
		output.Append($"## No. of threshold candidates = {Parameters.Threshold}\n");
		output.Append($"## Learning rate = {Parameters.LearningRate}\n\n");
		output.Append(ToString());

		for (var i = 0; i < Parameters.BagCount; i++)
			output.Append(Ensembles[i]).Append('\n');

		return output.ToString();
	}

	public override void LoadFromString(string model)
	{
		var ensembles = new List<Ensemble>();
		var lineByLine = new ModelLineProducer();

		lineByLine.Parse(model, (builder, maybeEndEns) =>
		{
			if (maybeEndEns && builder.ToString().EndsWith("</ensemble>"))
			{
				ensembles.Add(Ensemble.Parse(builder.ToString()));
				builder.Clear();
			}
		});

		var uniqueFeatures = new HashSet<int>();
		Ensembles = new Ensemble[ensembles.Count];
		for (var i = 0; i < ensembles.Count; i++)
		{
			Ensembles[i] = ensembles[i];

			// Obtain used features
			var fids = ensembles[i].Features;
			foreach (var fid in fids)
				uniqueFeatures.Add(fid);
		}

		Features = uniqueFeatures.ToArray();
	}
}
