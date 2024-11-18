using System.CommandLine;
using System.Runtime.CompilerServices;
using Microsoft.Extensions.Logging;
using RankLib.Eval;
using RankLib.Features;
using RankLib.Learning;
using RankLib.Learning.Boosting;
using RankLib.Learning.NeuralNet;
using RankLib.Learning.Tree;
using RankLib.Metric;
using RankLib.Utilities;
#pragma warning disable CS8604 // Possible null reference argument.
#pragma warning disable CS8602 // Dereference of a possibly null reference.

namespace RankLib.Cli;

public class EvaluateCommandOptions : ICommandOptions
{
	public FileInfo? TrainInputFile { get; set; }
	public RankerType Ranker { get; set; }
	public FileInfo? FeatureDescriptionInputFile { get; set; }
	public string TrainMetric { get; set; } = default!;
	public string? TestMetric { get; set; }
	public double? MaxLabel { get; set; }
	public FileInfo? QueryRelevanceInputFile { get; set; }
	public float TrainTestSplit { get; set; }
	public float TrainValidationSplit { get; set; }
	public int CrossValidationFolds { get; set; }
	public DirectoryInfo? CrossValidationOutputDirectory { get; set; }
	public string? CrossValidationModelName { get; set; }
	public FileInfo? ValidateFile { get; set; }
	public IEnumerable<FileInfo>? TestInputFiles { get; set; }
	public NormalizerType? Norm { get; set; }
	public FileInfo? ModelOutputFile { get; set; }
	public IEnumerable<FileInfo>? ModelInputFiles { get; set; }
	public int? Thread { get; set; }
	public FileInfo? RankInputFile { get; set; }
	public FileInfo? IndriRankingOutputFile { get; set; }
	public bool UseSparseRepresentation { get; set; }
	public bool MissingZero { get; set; }
	public FileInfo? IndividualRanklistPerformanceOutputFile { get; set; }
	public FileInfo? Score { get; set; }
	public int? Epoch { get; set; }
	public int? Layer { get; set; }
	public int? Node { get; set; }
	public double? LearningRate { get; set; }
	public int? ThresholdCandidates { get; set; }
	public bool? NoTrainEnqueue { get; set; }
	public int? MaxSelections { get; set; }
	public int? RandomRestarts { get; set; }
	public int? Iterations { get; set; }
	public int? Round { get; set; }
	public double? Regularization { get; set; }
	public double? Tolerance { get; set; }
	public int? Tree { get; set; }
	public int? Leaf { get; set; }
	public float? Shrinkage { get; set; }
	public int? MinimumLeafSupport { get; set; }
	public int? EarlyStop { get; set; }
	public int? Bag { get; set; }
	public float? SubSamplingRate { get; set; }
	public float? FeatureSamplingRate { get; set; }
	public RankerType? RandomForestsRanker { get; set; }
	public double? L2 { get; set; }
	public bool MustHaveRelevantDocs { get; set; }
	public int? RandomSeed { get; set; }
}

public class EvaluateCommand : Command<EvaluateCommandOptions, EvaluateCommandOptionsHandler>
{
	public EvaluateCommand()
	: base("eval", "Trains and evaluates a ranker, or evaluates a previously saved ranker model.")
	{
		AddOption(new Option<FileInfo>(["-train", "--train-input-file"], "Training data file").ExistingOnly());
		AddOption(new Option<RankerType>(["-ranker", "--ranker"], () => RankerType.CoordinateAscent, "Ranking algorithm to use"));
		AddOption(new Option<FileInfo>(["-feature", "--feature-description-input-file"], "Feature description file. list features to be considered by the learner, each on a separate line. If not specified, all features will be used.").ExistingOnly());
		AddOption(new Option<string>(["-metric2t", "--train-metric"], () => "ERR@10", "Metric to optimize on the training data. Supports MAP, NDCG@k, DCG@k, P@k, RR@k, ERR@k."));
		AddOption(new Option<double?>(["-gmax", "--max-label"], $"Highest judged relevance label. It affects the calculation of ERR i.e. 5-point scale [0,1,2,3,4] where value used is 2^gmax [default: {ERRScorer.DefaultMax}"));
		AddOption(new Option<FileInfo>(["-qrel", "--query-relevance-input-file"], "TREC-style relevance judgment file").ExistingOnly());
		AddOption(new Option<bool>(["-missingZero", "--missing-zero"], "Substitute zero for missing feature values rather than throwing an exception."));
		AddOption(new Option<FileInfo>(["-validate", "--validate-file"], "Specify if you want to tune your system on the validation data").ExistingOnly());
		AddOption(new Option<float>(["-tvs", "--train-validation-split"], "If you don't have separate validation data, use this to set train-validation split to be. Must be between 0 and 1, where train=<value> and validate=(1.0 - <value>)"));
		AddOption(new Option<FileInfo>(["-save", "--model-output-file"], "Save the model learned [default: no save]"));

		AddOption(new Option<IEnumerable<FileInfo>>(["-test", "--test-input-files"], "Specify if you want to evaluate the trained model on this data").ExistingOnly());
		AddOption(new Option<float>(["-tts", "--train-test-split"], "Set train-test split. Must be between 0 and 1, where train=<value> and test=(1.0 - <value>). Overrides --train-validation-split (-tvs)"));
		AddOption(new Option<string>(["-metric2T", "--test-metric"], "Metric to evaluate on the test data [default: same as -metric2t]"));
		AddOption(new Option<NormalizerType>(["-norm", "--norm"], "Type of normalizer to use to normalize all feature vectors [default: no normalization]"));

		AddOption(new Option<int>(["-kcv", "--cross-validation-folds"], () => -1, "Specify how many folds to perform for k-fold cross validation using the specified training data. Defaults to no k-fold cross validation."));
		AddOption(new Option<DirectoryInfo>(["-kcvmd", "--cross-validation-output-directory"], "Directory for models trained via cross-validation"));
		AddOption(new Option<string>(["-kcvmn", "--cross-validation-model-name"], "Name for model learned in each fold. It will be prefix-ed with the fold-number"));

		AddOption(new Option<IEnumerable<FileInfo>>(["-load", "--model-input-files"], "Load saved model file for evaluation").ExistingOnly());
		AddOption(new Option<int?>(["-thread", "--thread"], "Number of threads to use. If unspecified, will use all available processors"));
		AddOption(new Option<FileInfo>(["-rank", "--rank-input-file"], "Rank the samples in the specified file (specify either this or -test but not both)").ExistingOnly());
		AddOption(new Option<FileInfo>(["-indri", "--rank-output-file"], "Indri ranking file").ExistingOnly());
		AddOption(new Option<bool>(["-sparse", "--use-sparse-representation"], "Use data points with sparse representation"));
		AddOption(new Option<FileInfo>(["-idv", "--individual-ranklist-performance-output-file"], "Individual rank list model performance (in test metric). Has to be used with -test").ExistingOnly());
		AddOption(new Option<FileInfo>(["-score", "--score"], "Store ranker's score for each object being ranked. Has to be used with -rank"));

		// RankNet specific parameters
		AddOption(new Option<int?>(["-epoch", "--epoch"], "RankNet parameter: The number of epochs to train"));
		AddOption(new Option<int?>(["-layer", "--layer"], "RankNet parameter: The number of hidden layers"));
		AddOption(new Option<int?>(["-node", "--node"], "RankNet parameter: The number of hidden nodes per layer"));
		AddOption(new Option<double?>(["-lr", "--learning-rate"], "RankNet parameter: Learning rate"));

		// MART / LambdaMART / RankBoost specific parameters
		AddOption(new Option<int?>(["-tc", "--threshold-candidates"], "MART|LambdaMART|RankBoost parameter: Number of threshold candidates to search or for tree splitting. -1 to use all feature values"));

		// RankBoost / AdaRank specific parameters
		AddOption(new Option<int?>(["-round", "--round"], $"RankBoost|AdaRank parameter: The number of rounds to train [default: RankBoost:{RankBoostParameters.DefaultIterationCount}, AdaRank:{AdaRankParameters.DefaultIterationCount}]"));

		// AdaRank specific parameters
		AddOption(new Option<bool?>(["-noeq", "--no-train-enqueue"], $"AdaRank parameter: Train without enqueuing too-strong features. [default: {AdaRankParameters.DefaultTrainWithEnqueue}]"));
		AddOption(new Option<int?>(["-max", "--max-selections"], $"AdaRank parameter: The maximum number of times can a feature be consecutively selected without changing performance [default: {AdaRankParameters.DefaultMaximumSelectedCount}]"));

		// AdaRank / Coordinate Ascent specific parameters
		AddOption(new Option<double?>(["-tolerance", "--tolerance"], $"AdaRank|CoordinateAscent parameter: Tolerance between two consecutive rounds of learning. [default: to AdaRank:{AdaRankParameters.DefaultTolerance}, CoordinateAscent:{CoordinateAscentParameters.DefaultTolerance}]"));

		// Coordinate Ascent specific parameters
		AddOption(new Option<int?>(["-r", "--random-restarts"], $"CoordinateAscent parameter: The number of random restarts [default: {CoordinateAscentParameters.DefaultRandomRestartCount}]"));
		AddOption(new Option<int?>(["-i", "--iterations"], $"CoordinateAscent parameter: The number of iterations to search in each dimension [default: {CoordinateAscentParameters.DefaultMaximumIterationCount}]"));
		AddOption(new Option<double?>(["-reg", "--regularization"], "CoordinateAscent parameter: Regularization parameter [default: no regularization]"));

		// MART / LambdaMART specific parameters
		AddOption(new Option<int?>(["-tree", "--tree"], $"MART|LambdaMART parameter: Number of trees [default: {LambdaMARTParameters.DefaultTreeCount}]"));
		AddOption(new Option<int?>(["-leaf", "--leaf"], $"MART|LambdaMART parameter: Number of leaves for each tree [default: {LambdaMARTParameters.DefaultTreeLeavesCount}]"));
		AddOption(new Option<float?>(["-shrinkage", "--shrinkage"], $"MART|LambdaMART parameter: Shrinkage, or learning rate [default: {LambdaMARTParameters.DefaultLearningRate}"));
		AddOption(new Option<int?>(["-mls", "--minimum-leaf-support"], $"MART|LambdaMART parameter: Minimum leaf support. Minimum number of samples each leaf has to contain [default: {LambdaMARTParameters.DefaultMinimumLeafSupport}]"));
		AddOption(new Option<int?>(["-estop", "--early-stop"], $"MART|LambdaMART parameter: Stop early when no improvement is observed on validation data in e consecutive rounds [default: {LambdaMARTParameters.DefaultStopEarlyRoundCount}"));

		// Random Forests specific parameters
		AddOption(new Option<int?>(["-bag", "--bag"], $"RandomForests parameter: Number of bags [default: {RandomForestsParameters.DefaultBagCount}]"));
		AddOption(new Option<float?>(["-srate", "--sub-sampling-rate"], $"RandomForests parameter: Sub-sampling rate [default: {RandomForestsParameters.DefaultSubSamplingRate}]"));
		AddOption(new Option<float?>(["-frate", "--feature-sampling-rate"], $"RandomForests parameter: Feature sampling rate [default: {RandomForestsParameters.DefaultFeatureSamplingRate}]"));
		AddOption(new Option<string>(["-rtype", "--random-forests-ranker"], $"RandomForests parameter: Ranker type to bag. Random Forests only support MART/LambdaMART [default: {RandomForestsParameters.DefaultRankerType}]")
			.FromAmong(RankerType.MART.ToString(), RankerType.LambdaMART.ToString()));


		AddOption(new Option<double?>(["-L2", "--l2"], $"LinearRegression parameter: L2-norm regularization parameter [default: {LinearRegressionParameters.DefaultLambda}]"));
		AddOption(new Option<bool?>(["-hr", "--must-have-relevant-docs"], () => false, "Whether to ignore ranked list without any relevant document"));
		AddOption(new Option<int?>(
			"--random-seed",
			"A seed to use for random number generation. This is useful for internal " +
			"testing purposes and should not be used for production.")
		{ IsHidden = true });
	}
}

public class EvaluateCommandOptionsHandler : ICommandOptionsHandler<EvaluateCommandOptions>
{
	private readonly ILoggerFactory _loggerFactory;
	private readonly EvaluatorFactory _evaluatorFactory;
	private LambdaMARTParameters? _lambdaMARTParameters;
	private RankNetParameters? _rankNetParameters;
	private ListNetParameters? _listNetParameters;
	private RankBoostParameters? _rankBoostParameters;
	private RandomForestsParameters? _randomForestsParameters;
	private AdaRankParameters? _adaRankParameters;
	private CoordinateAscentParameters? _coordinateAscentParameters;
	private LinearRegressionParameters? _linearRegressionParameters;

	public EvaluateCommandOptionsHandler(ILoggerFactory loggerFactory, EvaluatorFactory evaluatorFactory)
	{
		_loggerFactory = loggerFactory;
		_evaluatorFactory = evaluatorFactory;
	}

	private LambdaMARTParameters LambdaMARTParameters => _lambdaMARTParameters ??= new LambdaMARTParameters();
	private RankNetParameters RankNetParameters => _rankNetParameters ??= new RankNetParameters();
	private ListNetParameters ListNetParameters => _listNetParameters ??= new ListNetParameters();
	private RankBoostParameters RankBoostParameters => _rankBoostParameters ??= new RankBoostParameters();
	private RandomForestsParameters RandomForestsParameters => _randomForestsParameters ??= new RandomForestsParameters();
	private AdaRankParameters AdaRankParameters => _adaRankParameters ??= new AdaRankParameters();
	private CoordinateAscentParameters CoordinateAscentParameters => _coordinateAscentParameters ??= new CoordinateAscentParameters();
	private LinearRegressionParameters LinearRegressionParameters => _linearRegressionParameters ??= new LinearRegressionParameters();

	public async Task<int> HandleAsync(EvaluateCommandOptions options, CancellationToken cancellationToken)
	{
		if (options.RandomSeed != null)
			ThreadsafeSeedableRandom.Seed = options.RandomSeed.Value;

		var logger = _loggerFactory.CreateLogger<Evaluator>();

		var trainMetric = options.TrainMetric;
		var testMetric = !string.IsNullOrEmpty(options.TestMetric)
			? options.TestMetric
			: trainMetric;

		var trainFile = options.TrainInputFile;
		var testFiles = options.TestInputFiles != null
			? options.TestInputFiles.Select(f => f.FullName).ToList()
			: [];
		var validationFile = options.ValidateFile;
		var savedModelFiles = options.ModelInputFiles != null
			? options.ModelInputFiles.Select(f => f.FullName).ToList()
			: [];

		var rankFile = options.RankInputFile;

		var tvSplit = options.TrainValidationSplit;
		var ttSplit = options.TrainTestSplit;

		var foldCount = options.CrossValidationFolds;
		var kcvModelDir = options.CrossValidationOutputDirectory;
		var kcvModelFile = options.CrossValidationModelName;

		var featureDescriptionFile = options.FeatureDescriptionInputFile;
		var indriRankingFile = options.IndriRankingOutputFile;
		var prpFile = options.IndividualRanklistPerformanceOutputFile;
		var scoreFile = options.Score;

		if (options.MissingZero)
			DataPoint.MissingZero = true;

		if (options.Epoch != null)
		{
			RankNetParameters.IterationCount = options.Epoch.Value;
			ListNetParameters.IterationCount = options.Epoch.Value;
		}

		if (options.Layer != null)
			RankNetParameters.HiddenLayerCount = options.Layer.Value;

		if (options.Node != null)
			RankNetParameters.HiddenNodePerLayerCount = options.Node.Value;

		if (options.LearningRate != null)
		{
			RankNetParameters.LearningRate = options.LearningRate.Value;
			ListNetParameters.LearningRate = Neuron.DefaultLearningRate;
		}

		if (options.ThresholdCandidates != null)
		{
			RankBoostParameters.Threshold = options.ThresholdCandidates.Value;
			LambdaMARTParameters.Threshold = options.ThresholdCandidates.Value;
		}

		if (options.NoTrainEnqueue != null)
			AdaRankParameters.TrainWithEnqueue = false;

		if (options.MaxSelections != null)
			AdaRankParameters.MaximumSelectedCount = options.MaxSelections.Value;

		if (options.RandomRestarts != null)
			CoordinateAscentParameters.RandomRestartCount = options.RandomRestarts.Value;

		if (options.Iterations != null)
			CoordinateAscentParameters.MaximumIterationCount = options.Iterations.Value;

		if (options.Round != null)
		{
			RankBoostParameters.IterationCount = options.Round.Value;
			AdaRankParameters.IterationCount = options.Round.Value;
		}

		if (options.Regularization != null)
		{
			CoordinateAscentParameters.Slack = options.Regularization.Value;
			CoordinateAscentParameters.Regularized = true;
		}

		if (options.Tolerance != null)
		{
			AdaRankParameters.Tolerance = options.Tolerance.Value;
			CoordinateAscentParameters.Tolerance = options.Tolerance.Value;
		}

		if (options.Tree != null)
		{
			LambdaMARTParameters.TreeCount = options.Tree.Value;
			RandomForestsParameters.TreeCount = LambdaMARTParameters.TreeCount;
		}

		if (options.Leaf != null)
		{
			LambdaMARTParameters.TreeLeavesCount = options.Leaf.Value;
			RandomForestsParameters.TreeLeavesCount = LambdaMARTParameters.TreeLeavesCount;
		}

		if (options.Shrinkage != null)
		{
			LambdaMARTParameters.LearningRate = options.Shrinkage.Value;
			RandomForestsParameters.LearningRate = LambdaMARTParameters.LearningRate;
		}

		if (options.MinimumLeafSupport != null)
		{
			LambdaMARTParameters.MinimumLeafSupport = options.MinimumLeafSupport.Value;
			RandomForestsParameters.MinimumLeafSupport = LambdaMARTParameters.MinimumLeafSupport;
		}

		if (options.EarlyStop != null)
			LambdaMARTParameters.StopEarlyRoundCount = options.EarlyStop.Value;

		if (options.Bag != null)
			RandomForestsParameters.BagCount = options.Bag.Value;

		if (options.SubSamplingRate != null)
			RandomForestsParameters.SubSamplingRate = options.SubSamplingRate.Value;

		if (options.FeatureSamplingRate != null)
			RandomForestsParameters.FeatureSamplingRate = options.FeatureSamplingRate.Value;

		if (options.RandomForestsRanker != null)
		{
			try
			{
				RandomForestsParameters.RankerType = options.RandomForestsRanker.Value;
			}
			catch (ArgumentException)
			{
				logger.LogCritical("{RandomForestsRanker} cannot be bagged. Random Forests only supports MART/LambdaMART.", options.RandomForestsRanker);
				return 1;
			}
		}

		if (options.L2 != null)
			LinearRegressionParameters.Lambda = options.L2.Value;

		var threads = options.Thread ?? Environment.ProcessorCount;
		LambdaMARTParameters.MaxDegreeOfParallelism = threads;
		RandomForestsParameters.MaxDegreeOfParallelism = threads;

		Normalizer? normalizer = null;
		if (options.Norm != null)
		{
			switch (options.Norm)
			{
				case NormalizerType.Sum:
					normalizer = SumNormalizer.Instance;
					break;
				case NormalizerType.ZScore:
					normalizer = new ZScoreNormalizer();
					break;
				case NormalizerType.Linear:
					normalizer = new LinearNormalizer();
					break;
				default:
					logger.LogCritical("Unknown normalizer: {Normalizer}", options.Norm);
					return 1;
			}
		}

		Type? rankerType;
		IRankerParameters? rankerParameters;
		switch (options.Ranker)
		{
			case RankerType.MART:
				(rankerType, rankerParameters) = (typeof(MART), LambdaMARTParameters);
				break;
			case RankerType.RankBoost:
				(rankerType, rankerParameters) = (typeof(RankBoost), RankBoostParameters);
				break;
			case RankerType.RankNet:
				(rankerType, rankerParameters) = (typeof(RankNet), RankNetParameters);
				break;
			case RankerType.AdaRank:
				(rankerType, rankerParameters) = (typeof(AdaRank), AdaRankParameters);
				break;
			case RankerType.CoordinateAscent:
				(rankerType, rankerParameters) = (typeof(CoordinateAscent), CoordinateAscentParameters);
				break;
			case RankerType.LambdaRank:
				(rankerType, rankerParameters) = (typeof(LambdaRank), RankNetParameters);
				break;
			case RankerType.LambdaMART:
				(rankerType, rankerParameters) = (typeof(LambdaMART), LambdaMARTParameters);
				break;
			case RankerType.ListNet:
				(rankerType, rankerParameters) = (typeof(ListNet), ListNetParameters);
				break;
			case RankerType.RandomForests:
				(rankerType, rankerParameters) = (typeof(RandomForests), RandomForestsParameters);
				break;
			case RankerType.LinearRegression:
				(rankerType, rankerParameters) = (typeof(LinearRegression), LinearRegressionParameters);
				break;
			default:
				logger.LogCritical("Unknown ranker: {Ranker}", options.Ranker);
				return 1;
		}

		var evaluator = _evaluatorFactory.CreateEvaluator(
			trainMetric,
			testMetric,
			normalizer,
			options.MaxLabel,
			options.MustHaveRelevantDocs,
			options.UseSparseRepresentation,
			options.QueryRelevanceInputFile?.FullName);

		if (trainFile != null)
		{
			logger.LogInformation("Training data: {TrainFile}", trainFile);

			if (foldCount != -1)
			{
				logger.LogInformation("Cross validation: {FoldCv} folds.", foldCount);
				if (tvSplit > 0)
					logger.LogInformation("Train-Validation split: {TvSplit}", tvSplit);
			}
			else
			{
				if (testFiles.Count > 0)
					logger.LogInformation("Test data: {TestFile}", string.Join(", ", testFiles));
				else if (ttSplit > 0)
					logger.LogInformation("Train-Test split: {TrainTestSplit}", ttSplit);

				if (validationFile != null)
					logger.LogInformation("Validation data: {ValidationFile}", validationFile);
				else if (ttSplit <= 0 && tvSplit > 0)
					logger.LogInformation("Train-Validation split: {TrainValidationSplit}", tvSplit);
			}

			logger.LogInformation("Feature vector representation: {VectorRepresentation}.", options.UseSparseRepresentation ? "Sparse" : "Dense");
			logger.LogInformation("Ranking method: {Ranker}", options.Ranker);

			if (featureDescriptionFile != null)
				logger.LogInformation("Feature description file: {FeatureDescriptionFile}", featureDescriptionFile);
			else
				logger.LogInformation("Feature description file: Unspecified. All features will be used.");

			logger.LogInformation("Train metric: {TrainMetric}", trainMetric);
			logger.LogInformation("Test metric: {TestMetric}", testMetric);

			if (trainMetric.StartsWith("ERR", StringComparison.OrdinalIgnoreCase)
				|| (testMetric != null && testMetric.StartsWith("ERR", StringComparison.OrdinalIgnoreCase)))
				logger.LogInformation("Highest relevance label (to compute ERR): {HighRelevanceLabel}", options.MaxLabel ?? ERRScorer.DefaultMax);

			if (options.QueryRelevanceInputFile != null)
				logger.LogInformation("TREC-format relevance judgment (only affects MAP and NDCG scores): {QueryRelevanceJudgementFile}", options.QueryRelevanceInputFile.FullName);

			logger.LogInformation("Feature normalization: {FeatureNormalization}", normalizer != null ? normalizer.Name : "No");

			if (kcvModelDir != null)
				logger.LogInformation("Models directory: {KcvModelDir}", kcvModelDir);

			if (!string.IsNullOrEmpty(kcvModelFile))
				logger.LogInformation("Models' name: {KcvModelFile}", kcvModelFile);

			if (options.ModelOutputFile != null)
				logger.LogInformation("Model file: {ModelFile}", options.ModelOutputFile.FullName);

			logger.LogInformation("[+] {Ranker}'s Parameters:", options.Ranker);
			logger.LogInformation("{RankerParameters}", rankerParameters.ToString());

			// starting to do some work
			if (foldCount != -1)
			{
				//- Behavioral changes: Write kcv models if kcvmd OR kcvmn defined.  Use
				//  default names for missing arguments: "kcvmodels" default directory
				//  and "kcv" default model name.
				if (kcvModelDir != null && string.IsNullOrEmpty(kcvModelFile))
					kcvModelFile = "kcv";
				else if (kcvModelDir == null && !string.IsNullOrEmpty(kcvModelFile))
					kcvModelDir = new DirectoryInfo(Path.Combine(Directory.GetCurrentDirectory(), "kcvmodels"));

				//- models won't be saved if kcvModelDir=""   [OBSOLETE]
				//- Models saved if EITHER kcvmd OR kcvmn defined.  Use default names for missing values.
				await evaluator.EvaluateAsync(
					rankerType,
					trainFile.FullName,
					featureDescriptionFile?.FullName,
					foldCount,
					tvSplit,
					kcvModelDir!.FullName,
					kcvModelFile!,
					rankerParameters).ConfigureAwait(false);
			}
			else
			{
				if (ttSplit > 0.0)
				{
					await evaluator.EvaluateAsync(
						rankerType,
						trainFile.FullName,
						validationFile?.FullName,
						featureDescriptionFile?.FullName,
						ttSplit,
						options.ModelOutputFile?.FullName,
						rankerParameters).ConfigureAwait(false);
				}
				else if (tvSplit > 0.0)
				{
					await evaluator.EvaluateAsync(
						rankerType,
						trainFile.FullName,
						tvSplit,
						testFiles.LastOrDefault(),
						featureDescriptionFile?.FullName,
						options.ModelOutputFile?.FullName,
						rankerParameters).ConfigureAwait(false);
				}
				else
				{
					await evaluator.EvaluateAsync(
						rankerType,
						trainFile.FullName,
						validationFile?.FullName,
						testFiles.LastOrDefault(),
						featureDescriptionFile?.FullName,
						options.ModelOutputFile?.FullName,
						rankerParameters).ConfigureAwait(false);
				}
			}
		}
		else
		{
			logger.LogInformation("Model file: {SavedModelFile}", savedModelFiles.Count > 0 ? string.Join(",", savedModelFiles) : "Not Provided");
			logger.LogInformation("Feature normalization: {Normalization}", normalizer != null ? normalizer.Name : "No");

			if (rankFile != null)
			{
				if (scoreFile != null)
				{
					switch (savedModelFiles.Count)
					{
						case > 1:
							evaluator.Score(savedModelFiles, rankFile.FullName, scoreFile.FullName);
							break;
						case 1:
							evaluator.Score(savedModelFiles[0], rankFile.FullName, scoreFile.FullName);
							break;
					}
				}
				else if (indriRankingFile != null)
				{
					switch (savedModelFiles.Count)
					{
						case > 1:
							evaluator.Rank(savedModelFiles, rankFile.FullName, indriRankingFile.FullName);
							break;
						case 1:
							evaluator.Rank(savedModelFiles[0], rankFile.FullName, indriRankingFile.FullName);
							break;
						default:
							// This is *ONLY* for debugging purposes. It is *NOT* exposed via cmd-line
							// It will evaluate the input ranking (without being re-ranked by any model) using any measure specified via metric2T
							evaluator.Rank(rankFile.FullName, indriRankingFile.FullName);
							break;
					}
				}
				else
				{
					logger.LogCritical("This function has been removed. Consider using -score in addition to " +
									   "your current parameters and do the ranking yourself based on these scores.");
					return 1;
				}
			}
			else
			{
				logger.LogInformation("Test metric: {TestMetric}", testMetric);
				if (testMetric.StartsWith("ERR", StringComparison.OrdinalIgnoreCase))
					logger.LogInformation("Highest relevance label (to compute ERR): {HighestRelevanceLabel}", options.MaxLabel ?? ERRScorer.DefaultMax);

				if (testFiles.Count == 0)
				{
					logger.LogCritical("No test files provided. Please provide one or more test files with -test");
					return 1;
				}

				if (savedModelFiles.Count > 1)
				{
					if (testFiles.Count > 1)
						evaluator.Test(savedModelFiles, testFiles, prpFile.FullName);
					else if (testFiles.Count > 0)
						evaluator.Test(savedModelFiles, testFiles.Last(), prpFile.FullName);
				}
				else if (savedModelFiles.Count == 1)
					evaluator.Test(savedModelFiles[0], testFiles.Last(), prpFile?.FullName);
				else if (scoreFile != null)
					evaluator.TestWithScoreFile(testFiles.Last(), scoreFile.FullName);
				else
					evaluator.Test(testFiles.Last(), prpFile?.FullName);
			}
		}

		return 0;
	}
}
