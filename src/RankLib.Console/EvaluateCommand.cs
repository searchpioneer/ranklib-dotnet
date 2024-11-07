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

namespace RankLib.Console;

public class EvaluateCommandOptions : ICommandOptions
{
	public FileInfo? Train { get; set; }
	public RankerType Ranker { get; set; }
	public FileInfo? Feature { get; set; }

	/// <summary>
	/// Train metric
	/// </summary>
	public string Metric2t { get; set; } = default!;

	/// <summary>
	/// Test metric
	/// </summary>
	public string? Metric2T { get; set; }

	public double? GMax { get; set; }
	public FileInfo? QRel { get; set; }
	public float Tts { get; set; }
	public float Tvs { get; set; }
	public int Kcv { get; set; }
	public DirectoryInfo? Kcvmd { get; set; }
	public string? Kcvmn { get; set; }
	public FileInfo? Validate { get; set; }
	public IEnumerable<FileInfo>? Test { get; set; }
	public NormalizerType? Norm { get; set; }
	public FileInfo? Save { get; set; }
	public IEnumerable<FileInfo>? Load { get; set; }
	public int Thread { get; set; }
	public FileInfo? Rank { get; set; }

	public FileInfo? Indri { get; set; }

	public bool Sparse { get; set; }

	public bool MissingZero { get; set; }

	public FileInfo? Idv { get; set; }
	public FileInfo? Score { get; set; }

	public int? Epoch { get; set; }
	public int? Layer { get; set; }
	public int? Node { get; set; }
	public double? Lr { get; set; }
	public int? Tc { get; set; }

	public bool? NoEq { get; set; }
	public int? Max { get; set; }
	public int? R { get; set; }
	public int? I { get; set; }
	public int? Round { get; set; }
	public double? Reg { get; set; }
	public double? Tolerance { get; set; }
	public int? Tree { get; set; }
	public int? Leaf { get; set; }
	public float? Shrinkage { get; set; }
	public int? Mls { get; set; }
	public int? EStop { get; set; }
	public int? Bag { get; set; }
	public float? SRate { get; set; }
	public float? FRate { get; set; }
	public RankerType? RType { get; set; }
	public double? L2 { get; set; }
	public bool Hr { get; set; }
	public int? RandomSeed { get; set; }
}

public class EvaluateCommand : Command<EvaluateCommandOptions, EvaluateCommandOptionsHandler>
{
	public EvaluateCommand()
	: base("eval", "Trains and evaluates a ranker, or evaluates a previously saved ranker model.")
	{
		AddOption(new Option<FileInfo>("--train", "Training data file").ExistingOnly());
		AddOption(new Option<RankerType>("--ranker", () => RankerType.CoordinateAscent, "Ranking algorithm to use"));
		AddOption(new Option<FileInfo>("--feature", "Feature description file: list features to be considered by the learner, each on a separate line. If not specified, all features will be used.").ExistingOnly());
		AddOption(new Option<string>("--metric2t", () => "ERR@10", "Metric to optimize on the training data. Supports MAP, NDCG@k, DCG@k, P@k, RR@k, ERR@k."));
		AddOption(new Option<double?>("--gmax", "Highest judged relevance label. It affects the calculation of ERR (default=4, i.e. 5-point scale {0,1,2,3,4} where value used is 2^gmax)"));
		AddOption(new Option<FileInfo>("--qrel", "TREC-style relevance judgment file").ExistingOnly());
		AddOption(new Option<bool>("--missingZero", "Substitute zero for missing feature values rather than throwing an exception."));
		AddOption(new Option<FileInfo>("--validate", "Specify if you want to tune your system on the validation data (default=unspecified)").ExistingOnly());
		AddOption(new Option<float>("--tvs", "If you don't have separate validation data, use this to set train-validation split to be (x)(1.0-x)"));
		AddOption(new Option<FileInfo>("--save", "Save the model learned (default=not-save)"));

		AddOption(new Option<IEnumerable<FileInfo>>("--test", "Specify if you want to evaluate the trained model on this data (default=unspecified)").ExistingOnly());
		AddOption(new Option<float>("--tts", "Set train-test split to be (x)(1.0-x). -tts will override -tvs"));
		AddOption(new Option<string>("--metric2T", "Metric to evaluate on the test data (default to the same as specified for --metric2t)"));
		AddOption(new Option<NormalizerType>("--norm", "Type of normalizer to use to normalize all feature vectors (default=no-normalization)"));

		AddOption(new Option<int>("--kcv", () => -1, "Specify if you want to perform k-fold cross validation using the specified training data (default=NoCV)"));
		AddOption(new Option<DirectoryInfo>("--kcvmd", "Directory for models trained via cross-validation (default=not-save)"));
		AddOption(new Option<string>("--kcvmn", "Name for model learned in each fold. It will be prefix-ed with the fold-number (default=empty)"));

		AddOption(new Option<IEnumerable<FileInfo>>("--load", "Load saved model file for evaluation").ExistingOnly());
		AddOption(new Option<int>("--thread", () => Environment.ProcessorCount, "Number of threads to use. If unspecified, will use all available processors"));
		AddOption(new Option<FileInfo>("--rank", "Rank the samples in the specified file (specify either this or -test but not both)").ExistingOnly());
		AddOption(new Option<FileInfo>("--indri", "Indri ranking file").ExistingOnly());
		AddOption(new Option<bool>("--sparse", "Use data points with sparse representation"));
		AddOption(new Option<FileInfo>("--idv", "Per-ranked list model performance (in test metric). Has to be used with --test").ExistingOnly());
		AddOption(new Option<FileInfo>("--score", "Store ranker's score for each object being ranked. Has to be used with --rank"));

		// RankNet specific parameters
		AddOption(new Option<int?>("--epoch", "The number of epochs to train"));
		AddOption(new Option<int?>("--layer", "The number of hidden layers"));
		AddOption(new Option<int?>("--node", "The number of hidden nodes per layer"));
		AddOption(new Option<double?>("--lr", "Learning rate"));

		// MART / LambdaMART / RankBoost specific parameters
		AddOption(new Option<int?>("--tc", "Number of threshold candidates to search or for tree splitting. -1 to use all feature values"));

		// RankBoost / AdaRank specific parameters
		AddOption(new Option<int?>("--round", "The number of rounds to train"));

		// AdaRank specific parameters
		AddOption(new Option<bool?>("--noeq", "Train without enqueuing too-strong features"));
		AddOption(new Option<int?>("--max", "The maximum number of times can a feature be consecutively selected without changing performance"));

		// AdaRank / Coordinate Ascent specific parameters
		AddOption(new Option<double?>("--tolerance", "Tolerance between two consecutive rounds of learning"));

		// Coordinate Ascent specific parameters
		AddOption(new Option<int?>("--r", "The number of random restarts"));
		AddOption(new Option<int?>("--i", "The number of iterations to search in each dimension"));
		AddOption(new Option<double?>("--reg", "Regularization parameter"));

		// MART / LambdaMART specific parameters
		AddOption(new Option<int?>("--tree", "Number of trees"));
		AddOption(new Option<int?>("--leaf", "Number of leaves for each tree"));
		AddOption(new Option<float?>("--shrinkage", "Shrinkage, or learning rate"));
		AddOption(new Option<int?>("--mls", "Minimum leaf support. Minimum number of samples each leaf has to contain"));
		AddOption(new Option<int?>("--estop", "Stop early when no improvement is observed on validation data in e consecutive rounds (default=100)"));

		// Random Forests specific parameters
		AddOption(new Option<int?>("--bag", "Number of bags (default=300)"));
		AddOption(new Option<float?>(["--srate"], () => RandomForestsParameters.DefaultSubSamplingRate, "Sub-sampling rate"));
		AddOption(new Option<float?>("--frate", () => RandomForestsParameters.DefaultFeatureSamplingRate, "Feature sampling rate"));
		AddOption(new Option<string>("--rtype", "Random Forests ranker type to bag. Random Forests only support MART/LambdaMART")
			.FromAmong(RankerType.MART.ToString(), RankerType.LambdaMART.ToString()));
		AddOption(new Option<double?>("--L2", "TODO: Lambda"));
		AddOption(new Option<bool?>("--hr", () => false, "Whether to ignore ranked list without any relevant document."));
		AddOption(new Option<int?>(
			"--randomSeed",
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

		var trainFile = options.Train;
		var foldCv = options.Kcv;
		var testMetric = !string.IsNullOrEmpty(options.Metric2T)
			? options.Metric2T
			: options.Metric2t;
		var trainMetric = options.Metric2t;
		var testFile = options.Test?.LastOrDefault();
		var testFiles = options.Test;
		var rankFile = options.Rank;

		var tvSplit = options.Tvs;
		var ttSplit = options.Tts;

		var savedModelFiles = options.Load != null
			? options.Load.Select(f => f.FullName).ToList()
			: [];

		var kcvModelDir = options.Kcvmd;
		var kcvModelFile = options.Kcvmn;
		var validationFile = options.Validate;
		var featureDescriptionFile = options.Feature;
		var indriRankingFile = options.Indri;
		var prpFile = options.Idv;
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

		if (options.Lr != null)
		{
			RankNetParameters.LearningRate = options.Lr.Value;
			ListNetParameters.LearningRate = Neuron.DefaultLearningRate;
		}

		if (options.Tc != null)
		{
			RankBoostParameters.Threshold = options.Tc.Value;
			LambdaMARTParameters.Threshold = options.Tc.Value;
		}

		if (options.NoEq != null)
			AdaRankParameters.TrainWithEnqueue = false;

		if (options.Max != null)
			AdaRankParameters.MaximumSelectedCount = options.Max.Value;

		if (options.R != null)
			CoordinateAscentParameters.RandomRestartCount = options.R.Value;

		if (options.I != null)
			CoordinateAscentParameters.MaximumIterationCount = options.I.Value;

		if (options.Round != null)
		{
			RankBoostParameters.IterationCount = options.Round.Value;
			AdaRankParameters.IterationCount = options.Round.Value;
		}

		if (options.Reg != null)
		{
			CoordinateAscentParameters.Slack = options.Reg.Value;
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

		if (options.Mls != null)
		{
			LambdaMARTParameters.MinimumLeafSupport = options.Mls.Value;
			RandomForestsParameters.MinimumLeafSupport = LambdaMARTParameters.MinimumLeafSupport;
		}

		if (options.EStop != null)
			LambdaMARTParameters.StopEarlyRoundCount = options.EStop.Value;

		if (options.Bag != null)
			RandomForestsParameters.BagCount = options.Bag.Value;

		if (options.SRate != null)
			RandomForestsParameters.SubSamplingRate = options.SRate.Value;

		if (options.FRate != null)
			RandomForestsParameters.FeatureSamplingRate = options.FRate.Value;

		if (options.RType != null)
		{
			try
			{
				RandomForestsParameters.RankerType = options.RType.Value;
			}
			catch (ArgumentException)
			{
				logger.LogCritical($"{options.RType} cannot be bagged. Random Forests only supports MART/LambdaMART.");
				return 1;
			}
		}

		if (options.L2 != null)
			LinearRegressionParameters.Lambda = options.L2.Value;

		var threads = options.Thread == -1
			? Environment.ProcessorCount
			: options.Thread;

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
			options.GMax,
			options.Hr,
			options.Sparse,
			options.QRel?.FullName);

		if (trainFile != null)
		{
			logger.LogInformation("Training data: {TrainFile}", trainFile);

			if (foldCv != -1)
			{
				logger.LogInformation("Cross validation: {FoldCv} folds.", foldCv);
				if (tvSplit > 0)
					logger.LogInformation("Train-Validation split: {TvSplit}", tvSplit);
			}
			else
			{
				if (testFile != null)
					logger.LogInformation("Test data: {TestFile}", testFile);
				else if (ttSplit > 0)
					logger.LogInformation("Train-Test split: {TrainTestSplit}", ttSplit);

				if (validationFile != null)
					logger.LogInformation("Validation data: {ValidationFile}", validationFile);
				else if (ttSplit <= 0 && tvSplit > 0)
					logger.LogInformation("Train-Validation split: {TrainValidationSplit}", tvSplit);
			}

			logger.LogInformation("Feature vector representation: {VectorRepresentation}.", options.Sparse ? "Sparse" : "Dense");
			logger.LogInformation("Ranking method: {Ranker}", options.Ranker);

			if (featureDescriptionFile != null)
				logger.LogInformation("Feature description file: {FeatureDescriptionFile}", featureDescriptionFile);
			else
				logger.LogInformation("Feature description file: Unspecified. All features will be used.");

			logger.LogInformation("Train metric: {TrainMetric}", trainMetric);
			logger.LogInformation("Test metric: {TestMetric}", testMetric);

			if (trainMetric.StartsWith("ERR", StringComparison.OrdinalIgnoreCase)
				|| (testMetric != null && testMetric.StartsWith("ERR", StringComparison.OrdinalIgnoreCase)))
				logger.LogInformation("Highest relevance label (to compute ERR): {HighRelevanceLabel}", (int)SimpleMath.LogBase2(ERRScorer.DefaultMax));

			if (options.QRel != null)
				logger.LogInformation("TREC-format relevance judgment (only affects MAP and NDCG scores): {QueryRelevanceJudgementFile}", options.QRel.FullName);

			logger.LogInformation("Feature normalization: {FeatureNormalization}", normalizer != null ? normalizer.Name : "No");

			if (kcvModelDir != null)
				logger.LogInformation("Models directory: {KcvModelDir}", kcvModelDir);

			if (!string.IsNullOrEmpty(kcvModelFile))
				logger.LogInformation("Models' name: {KcvModelFile}", kcvModelFile);

			if (options.Save != null)
				logger.LogInformation("Model file: {ModelFile}", options.Save.FullName);

			logger.LogInformation("[+] {Ranker}'s Parameters:", options.Ranker);
			logger.LogInformation("{RankerParameters}", rankerParameters.ToString());

			// starting to do some work
			if (foldCv != -1)
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
					foldCv,
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
						options.Save?.FullName,
						rankerParameters).ConfigureAwait(false);
				}
				else if (tvSplit > 0.0)
				{
					await evaluator.EvaluateAsync(
						rankerType,
						trainFile.FullName,
						tvSplit,
						testFile?.FullName,
						featureDescriptionFile?.FullName,
						options.Save?.FullName,
						rankerParameters).ConfigureAwait(false);
				}
				else
				{
					await evaluator.EvaluateAsync(
						rankerType,
						trainFile.FullName,
						validationFile?.FullName,
						testFile?.FullName,
						featureDescriptionFile?.FullName,
						options.Save?.FullName,
						rankerParameters).ConfigureAwait(false);
				}
			}
		}
		else
		{
			logger.LogInformation("Model file: {SavedModelFile}", string.Join(",", savedModelFiles));
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
					logger.LogInformation("Highest relevance label (to compute ERR): {HighestRelevanceLabel}", options.GMax ?? (int)SimpleMath.LogBase2(ERRScorer.DefaultMax));

				if (savedModelFiles.Count > 1)
				{
					if (testFiles.Count() > 1)
						evaluator.Test(savedModelFiles, testFiles.Select(f => f.FullName).ToList(), prpFile.FullName);
					else
						evaluator.Test(savedModelFiles, testFile.FullName, prpFile.FullName);
				}
				else if (savedModelFiles.Count == 1)
					evaluator.Test(savedModelFiles[0], testFile.FullName, prpFile?.FullName);
				else if (scoreFile != null)
					evaluator.TestWithScoreFile(testFile!.FullName, scoreFile.FullName);
				else
					evaluator.Test(testFile!.FullName, prpFile?.FullName);
			}
		}

		return 0;
	}
}
