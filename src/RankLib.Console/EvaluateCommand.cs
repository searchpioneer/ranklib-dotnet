using System.CommandLine;
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
	public string Metric2t { get; set; }
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
	public bool? Hr { get; set; }
}

public class EvaluateCommand : Command<EvaluateCommandOptions, EvaluateCommandOptionsHandler>
{
	public EvaluateCommand()
	: base("eval", "evaluate")
	{
		AddOption(new Option<FileInfo>("--train", "Training data file").ExistingOnly());
		AddOption(new Option<RankerType>("--ranker", () => RankerType.COOR_ASCENT, "Ranking algorithm to use"));
		AddOption(new Option<string>("--feature", "Feature description file: list features to be considered by the learner, each on a separate line. If not specified, all features will be used."));
		AddOption(new Option<string>("--metric2t", () => "ERR@10", "Metric to optimize on the training data"));
		AddOption(new Option<double?>("--gmax", "Highest judged relevance label"));
		AddOption(new Option<FileInfo>("--qrel", "TREC-style relevance judgment file").ExistingOnly());
		AddOption(new Option<bool>("--missingZero", "Substitute zero for missing feature values rather than throwing an exception."));
		AddOption(new Option<FileInfo>("--validate", "Specify if you want to tune your system on the validation data (default=unspecified)"));
		AddOption(new Option<float>("--tvs", "If you don't have separate validation data, use this to set train-validation split to be (x)(1.0-x)"));
		AddOption(new Option<FileInfo>("--save", "Save the model learned (default=not-save)"));


		AddOption(new Option<IEnumerable<FileInfo>>("--test", "Specify if you want to evaluate the trained model on this data (default=unspecified)"));
		AddOption(new Option<float>("--tts", "Set train-test split to be (x)(1.0-x). -tts will override -tvs"));
		AddOption(new Option<string>("--metric2T", "Metric to evaluate on the test data (default to the same as specified for -metric2t)"));
		AddOption(new Option<NormalizerType>("--norm", "Normalize all feature vectors (default=no-normalization)"));

		AddOption(new Option<int>("--kcv", () => -1, "Specify if you want to perform k-fold cross validation using the specified training data (default=NoCV)"));
		AddOption(new Option<DirectoryInfo>("--kcvmd", "Directory for models trained via cross-validation (default=not-save)"));
		AddOption(new Option<string>("--kcvmn", "Name for model learned in each fold. It will be prefix-ed with the fold-number (default=empty)"));

		AddOption(new Option<IEnumerable<FileInfo>>("--load", "Load saved model file"));
		AddOption(new Option<int>("--thread", () => -1, "Number of threads to use"));
		AddOption(new Option<FileInfo>("--rank", "Rank the samples in the specified file (specify either this or -test but not both)").ExistingOnly());
		AddOption(new Option<FileInfo>("--indri", "Indri ranking file").ExistingOnly());
		AddOption(new Option<bool>("--sparse", "Use sparse representation"));
		AddOption(new Option<FileInfo>("--idv", "Per-ranked list model performance (in test metric). Has to be used with -test").ExistingOnly());
		AddOption(new Option<FileInfo>("--score", "TODO"));
		AddOption(new Option<int?>("--epoch", "TODO"));
		AddOption(new Option<int?>("--layer", "TODO: layer count"));
		AddOption(new Option<int?>("--node", "TODO: node count"));
		AddOption(new Option<double?>("--lr", "TODO: Learning rate"));
		AddOption(new Option<int?>("--tc", "TODO: Learning rate"));


		AddOption(new Option<bool?>("--noeq", "TODO: AdaRank train with enqueue"));
		AddOption(new Option<int?>("--max", "TODO: AdaRank max select count"));
		AddOption(new Option<int?>("--r", "TODO: CoorAscent restart"));
		AddOption(new Option<int?>("--i", "TODO: CoorAscent max iteration"));
		AddOption(new Option<int?>("--round", "TODO: AdaRank / RankBoost NIteration"));
		AddOption(new Option<double?>("--reg", "TODO: CoorAscent regularization"));
		AddOption(new Option<double?>("--tolerance", "TODO: AdaRank / CoorAscent tolerance"));

		AddOption(new Option<int?>("--tree", "TODO: Number of trees"));
		AddOption(new Option<int?>("--leaf", "TODO: Number of leaves"));
		AddOption(new Option<float?>("--shrinkage", "TODO: Learning Rate"));
		AddOption(new Option<int?>("--mls", "Min leaf support. minimum #samples each leaf has to contain (default=1)"));
		AddOption(new Option<int?>("--estop", "Stop early when no improvement is observed on validaton data in N consecutive rounds (default=100)"));
		AddOption(new Option<int?>("--bag", "Number of bags (default=300)"));
		AddOption(new Option<float?>("--srate", "Sub-sampling rate (default=1.0)"));
		AddOption(new Option<float?>("--frate", "Feature sampling rate (default=0.3)"));
		AddOption(new Option<RankerType?>("--rtype", "RfRanker ranker type to bag. Random Forests only support MART/LambdaMART"));
		AddOption(new Option<double?>("--L2", "TODO: Lambda"));
		AddOption(new Option<bool?>("--hr", "TODO: Must Have Relevance Doc"));
	}
}

public class EvaluateCommandOptionsHandler : ICommandOptionsHandler<EvaluateCommandOptions>
{
	private readonly ILoggerFactory _loggerFactory;
	private readonly EvaluatorFactory _evaluatorFactory;
	private readonly RankerFactory _rankerFactory;
	private LambdaMARTParameters? _lambdaMARTParameters;

	public EvaluateCommandOptionsHandler(
		ILoggerFactory loggerFactory,
		EvaluatorFactory evaluatorFactory,
		RankerFactory rankerFactory)
	{
		_loggerFactory = loggerFactory;
		_evaluatorFactory = evaluatorFactory;
		_rankerFactory = rankerFactory;
	}

	// TODO: this needs to be passed to any LambdaMART created
	private LambdaMARTParameters LambdaMARTParameters => _lambdaMARTParameters ??= new LambdaMARTParameters();

	public Task<int> HandleAsync(EvaluateCommandOptions options, CancellationToken cancellationToken)
	{
		var logger = _loggerFactory.CreateLogger<Evaluator>();

		var trainFile = options.Train;
		var foldCv = options.Kcv;
		var testMetric = options.Metric2T;
		var trainMetric = options.Metric2t;
		var testFile = options.Test?.LastOrDefault();
		var testFiles = options.Test;
		var rankFile = options.Rank;

		var tvSplit = options.Tvs;
		var ttSplit = options.Tts;

		var savedModelFiles = options.Load;
		var savedModelFile = options.Load?.LastOrDefault();

		var kcvModelDir = options.Kcvmd;
		var kcvModelFile = options.Kcvmn;

		var validationFile = options.Validate;
		var featureDescriptionFile = options.Feature;

		var indriRankingFile = options.Indri;
		var prpFile = options.Idv;

		var scoreFile = options.Score;

		if (options.Save != null)
		{
			Evaluator.ModelFile = options.Save.FullName;
		}

		if (options.Sparse)
		{
			Evaluator.UseSparseRepresentation = true;
		}

		if (options.MissingZero)
		{
			DataPoint.MissingZero = true;
		}

		if (options.GMax != null)
		{
			ERRScorer.MAX = Math.Pow(2, options.GMax.Value);
		}

		if (options.Epoch != null)
		{
			RankNet.NIteration = options.Epoch.Value;
			ListNet.nIteration = options.Epoch.Value;
		}

		if (options.Layer != null)
		{
			RankNet.NHiddenLayer = options.Layer.Value;
		}

		if (options.Node != null)
		{
			RankNet.NHiddenNodePerLayer = options.Node.Value;
		}

		if (options.Lr != null)
		{
			RankNet.LearningRate = options.Lr.Value;
			ListNet.learningRate = Neuron.LearningRate;
		}

		if (options.Tc != null)
		{
			RankBoost.NThreshold = options.Tc.Value;
			LambdaMARTParameters.nThreshold = options.Tc.Value;
		}

		if (options.NoEq != null)
		{
			AdaRank.TrainWithEnqueue = false;
		}

		if (options.Max != null)
		{
			AdaRank.MaxSelCount = options.Max.Value;
		}

		if (options.R != null)
		{
			CoorAscent.nRestart = options.R.Value;
		}

		if (options.I != null)
		{
			CoorAscent.nMaxIteration = options.I.Value;
		}

		if (options.Round != null)
		{
			RankBoost.NIteration = options.Round.Value;
			AdaRank.NIteration = options.Round.Value;
		}

		if (options.Reg != null)
		{
			CoorAscent.slack = options.Reg.Value;
			CoorAscent.regularized = true;
		}

		if (options.Tolerance != null)
		{
			AdaRank.Tolerance = options.Tolerance.Value;
			CoorAscent.tolerance = options.Tolerance.Value;
		}

		if (options.Tree != null)
		{
			LambdaMARTParameters.nTrees = options.Tree.Value;
			RFRanker.nTrees = LambdaMARTParameters.nTrees;
		}

		if (options.Leaf != null)
		{
			LambdaMARTParameters.nTreeLeaves = options.Leaf.Value;
			RFRanker.nTreeLeaves = LambdaMARTParameters.nTreeLeaves;
		}

		if (options.Shrinkage != null)
		{
			LambdaMARTParameters.learningRate = options.Shrinkage.Value;
			RFRanker.learningRate = LambdaMARTParameters.learningRate;
		}

		if (options.Mls != null)
		{
			LambdaMARTParameters.minLeafSupport = options.Mls.Value;
			RFRanker.minLeafSupport = LambdaMARTParameters.minLeafSupport;
		}

		if (options.EStop != null)
		{
			LambdaMARTParameters.nRoundToStopEarly = options.EStop.Value;
		}

		if (options.Bag != null)
		{
			RFRanker.nBag = options.Bag.Value;
		}

		if (options.SRate != null)
		{
			RFRanker.subSamplingRate = options.SRate.Value;
		}

		if (options.FRate != null)
		{
			RFRanker.featureSamplingRate = options.FRate.Value;
		}

		if (options.RType != null)
		{
			if (options.RType != RankerType.MART && options.RType != RankerType.LAMBDAMART)
			{
				throw RankLibException.Create(options.RType + " cannot be bagged. Random Forests only supports MART/LambdaMART.");
			}

			RFRanker.rType = options.RType.Value;
		}

		if (options.L2 != null)
		{
			LinearRegRank.lambda = options.L2.Value;
		}

		if (options.Hr != null)
		{
			Evaluator.MustHaveRelDoc = true;
		}

		if (options.Thread == -1)
		{
			options.Thread = Environment.ProcessorCount;
		}
		MyThreadPool.Init(options.Thread);

		if (string.IsNullOrEmpty(options.Metric2T))
		{
			options.Metric2T = options.Metric2t;
		}

		Normalizer? normalizer = null;
		if (options.Norm != null)
		{
			normalizer = options.Norm switch
			{
				NormalizerType.Sum => SumNormalizer.Instance,
				NormalizerType.ZScore => new ZScoreNormalizer(),
				NormalizerType.Linear => new LinearNormalizer(),
				_ => throw RankLibException.Create("Unknown normalizer: " + options.Norm)
			};
		}

		var evaluator = _evaluatorFactory.CreateEvaluator(
			options.Ranker,
			options.Metric2t,
			options.Metric2T,
			normalizer,
			options.QRel?.FullName);

		if (options.Train != null)
		{
			logger.LogInformation("Training data: {TrainFile}", trainFile);

			if (foldCv != -1)
			{
				logger.LogInformation("Cross validation: {FoldCv} folds.", foldCv);
				if (tvSplit > 0)
				{
					logger.LogInformation("Train-Validation split: {TvSplit}", tvSplit);
				}
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

			logger.LogInformation($"Feature vector representation: {(Evaluator.UseSparseRepresentation ? "Sparse" : "Dense")}.");
			logger.LogInformation("Ranking method: {Ranker}", options.Ranker);

			if (featureDescriptionFile != null)
				logger.LogInformation("Feature description file: {FeatureDescriptionFile}", featureDescriptionFile);
			else
				logger.LogInformation("Feature description file: Unspecified. All features will be used.");

			logger.LogInformation("Train metric: {TrainMetric}", trainMetric);
			logger.LogInformation("Test metric: {TestMetric}", testMetric);

			if (trainMetric.StartsWith("ERR", StringComparison.OrdinalIgnoreCase)
			    || testMetric != null && testMetric.StartsWith("ERR", StringComparison.OrdinalIgnoreCase))
				logger.LogInformation("Highest relevance label (to compute ERR): {HighRelevanceLabel}", (int)SimpleMath.LogBase2(ERRScorer.MAX));

			if (options.QRel != null)
				logger.LogInformation("TREC-format relevance judgment (only affects MAP and NDCG scores): {QueryRelevanceJudgementFile}", options.QRel.FullName);

			logger.LogInformation("Feature normalization: {FeatureNormalization}", normalizer != null ? normalizer.Name : "No");

			if (kcvModelDir != null)
			{
				logger.LogInformation("Models directory: {KcvModelDir}", kcvModelDir);
			}

			if (!string.IsNullOrEmpty(kcvModelFile))
			{
				logger.LogInformation($"Models' name: {kcvModelFile}");
			}

			if (!string.IsNullOrEmpty(Evaluator.ModelFile))
			{
				logger.LogInformation($"Model file: {Evaluator.ModelFile}");
			}

			logger.LogInformation($"[+] {options.Ranker}'s Parameters:");

			_rankerFactory.CreateRanker(options.Ranker).PrintParameters();

			// starting to do some work
			if (foldCv != -1)
			{
				//- Behavioral changes: Write kcv models if kcvmd OR kcvmn defined.  Use
				//  default names for missing arguments: "kcvmodels" default directory
				//  and "kcv" default model name.
				if (kcvModelDir != null && string.IsNullOrEmpty(kcvModelFile))
				{
					kcvModelFile = "kcv";
				}
				else if (kcvModelDir == null && !string.IsNullOrEmpty(kcvModelFile))
				{
					kcvModelDir = new DirectoryInfo(Path.Combine(Directory.GetCurrentDirectory(), "kcvmodels"));
				}

				//- models won't be saved if kcvModelDir=""   [OBSOLETE]
				//- Models saved if EITHER kcvmd OR kcvmn defined.  Use default names for missing values.
				evaluator.Evaluate(trainFile!.FullName, featureDescriptionFile?.FullName, foldCv, tvSplit, kcvModelDir!.FullName, kcvModelFile!);
			}
			else
			{
				if (ttSplit > 0.0)
				{
					evaluator.Evaluate(trainFile.FullName, validationFile.FullName, featureDescriptionFile?.FullName, ttSplit);
				}
				else if (tvSplit > 0.0)
				{
					evaluator.Evaluate(trainFile.FullName, tvSplit, testFile.FullName, featureDescriptionFile.FullName);
				}
				else
				{
					evaluator.Evaluate(trainFile.FullName, validationFile?.FullName, testFile?.FullName, featureDescriptionFile?.FullName);
				}
			}
		}
		else
		{
			logger.LogInformation($"Model file: {savedModelFile}");
			logger.LogInformation($"Feature normalization: {(normalizer != null ? normalizer.Name : "No")}");

			if (rankFile != null)
			{
				if (scoreFile != null)
				{
					if (savedModelFiles.Count() > 1)
					{
						evaluator.Score(savedModelFiles.Select(f => f.FullName).ToList(), rankFile.FullName, scoreFile.FullName);
					}
					else
					{
						evaluator.Score(savedModelFile.FullName, rankFile.FullName, scoreFile.FullName);
					}
				}
				else if (indriRankingFile != null)
				{
					if (savedModelFiles?.Count() > 1)
					{
						evaluator.Rank(savedModelFiles.Select(f => f.FullName).ToList(), rankFile.FullName, indriRankingFile.FullName);
					}
					else if (savedModelFiles?.Count() == 1)
					{
						evaluator.Rank(savedModelFile.FullName, rankFile.FullName, indriRankingFile.FullName);
					}
					else
					{
						evaluator.Rank(rankFile.FullName, indriRankingFile.FullName);
					}
				}
				else
				{
					throw RankLibException.Create("This function has been removed. Consider using -score in addition to " +
											  "your current parameters and do the ranking yourself based on these scores.");
				}
			}
			else
			{
				logger.LogInformation("Test metric: {TestMetric}", testMetric);
				if (testMetric.StartsWith("ERR", StringComparison.OrdinalIgnoreCase))
				{
					logger.LogInformation($"Highest relevance label (to compute ERR): {(int)SimpleMath.LogBase2(ERRScorer.MAX)}");
				}

				if (savedModelFiles.Count() > 1)
				{
					if (testFiles.Count() > 1)
					{
						evaluator.Test(savedModelFiles.Select(f => f.FullName).ToList(), testFiles.Select(f => f.FullName).ToList(), prpFile.FullName);
					}
					else
					{
						evaluator.Test(savedModelFiles.Select(f => f.FullName).ToList(), testFile.FullName, prpFile.FullName);
					}
				}
				else if (savedModelFiles.Count() == 1)
				{
					evaluator.Test(savedModelFile.FullName, testFile.FullName, prpFile.FullName);
				}
				else if (scoreFile != null)
				{
					evaluator.TestWithScoreFile(testFile!.FullName, scoreFile.FullName);
				}
				else
				{
					evaluator.Test(testFile!.FullName, prpFile.FullName);
				}
			}
		}

		return Task.FromResult(0);
	}
}
