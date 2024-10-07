using System.Globalization;
using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Features;
using RankLib.Learning;
using RankLib.Learning.Boosting;
using RankLib.Learning.NeuralNet;
using RankLib.Learning.Tree;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Eval;

public class Evaluator
{
	internal static ILoggerFactory LoggerFactory = new NullLoggerFactory();
	internal static ILogger<Evaluator> Logger = NullLogger<Evaluator>.Instance;

	// main settings
	public static bool MustHaveRelDoc = false;
	public static bool UseSparseRepresentation = false;
	public static bool normalize = false;
	public static Normalizer Nml = new SumNormalizer();
	public static string ModelFile = "";

	public static string QrelFile = ""; // measure such as NDCG and MAP requires "complete" judgment.

	// tmp settings, for personal use
	public static string NewFeatureFile = "";
	public static bool KeepOrigFeatures = false;
	public static int TopNew = 2000;

	protected RankerFactory RankerFactory = new();
	protected MetricScorerFactory MetricScorerFactory = new();

	protected MetricScorer? TrainScorer;
	protected MetricScorer? TestScorer;
	protected RankerType RankerType = RankerType.MART;

	public static void Main(string[] args)
	{
		Evaluator.Logger = LoggerFactory.CreateLogger<Evaluator>();
		FeatureManager.Logger = LoggerFactory.CreateLogger<FeatureManager>();

		string[] rType = ["MART",
			"RankNet",
			"RankBoost",
			"AdaRank",
			"Coordinate Ascent",
			"LambdaRank",
			"LambdaMART",
			"ListNet",
			"Random Forests",
			"Linear Regression"
		];
		RankerType[] rType2 = [RankerType.MART,
			RankerType.RANKNET,
			RankerType.RANKBOOST,
			RankerType.ADARANK,
			RankerType.COOR_ASCENT,
			RankerType.LAMBDARANK,
			RankerType.LAMBDAMART,
			RankerType.LISTNET,
			RankerType.RANDOM_FOREST,
			RankerType.LINEAR_REGRESSION
		];

		var trainFile = "";
		var featureDescriptionFile = "";
		float ttSplit = 0; // train-test split
		float tvSplit = 0; // train-validation split
		var foldCV = -1;
		var validationFile = "";
		var testFile = "";
		var testFiles = new List<string>();
		var rankerType = 4;
		var trainMetric = "ERR@10";
		var testMetric = "";
		Evaluator.normalize = false;
		var savedModelFile = "";
		var savedModelFiles = new List<string>();
		var kcvModelDir = "";
		var kcvModelFile = "";
		var rankFile = "";
		var prpFile = "";
		var nThread = -1;
		var indriRankingFile = "";
		var scoreFile = "";

		if (args.Length < 2)
		{
			Logger.LogInformation("Usage: dotnet run <Params>");
			Logger.LogInformation("Params:");
			Logger.LogInformation("  [+] Training (+ tuning and evaluation)");
			Logger.LogInformation("\t-train <file>\t\tTraining data");
			Logger.LogInformation("\t-ranker <type>\t\tSpecify which ranking algorithm to use");
			Logger.LogInformation("\t\t\t\t0: MART (gradient boosted regression tree)");
			Logger.LogInformation("\t\t\t\t1: RankNet");
			Logger.LogInformation("\t\t\t\t2: RankBoost");
			Logger.LogInformation("\t\t\t\t3: AdaRank");
			Logger.LogInformation("\t\t\t\t4: Coordinate Ascent");
			Logger.LogInformation("\t\t\t\t6: LambdaMART");
			Logger.LogInformation("\t\t\t\t7: ListNet");
			Logger.LogInformation("\t\t\t\t8: Random Forests");
			Logger.LogInformation("\t\t\t\t9: Linear regression (L2 regularization)");
			Logger.LogInformation(
				"\t[ -feature <file> ]\tFeature description file: list features to be considered by the learner, each on a separate line");
			Logger.LogInformation("\t\t\t\tIf not specified, all features will be used.");
			Logger.LogInformation("\t[ -metric2t <metric> ]\tMetric to optimize on the training data.  Supported: MAP, NDCG@k, DCG@k, P@k, RR@k, ERR@k (default=ERR@10)");
			Logger.LogInformation("\t[ -gmax <label> ]\tHighest judged relevance label. It affects the calculation of ERR.");
			Logger.LogInformation("\t[ -qrel <file> ]\tTREC-style relevance judgment file. It only affects MAP and NDCG (default=unspecified)");
			Logger.LogInformation("\t[ -silent ]\t\tDo not print progress messages (which are printed by default)");
			Logger.LogInformation("\t[ -missingZero ]\tSubstitute zero for missing feature values rather than throwing an exception.");
			Logger.LogInformation("\t[ -validate <file> ]\tSpecify if you want to tune your system on the validation data (default=unspecified)");
			Logger.LogInformation("\t[ -tvs <x \\in [0..1]> ]\tIf you don't have separate validation data, use this to set train-validation split to be (x)(1.0-x)");
			Logger.LogInformation("\t[ -save <model> ]\tSave the model learned (default=not-save)");
			Logger.LogInformation("\t[ -test <file> ]\tSpecify if you want to evaluate the trained model on this data (default=unspecified)");
			Logger.LogInformation("\t[ -tts <x \\in [0..1]> ]\tSet train-test split to be (x)(1.0-x). -tts will override -tvs");
			Logger.LogInformation("\t[ -metric2T <metric> ]\tMetric to evaluate on the test data (default to the same as specified for -metric2t)");
			Logger.LogInformation("\t[ -norm <method>]\tNormalize all feature vectors (default=no-normalization). Method can be:");
			Logger.LogInformation("\t\t\t\tsum: normalize each feature by the sum of all its values");
			Logger.LogInformation("\t\t\t\tzscore: normalize each feature by its mean/standard deviation");
			Logger.LogInformation("\t\t\t\tlinear: normalize each feature by its min/max values");
			Logger.LogInformation("\t[ -kcv <k> ]\t\tSpecify if you want to perform k-fold cross validation using the specified training data (default=NoCV)");
			Logger.LogInformation("\t[ -kcvmd <dir> ]\tDirectory for models trained via cross-validation (default=not-save)");
			Logger.LogInformation("\t[ -kcvmn <model> ]\tName for model learned in each fold. It will be prefixed with the fold-number (default=empty)");
			return;
		}

		for (var i = 0; i < args.Length; i++)
		{
			if (args[i].Equals("-train", StringComparison.OrdinalIgnoreCase))
			{
				trainFile = args[++i];
			}
			else if (args[i].Equals("-ranker", StringComparison.OrdinalIgnoreCase))
			{
				rankerType = int.Parse(args[++i]);
			}
			else if (args[i].Equals("-feature", StringComparison.OrdinalIgnoreCase))
			{
				featureDescriptionFile = args[++i];
			}
			else if (args[i].Equals("-metric2t", StringComparison.OrdinalIgnoreCase))
			{
				trainMetric = args[++i];
			}
			else if (args[i].Equals("-metric2T", StringComparison.OrdinalIgnoreCase))
			{
				testMetric = args[++i];
			}
			else if (args[i].Equals("-gmax", StringComparison.OrdinalIgnoreCase))
			{
				ERRScorer.MAX = Math.Pow(2, double.Parse(args[++i], CultureInfo.InvariantCulture));
			}
			else if (args[i].Equals("-qrel", StringComparison.OrdinalIgnoreCase))
			{
				QrelFile = args[++i];
			}
			else if (args[i].Equals("-tts", StringComparison.OrdinalIgnoreCase))
			{
				ttSplit = float.Parse(args[++i], CultureInfo.InvariantCulture);
			}
			else if (args[i].Equals("-tvs", StringComparison.OrdinalIgnoreCase))
			{
				tvSplit = float.Parse(args[++i], CultureInfo.InvariantCulture);
			}
			else if (args[i].Equals("-kcv", StringComparison.OrdinalIgnoreCase))
			{
				foldCV = int.Parse(args[++i]);
			}
			else if (args[i].Equals("-validate", StringComparison.OrdinalIgnoreCase))
			{
				validationFile = args[++i];
			}
			else if (args[i].Equals("-test", StringComparison.OrdinalIgnoreCase))
			{
				testFile = args[++i];
				testFiles.Add(testFile);
			}
			else if (args[i].Equals("-norm", StringComparison.OrdinalIgnoreCase))
			{
				normalize = true;
				var n = args[++i];
				if (n.Equals("sum", StringComparison.OrdinalIgnoreCase))
				{
					Nml = new SumNormalizer();
				}
				else if (n.Equals("zscore", StringComparison.OrdinalIgnoreCase))
				{
					Nml = new ZScoreNormalizer();
				}
				else if (n.Equals("linear", StringComparison.OrdinalIgnoreCase))
				{
					Nml = new LinearNormalizer();
				}
				else
				{
					throw RankLibError.Create("Unknown normalizer: " + n);
				}
			}
			else if (args[i].Equals("-sparse", StringComparison.OrdinalIgnoreCase))
			{
				UseSparseRepresentation = true;
			}
			else if (args[i].Equals("-save", StringComparison.OrdinalIgnoreCase))
			{
				Evaluator.ModelFile = args[++i];
			}
			else if (args[i].Equals("-kcvmd", StringComparison.OrdinalIgnoreCase))
			{
				kcvModelDir = args[++i];
			}
			else if (args[i].Equals("-kcvmn", StringComparison.OrdinalIgnoreCase))
			{
				kcvModelFile = args[++i];
			}
			else if (args[i].Equals("-missingZero", StringComparison.OrdinalIgnoreCase))
			{
				DataPoint.MissingZero = true;
			}
			else if (args[i].Equals("-load", StringComparison.OrdinalIgnoreCase))
			{
				savedModelFile = args[++i];
				savedModelFiles.Add(args[i]);
			}
			else if (args[i].Equals("-idv", StringComparison.OrdinalIgnoreCase))
			{
				prpFile = args[++i];
			}
			else if (args[i].Equals("-rank", StringComparison.OrdinalIgnoreCase))
			{
				rankFile = args[++i];
			}
			else if (args[i].Equals("-score", StringComparison.OrdinalIgnoreCase))
			{
				scoreFile = args[++i];
			}
			else if (args[i].Equals("-epoch", StringComparison.OrdinalIgnoreCase))
			{
				RankNet.NIteration = int.Parse(args[++i]);
				ListNet.nIteration = int.Parse(args[i]);
			}
			else if (args[i].Equals("-layer", StringComparison.OrdinalIgnoreCase))
			{
				RankNet.NHiddenLayer = int.Parse(args[++i]);
			}
			else if (args[i].Equals("-node", StringComparison.OrdinalIgnoreCase))
			{
				RankNet.NHiddenNodePerLayer = int.Parse(args[++i]);
			}
			else if (args[i].Equals("-lr", StringComparison.OrdinalIgnoreCase))
			{
				RankNet.LearningRate = double.Parse(args[++i]);
				ListNet.learningRate = Neuron.LearningRate;
			}

			//RankBoost
			else if (args[i].Equals("-tc", StringComparison.OrdinalIgnoreCase))
			{
				RankBoost.NThreshold = int.Parse(args[++i]);
				LambdaMART.nThreshold = int.Parse(args[i]);
			}

			//AdaRank
			else if (args[i].Equals("-noeq", StringComparison.OrdinalIgnoreCase))
			{
				AdaRank.TrainWithEnqueue = false;
			}
			else if (args[i].Equals("-max", StringComparison.OrdinalIgnoreCase))
			{
				AdaRank.MaxSelCount = int.Parse(args[++i]);
			}
			else if (args[i].Equals("-r", StringComparison.OrdinalIgnoreCase))
			{
				CoorAscent.nRestart = int.Parse(args[++i]);
			}
			else if (args[i].Equals("-i", StringComparison.OrdinalIgnoreCase))
			{
				CoorAscent.nMaxIteration = int.Parse(args[++i]);
			}
			else if (args[i].Equals("-round", StringComparison.OrdinalIgnoreCase))
			{
				RankBoost.NIteration = int.Parse(args[++i]);
				AdaRank.NIteration = int.Parse(args[i]);
			}
			else if (args[i].Equals("-reg", StringComparison.OrdinalIgnoreCase))
			{
				CoorAscent.slack = double.Parse(args[++i]);
				CoorAscent.regularized = true;
			}
			else if (args[i].Equals("-tolerance", StringComparison.OrdinalIgnoreCase))
			{
				AdaRank.Tolerance = double.Parse(args[++i]);
				CoorAscent.tolerance = double.Parse(args[i]);
			}

			//MART / LambdaMART / Random forest
			else if (args[i].Equals("-tree", StringComparison.OrdinalIgnoreCase))
			{
				LambdaMART.nTrees = int.Parse(args[++i]);
				RFRanker.nTrees = int.Parse(args[i]);
			}
			else if (args[i].Equals("-leaf", StringComparison.OrdinalIgnoreCase))
			{
				LambdaMART.nTreeLeaves = int.Parse(args[++i]);
				RFRanker.nTreeLeaves = int.Parse(args[i]);
			}
			else if (args[i].Equals("-shrinkage", StringComparison.OrdinalIgnoreCase))
			{
				LambdaMART.learningRate = float.Parse(args[++i]);
				RFRanker.learningRate = float.Parse(args[i]);
			}
			else if (args[i].Equals("-mls", StringComparison.OrdinalIgnoreCase))
			{
				LambdaMART.minLeafSupport = int.Parse(args[++i]);
				RFRanker.minLeafSupport = LambdaMART.minLeafSupport;
			}
			else if (args[i].Equals("-estop", StringComparison.OrdinalIgnoreCase))
			{
				LambdaMART.nRoundToStopEarly = int.Parse(args[++i]);
			}
			else if (args[i].Equals("-bag", StringComparison.OrdinalIgnoreCase))
			{
				RFRanker.nBag = int.Parse(args[++i]);
			}
			else if (args[i].Equals("-srate", StringComparison.OrdinalIgnoreCase))
			{
				RFRanker.subSamplingRate = float.Parse(args[++i]);
			}
			else if (args[i].Equals("-frate", StringComparison.OrdinalIgnoreCase))
			{
				RFRanker.featureSamplingRate = float.Parse(args[++i]);
			}
			else if (args[i].Equals("-rtype", StringComparison.OrdinalIgnoreCase))
			{
				var rt = int.Parse(args[++i]);
				if (rt == 0 || rt == 6)
				{
					RFRanker.rType = rType2[rt];
				}
				else
				{
					throw RankLibError.Create(rType[rt] + " cannot be bagged. Random Forests only supports MART/LambdaMART.");
				}
			}

			else if (args[i].Equals("-L2", StringComparison.OrdinalIgnoreCase))
			{
				LinearRegRank.lambda = double.Parse(args[++i]);
			}
			else if (args[i].Equals("-thread", StringComparison.OrdinalIgnoreCase))
			{
				nThread = int.Parse(args[++i]);
			}
			else if (args[i].Equals("-nf", StringComparison.OrdinalIgnoreCase))
			{
				NewFeatureFile = args[++i];
			}
			else if (args[i].Equals("-keep", StringComparison.OrdinalIgnoreCase))
			{
				KeepOrigFeatures = true;
			}
			else if (args[i].Equals("-t", StringComparison.OrdinalIgnoreCase))
			{
				TopNew = int.Parse(args[++i]);
			}
			else if (args[i].Equals("-indri", StringComparison.OrdinalIgnoreCase))
			{
				indriRankingFile = args[++i];
			}
			else if (args[i].Equals("-hr", StringComparison.OrdinalIgnoreCase))
			{
				MustHaveRelDoc = true;
			}
			else
			{
				throw RankLibError.Create("Unknown command-line parameter: " + args[i]);
			}
		}

		// Other initialization, handling threads and metrics
		if (nThread == -1)
		{
			nThread = Environment.ProcessorCount;
		}
		MyThreadPool.Init(nThread);

		if (string.IsNullOrEmpty(testMetric))
		{
			testMetric = trainMetric;
		}

		Logger.LogInformation(KeepOrigFeatures ? "Keep orig. features" : "Discard orig. features");
		var evaluator = new Evaluator(rType2[rankerType], trainMetric, testMetric);

		if (!string.IsNullOrEmpty(trainFile))
		{
			Logger.LogInformation($"Training data: {trainFile}");

			if (foldCV != -1)
			{
				Logger.LogInformation($"Cross validation: {foldCV} folds.");
				if (tvSplit > 0)
				{
					Logger.LogInformation($"Train-Validation split: {tvSplit}");
				}
			}
			else
			{
				if (!string.IsNullOrEmpty(testFile))
				{
					Logger.LogInformation($"Test data: {testFile}");
				}
				else if (ttSplit > 0)
				{
					Logger.LogInformation($"Train-Test split: {ttSplit}");
				}

				if (!string.IsNullOrEmpty(validationFile))
				{
					Logger.LogInformation($"Validation data: {validationFile}");
				}
				else if (ttSplit <= 0 && tvSplit > 0)
				{
					Logger.LogInformation($"Train-Validation split: {tvSplit}");
				}
			}
			Logger.LogInformation($"Feature vector representation: {(UseSparseRepresentation ? "Sparse" : "Dense")}.");
			Logger.LogInformation($"Ranking method: {rType[rankerType]}");
			if (!string.IsNullOrEmpty(featureDescriptionFile))
			{
				Logger.LogInformation($"Feature description file: {featureDescriptionFile}");
			}
			else
			{
				Logger.LogInformation("Feature description file: Unspecified. All features will be used.");
			}
			Logger.LogInformation($"Train metric: {trainMetric}");
			Logger.LogInformation($"Test metric: {testMetric}");

			if (trainMetric.StartsWith("ERR", StringComparison.OrdinalIgnoreCase) || testMetric.StartsWith("ERR", StringComparison.OrdinalIgnoreCase))
			{
				Logger.LogInformation($"Highest relevance label (to compute ERR): {(int)SimpleMath.LogBase2(ERRScorer.MAX)}");
			}
			if (!string.IsNullOrEmpty(QrelFile))
			{
				Logger.LogInformation($"TREC-format relevance judgment (only affects MAP and NDCG scores): {QrelFile}");
			}
			Logger.LogInformation($"Feature normalization: {(normalize ? Nml.Name : "No")}");

			if (!string.IsNullOrEmpty(kcvModelDir))
			{
				Logger.LogInformation($"Models directory: {kcvModelDir}");
			}

			if (!string.IsNullOrEmpty(kcvModelFile))
			{
				Logger.LogInformation($"Models' name: {kcvModelFile}");
			}

			if (!string.IsNullOrEmpty(ModelFile))
			{
				Logger.LogInformation($"Model file: {ModelFile}");
			}

			Logger.LogInformation($"[+] {rType[rankerType]}'s Parameters:");
			var rf = new RankerFactory(LoggerFactory);

			rf.CreateRanker(rType2[rankerType]).PrintParameters();

			// starting to do some work
			if (foldCV != -1)
			{
				//- Behavioral changes: Write kcv models if kcvmd OR kcvmn defined.  Use
				//  default names for missing arguments: "kcvmodels" default directory
				//  and "kcv" default model name.
				if (!string.IsNullOrEmpty(kcvModelDir) && string.IsNullOrEmpty(kcvModelFile))
				{
					kcvModelFile = "kcv";
				}
				else if (string.IsNullOrEmpty(kcvModelDir) && !string.IsNullOrEmpty(kcvModelFile))
				{
					kcvModelDir = "kcvmodels";
				}

				//- models won't be saved if kcvModelDir=""   [OBSOLETE]
				//- Models saved if EITHER kcvmd OR kcvmn defined.  Use default names for missing values.
				evaluator.Evaluate(trainFile, featureDescriptionFile, foldCV, tvSplit, kcvModelDir, kcvModelFile);
			}
			else
			{
				if (ttSplit > 0.0)
				{
					evaluator.Evaluate(trainFile, validationFile, featureDescriptionFile, ttSplit);
				}
				else if (tvSplit > 0.0)
				{
					evaluator.Evaluate(trainFile, tvSplit, testFile, featureDescriptionFile);
				}
				else
				{
					evaluator.Evaluate(trainFile, validationFile, testFile, featureDescriptionFile);
				}
			}
		}
		else
		{
			Logger.LogInformation($"Model file: {savedModelFile}");
			Logger.LogInformation($"Feature normalization: {(normalize ? Nml.Name : "No")}");

			if (!string.IsNullOrEmpty(rankFile))
			{
				if (!string.IsNullOrEmpty(scoreFile))
				{
					if (savedModelFiles.Count > 1)
					{
						evaluator.Score(savedModelFiles, rankFile, scoreFile);
					}
					else
					{
						evaluator.Score(savedModelFile, rankFile, scoreFile);
					}
				}
				else if (!string.IsNullOrEmpty(indriRankingFile))
				{
					if (savedModelFiles.Count > 1)
					{
						evaluator.Rank(savedModelFiles, rankFile, indriRankingFile);
					}
					else if (savedModelFiles.Count == 1)
					{
						evaluator.Rank(savedModelFile, rankFile, indriRankingFile);
					}
					else
					{
						evaluator.Rank(rankFile, indriRankingFile);
					}
				}
				else
				{
					throw RankLibError.Create("This function has been removed. Consider using -score in addition to " +
											  "your current parameters and do the ranking yourself based on these scores.");
				}
			}
			else
			{
				Logger.LogInformation($"Test metric: {testMetric}");
				if (testMetric.StartsWith("ERR", StringComparison.OrdinalIgnoreCase))
				{
					Logger.LogInformation($"Highest relevance label (to compute ERR): {(int)SimpleMath.LogBase2(ERRScorer.MAX)}");
				}

				if (savedModelFiles.Count > 1)
				{
					if (testFiles.Count > 1)
					{
						evaluator.Test(savedModelFiles, testFiles, prpFile);
					}
					else
					{
						evaluator.Test(savedModelFiles, testFile, prpFile);
					}
				}
				else if (savedModelFiles.Count == 1)
				{
					evaluator.Test(savedModelFile, testFile, prpFile);
				}
				else if (!string.IsNullOrEmpty(scoreFile))
				{
					evaluator.TestWithScoreFile(testFile, scoreFile);
				}
				else
				{
					evaluator.Test(testFile, prpFile);
				}
			}
		}
	}

	public Evaluator(RankerType rType, Metric.Metric trainMetric, Metric.Metric testMetric, ILoggerFactory? loggerFactory = null)
	{
		loggerFactory ??= new NullLoggerFactory();

		RankerFactory = new(loggerFactory);
		MetricScorerFactory = new(loggerFactory);


		RankerType = rType;
		TrainScorer = MetricScorerFactory.CreateScorer(trainMetric);
		TestScorer = MetricScorerFactory.CreateScorer(testMetric);

		if (!string.IsNullOrEmpty(QrelFile))
		{
			TrainScorer.LoadExternalRelevanceJudgment(QrelFile);
			TestScorer.LoadExternalRelevanceJudgment(QrelFile);
		}
	}

	public Evaluator(RankerType rType, Metric.Metric trainMetric, int trainK, Metric.Metric testMetric, int testK)
	{
		RankerType = rType;
		TrainScorer = MetricScorerFactory.CreateScorer(trainMetric, trainK);
		TestScorer = MetricScorerFactory.CreateScorer(testMetric, testK);

		if (!string.IsNullOrEmpty(QrelFile))
		{
			TrainScorer.LoadExternalRelevanceJudgment(QrelFile);
			TestScorer.LoadExternalRelevanceJudgment(QrelFile);
		}
	}

	public Evaluator(RankerType rType, Metric.Metric trainMetric, Metric.Metric testMetric, int k)
	{
		RankerType = rType;
		TrainScorer = MetricScorerFactory.CreateScorer(trainMetric, k);
		TestScorer = MetricScorerFactory.CreateScorer(testMetric, k);

		if (!string.IsNullOrEmpty(QrelFile))
		{
			TrainScorer.LoadExternalRelevanceJudgment(QrelFile);
			TestScorer.LoadExternalRelevanceJudgment(QrelFile);
		}
	}

	public Evaluator(RankerType rType, Metric.Metric metric, int k)
	{
		RankerType = rType;
		TrainScorer = MetricScorerFactory.CreateScorer(metric, k);

		if (!string.IsNullOrEmpty(QrelFile))
		{
			TrainScorer.LoadExternalRelevanceJudgment(QrelFile);
		}
		TestScorer = TrainScorer;
	}

	public Evaluator(RankerType rType, string trainMetric, string testMetric)
	{
		RankerType = rType;
		TrainScorer = MetricScorerFactory.CreateScorer(trainMetric);
		TestScorer = MetricScorerFactory.CreateScorer(testMetric);

		if (!string.IsNullOrEmpty(QrelFile))
		{
			TrainScorer.LoadExternalRelevanceJudgment(QrelFile);
			TestScorer.LoadExternalRelevanceJudgment(QrelFile);
		}
	}

	public List<RankList> ReadInput(string inputFile) => FeatureManager.ReadInput(inputFile, MustHaveRelDoc, UseSparseRepresentation);

	public void Normalize(List<RankList> samples)
	{
		foreach (var sample in samples)
		{
			Nml.Normalize(sample);
		}
	}

	public void Normalize(List<RankList> samples, int[] fids)
	{
		foreach (var sample in samples)
		{
			Nml.Normalize(sample, fids);
		}
	}

	public void NormalizeAll(List<List<RankList>> samples, int[] fids)
	{
		foreach (var sample in samples)
		{
			Normalize(sample, fids);
		}
	}

	public int[]? ReadFeature(string featureDefFile)
	{
		if (string.IsNullOrEmpty(featureDefFile))
		{
			return null;
		}
		return FeatureManager.ReadFeature(featureDefFile);
	}

	public double Evaluate(Ranker? ranker, List<RankList> rl)
	{
		var rankedList = ranker != null ? ranker.Rank(rl) : rl;
		return TestScorer.Score(rankedList);
	}

	public void Evaluate(string trainFile, string? validationFile, string? testFile, string? featureDefFile)
	{
		var train = ReadInput(trainFile);
		var validation = !string.IsNullOrEmpty(validationFile) ? ReadInput(validationFile) : null;
		var test = !string.IsNullOrEmpty(testFile) ? ReadInput(testFile) : null;

		var features = ReadFeature(featureDefFile) ?? FeatureManager.GetFeatureFromSampleVector(train);

		if (normalize)
		{
			Normalize(train, features);
			if (validation != null)
				Normalize(validation, features);
			if (test != null)
				Normalize(test, features);
		}

		var trainer = new RankerTrainer(LoggerFactory);
		var ranker = trainer.Train(RankerType, train, validation, features, TestScorer);

		if (test != null)
		{
			var rankScore = Evaluate(ranker, test);
			Logger.LogInformation($"{TestScorer.Name} on test data: {Math.Round(rankScore, 4)}");
		}

		if (!string.IsNullOrEmpty(ModelFile))
		{
			ranker.Save(ModelFile);
			Logger.LogInformation($"Model saved to: {ModelFile}");
		}
	}

	public void Evaluate(string sampleFile, string validationFile, string featureDefFile, double percentTrain)
	{
		var trainingData = new List<RankList>();
		var testData = new List<RankList>();
		var features = PrepareSplit(sampleFile, featureDefFile, percentTrain, normalize, trainingData, testData);
		var validation = !string.IsNullOrEmpty(validationFile) ? ReadInput(validationFile) : null;

		if (normalize && validation != null)
		{
			Normalize(validation, features);
		}

		var trainer = new RankerTrainer(LoggerFactory);
		var ranker = trainer.Train(RankerType, trainingData, validation, features, TestScorer);

		var rankScore = Evaluate(ranker, testData);
		Logger.LogInformation($"{TestScorer.Name} on test data: {Math.Round(rankScore, 4)}");

		if (!string.IsNullOrEmpty(ModelFile))
		{
			ranker.Save(ModelFile);
			Logger.LogInformation($"Model saved to: {ModelFile}");
		}
	}

	public void Evaluate(string trainFile, double percentTrain, string testFile, string featureDefFile)
	{
		var train = new List<RankList>();
		var validation = new List<RankList>();
		var features = PrepareSplit(trainFile, featureDefFile, percentTrain, normalize, train, validation);
		var test = !string.IsNullOrEmpty(testFile) ? ReadInput(testFile) : null;

		if (normalize && test != null)
		{
			Normalize(test, features);
		}

		var trainer = new RankerTrainer(LoggerFactory);
		var ranker = trainer.Train(RankerType, train, validation, features, TestScorer);

		if (test != null)
		{
			var rankScore = Evaluate(ranker, test);
			Logger.LogInformation($"{TestScorer.Name} on test data: {Math.Round(rankScore, 4)}");
		}

		if (!string.IsNullOrEmpty(ModelFile))
		{
			ranker.Save(ModelFile);
			Logger.LogInformation($"Model saved to: {ModelFile}");
		}
	}

	public void Evaluate(string sampleFile, string featureDefFile, int nFold, string modelDir, string modelFile) => Evaluate(sampleFile, featureDefFile, nFold, -1, modelDir, modelFile);

	public void Evaluate(string sampleFile, string featureDefFile, int nFold, float tvs, string modelDir, string modelFile)
	{
		var trainingData = new List<List<RankList>>();
		var validationData = new List<List<RankList>>();
		var testData = new List<List<RankList>>();
		var samples = ReadInput(sampleFile);
		var features = ReadFeature(featureDefFile) ?? FeatureManager.GetFeatureFromSampleVector(samples);

		FeatureManager.PrepareCV(samples, nFold, tvs, trainingData, validationData, testData);

		if (normalize)
		{
			for (var i = 0; i < nFold; i++)
			{
				NormalizeAll(trainingData, features);
				NormalizeAll(validationData, features);
				NormalizeAll(testData, features);
			}
		}

		double scoreOnTrain = 0.0, scoreOnTest = 0.0, totalScoreOnTest = 0.0;
		var totalTestSampleSize = 0;

		var scores = new double[nFold][];

		for (var i = 0; i < nFold; i++)
		{
			scores[i] = new double[2];
			var train = trainingData[i];
			var validation = tvs > 0 ? validationData[i] : null;
			var test = testData[i];

			var trainer = new RankerTrainer(LoggerFactory);
			var ranker = trainer.Train(RankerType, train, validation, features, TestScorer);

			var testScore = Evaluate(ranker, test);
			scoreOnTrain += ranker.GetScoreOnTrainingData();
			scoreOnTest += testScore;
			totalScoreOnTest += testScore * test.Count;
			totalTestSampleSize += test.Count;

			scores[i][0] = ranker.GetScoreOnTrainingData();
			scores[i][1] = testScore;

			if (!string.IsNullOrEmpty(modelDir))
			{
				ranker.Save(Path.Combine(modelDir, $"f{i + 1}.{modelFile}"));
				Logger.LogInformation($"Fold-{i + 1} model saved to: {modelFile}");
			}
		}

		Logger.LogInformation("Summary:");
		Logger.LogInformation($"{TestScorer.Name}\t|   Train\t| Test");

		for (var i = 0; i < nFold; i++)
		{
			Logger.LogInformation($"Fold {i + 1}\t|   {Math.Round(scores[i][0], 4)}\t|  {Math.Round(scores[i][1], 4)}\t");
		}

		Logger.LogInformation($"Avg.\t|   {Math.Round(scoreOnTrain / nFold, 4)}\t|  {Math.Round(scoreOnTest / nFold, 4)}\t");
		Logger.LogInformation($"Total\t|   \t\t|  {Math.Round(totalScoreOnTest / totalTestSampleSize, 4)}\t");
	}

	public void Test(string testFile)
	{
		var test = ReadInput(testFile);
		var rankScore = Evaluate(null, test);
		Logger.LogInformation($"{TestScorer.Name} on test data: {Math.Round(rankScore, 4)}");
	}

	public void Test(string testFile, string prpFile)
	{
		var test = ReadInput(testFile);
		var rankScore = 0.0;
		var ids = new List<string>();
		var scores = new List<double>();

		foreach (var l in test)
		{
			var score = TestScorer.Score(l);
			ids.Add(l.Id);
			scores.Add(score);
			rankScore += score;
		}

		rankScore /= test.Count;
		ids.Add("all");
		scores.Add(rankScore);

		Logger.LogInformation($"{TestScorer.Name} on test data: {Math.Round(rankScore, 4)}");

		if (!string.IsNullOrEmpty(prpFile))
		{
			SavePerRankListPerformanceFile(ids, scores, prpFile);
			Logger.LogInformation($"Per-ranked list performance saved to: {prpFile}");
		}
	}

	public void Test(string modelFile, string testFile, string prpFile)
	{
		var ranker = RankerFactory.LoadRankerFromFile(modelFile);
		var features = ranker.Features;
		var test = ReadInput(testFile);

		if (normalize)
		{
			Normalize(test, features);
		}

		var rankScore = 0.0;
		var ids = new List<string>();
		var scores = new List<double>();

		foreach (var aTest in test)
		{
			var rankedList = ranker.Rank(aTest);
			var score = TestScorer.Score(rankedList);
			ids.Add(rankedList.Id);
			scores.Add(score);
			rankScore += score;
		}

		rankScore /= test.Count;
		ids.Add("all");
		scores.Add(rankScore);

		Logger.LogInformation($"{TestScorer.Name} on test data: {SimpleMath.Round(rankScore, 4)}");

		if (!string.IsNullOrEmpty(prpFile))
		{
			SavePerRankListPerformanceFile(ids, scores, prpFile);
			Logger.LogInformation($"Per-ranked list performance saved to: {prpFile}");
		}
	}

	public void Test(List<string> modelFiles, string testFile, string prpFile)
	{
		var trainingData = new List<List<RankList>>();
		var testData = new List<List<RankList>>();

		var nFold = modelFiles.Count;
		var samples = ReadInput(testFile);

		Logger.LogInformation($"Preparing {nFold}-fold test data... ");
		FeatureManager.PrepareCV(samples, nFold, trainingData, testData);

		var rankScore = 0.0;
		var ids = new List<string>();
		var scores = new List<double>();

		for (var f = 0; f < nFold; f++)
		{
			var test = testData[f];
			var ranker = RankerFactory.LoadRankerFromFile(modelFiles[f]);
			var features = ranker.Features;

			if (normalize)
			{
				Normalize(test, features);
			}

			foreach (var aTest in test)
			{
				var rankedList = ranker.Rank(aTest);
				var score = TestScorer.Score(rankedList);
				ids.Add(rankedList.Id);
				scores.Add(score);
				rankScore += score;
			}
		}

		rankScore /= ids.Count;
		ids.Add("all");
		scores.Add(rankScore);

		Logger.LogInformation($"{TestScorer.Name} on test data: {Math.Round(rankScore, 4)}");

		if (!string.IsNullOrEmpty(prpFile))
		{
			SavePerRankListPerformanceFile(ids, scores, prpFile);
			Logger.LogInformation($"Per-ranked list performance saved to: {prpFile}");
		}
	}

	public void Test(List<string> modelFiles, List<string> testFiles, string prpFile)
	{
		var nFold = modelFiles.Count;
		var rankScore = 0.0;
		var ids = new List<string>();
		var scores = new List<double>();

		for (var f = 0; f < nFold; f++)
		{
			var test = ReadInput(testFiles[f]);
			var ranker = RankerFactory.LoadRankerFromFile(modelFiles[f]);
			var features = ranker.Features;

			if (normalize)
			{
				Normalize(test, features);
			}

			foreach (var aTest in test)
			{
				var rankedList = ranker.Rank(aTest);
				var score = TestScorer.Score(rankedList);
				ids.Add(rankedList.Id);
				scores.Add(score);
				rankScore += score;
			}
		}

		rankScore /= ids.Count;
		ids.Add("all");
		scores.Add(rankScore);

		Logger.LogInformation($"{TestScorer.Name} on test data: {Math.Round(rankScore, 4)}");

		if (!string.IsNullOrEmpty(prpFile))
		{
			SavePerRankListPerformanceFile(ids, scores, prpFile);
			Logger.LogInformation($"Per-ranked list performance saved to: {prpFile}");
		}
	}

	public void TestWithScoreFile(string testFile, string scoreFile)
	{
		try
		{
			using (var inReader = FileUtils.SmartReader(scoreFile))
			{
				var test = ReadInput(testFile);
				var scores = new List<double>();

				while (inReader.ReadLine() is { } content)
				{
					content = content.Trim();
					if (!string.IsNullOrEmpty(content))
					{
						scores.Add(double.Parse(content));
					}
				}

				var k = 0;
				for (var i = 0; i < test.Count; i++)
				{
					var rl = test[i];
					var scoreArray = new double[rl.Count];

					for (var j = 0; j < rl.Count; j++)
					{
						scoreArray[j] = scores[k++];
					}

					test[i] = new RankList(rl, MergeSorter.Sort(scoreArray, false));
				}

				var rankScore = Evaluate(null, test);
				Logger.LogInformation($"{TestScorer.Name} on test data: {Math.Round(rankScore, 4)}");
			}
		}
		catch (IOException e)
		{
			throw RankLibError.Create(e);
		}
	}

	public void Score(string modelFile, string testFile, string outputFile)
	{
		var ranker = RankerFactory.LoadRankerFromFile(modelFile);
		var features = ranker.Features;
		var test = ReadInput(testFile);

		if (normalize)
		{
			Normalize(test, features);
		}

		try
		{
			using (var outWriter = new StreamWriter(new FileStream(outputFile, FileMode.Create), System.Text.Encoding.UTF8))
			{
				foreach (var l in test)
				{
					for (var j = 0; j < l.Count; j++)
					{
						outWriter.WriteLine($"{l.Id}\t{j}\t{ranker.Eval(l[j])}");
					}
				}
			}
		}
		catch (IOException ex)
		{
			throw RankLibError.Create("Error in Evaluator::Score(): ", ex);
		}
	}

	public void Score(List<string> modelFiles, string testFile, string outputFile)
	{
		var trainingData = new List<List<RankList>>();
		var testData = new List<List<RankList>>();
		var nFold = modelFiles.Count;
		var samples = ReadInput(testFile);

		Logger.LogInformation($"Preparing {nFold}-fold test data...");
		FeatureManager.PrepareCV(samples, nFold, trainingData, testData);

		try
		{
			using (var outWriter = new StreamWriter(new FileStream(outputFile, FileMode.Create), System.Text.Encoding.UTF8))
			{
				for (var f = 0; f < nFold; f++)
				{
					var test = testData[f];
					var ranker = RankerFactory.LoadRankerFromFile(modelFiles[f]);
					var features = ranker.Features;

					if (normalize)
					{
						Normalize(test, features);
					}

					foreach (var l in test)
					{
						for (var j = 0; j < l.Count; j++)
						{
							outWriter.WriteLine($"{l.Id}\t{j}\t{ranker.Eval(l[j])}");
						}
					}
				}
			}
		}
		catch (IOException ex)
		{
			throw RankLibError.Create("Error in Evaluator::Score(): ", ex);
		}
	}

	public void Score(List<string> modelFiles, List<string> testFiles, string outputFile)
	{
		var nFold = modelFiles.Count;

		try
		{
			using (var outWriter = new StreamWriter(new FileStream(outputFile, FileMode.Create), System.Text.Encoding.UTF8))
			{
				for (var f = 0; f < nFold; f++)
				{
					var test = ReadInput(testFiles[f]);
					var ranker = RankerFactory.LoadRankerFromFile(modelFiles[f]);
					var features = ranker.Features;

					if (normalize)
					{
						Normalize(test, features);
					}

					foreach (var l in test)
					{
						for (var j = 0; j < l.Count; j++)
						{
							outWriter.WriteLine($"{l.Id}\t{j}\t{ranker.Eval(l[j])}");
						}
					}
				}
			}
		}
		catch (IOException ex)
		{
			throw RankLibError.Create("Error in Evaluator::Score(): ", ex);
		}
	}

	public void Rank(string modelFile, string testFile, string indriRanking)
	{
		var ranker = RankerFactory.LoadRankerFromFile(modelFile);
		var features = ranker.Features;
		var test = ReadInput(testFile);

		if (normalize)
		{
			Normalize(test, features);
		}

		try
		{
			using var outWriter = new StreamWriter(new FileStream(indriRanking, FileMode.Create), Encoding.UTF8);
			foreach (var l in test)
			{
				var scores = new double[l.Count];
				for (var j = 0; j < l.Count; j++)
				{
					scores[j] = ranker.Eval(l[j]);
				}

				var idx = MergeSorter.Sort(scores, false);
				for (var j = 0; j < idx.Length; j++)
				{
					var k = idx[j];
					var str = $"{l.Id} Q0 {l[k].GetDescription().Replace("#", "").Trim()} {(j + 1)} {SimpleMath.Round(scores[k], 5)} indri";
					outWriter.WriteLine(str);
				}
			}
		}
		catch (IOException ex)
		{
			throw RankLibError.Create("Error in Evaluator::Rank(): ", ex);
		}
	}

	public void Rank(string testFile, string indriRanking)
	{
		var test = ReadInput(testFile);

		try
		{
			using (var outWriter = new StreamWriter(new FileStream(indriRanking, FileMode.Create), System.Text.Encoding.UTF8))
			{
				foreach (var l in test)
				{
					for (var j = 0; j < l.Count; j++)
					{
						var str = $"{l.Id} Q0 {l[j].GetDescription().Replace("#", "").Trim()} {(j + 1)} {SimpleMath.Round(1.0 - 0.0001 * j, 5)} indri";
						outWriter.WriteLine(str);
					}
				}
			}
		}
		catch (IOException ex)
		{
			throw RankLibError.Create("Error in Evaluator::Rank(): ", ex);
		}
	}

	public void Rank(List<string> modelFiles, string testFile, string indriRanking)
	{
		var trainingData = new List<List<RankList>>();
		var testData = new List<List<RankList>>();
		var nFold = modelFiles.Count;
		var samples = ReadInput(testFile);

		Logger.LogInformation($"Preparing {nFold}-fold test data...");
		FeatureManager.PrepareCV(samples, nFold, trainingData, testData);

		try
		{
			using (var outWriter = new StreamWriter(new FileStream(indriRanking, FileMode.Create), System.Text.Encoding.UTF8))
			{
				for (var f = 0; f < nFold; f++)
				{
					var test = testData[f];
					var ranker = RankerFactory.LoadRankerFromFile(modelFiles[f]);
					var features = ranker.Features;

					if (normalize)
					{
						Normalize(test, features);
					}

					foreach (var l in test)
					{
						var scores = new double[l.Count];
						for (var j = 0; j < l.Count; j++)
						{
							scores[j] = ranker.Eval(l[j]);
						}

						var idx = MergeSorter.Sort(scores, false);
						for (var j = 0; j < idx.Length; j++)
						{
							var k = idx[j];
							var str = $"{l.Id} Q0 {l[k].GetDescription().Replace("#", "").Trim()} {(j + 1)} {SimpleMath.Round(scores[k], 5)} indri";
							outWriter.WriteLine(str);
						}
					}
				}
			}
		}
		catch (IOException ex)
		{
			throw RankLibError.Create("Error in Evaluator::Rank(): ", ex);
		}
	}

	public void Rank(List<string> modelFiles, List<string> testFiles, string indriRanking)
	{
		var nFold = modelFiles.Count;

		try
		{
			using (var outWriter = new StreamWriter(new FileStream(indriRanking, FileMode.Create), System.Text.Encoding.UTF8))
			{
				for (var f = 0; f < nFold; f++)
				{
					var test = ReadInput(testFiles[f]);
					var ranker = RankerFactory.LoadRankerFromFile(modelFiles[f]);
					var features = ranker.Features;

					if (normalize)
					{
						Normalize(test, features);
					}

					foreach (var l in test)
					{
						var scores = new double[l.Count];
						for (var j = 0; j < l.Count; j++)
						{
							scores[j] = ranker.Eval(l[j]);
						}

						var idx = MergeSorter.Sort(scores, false);
						for (var j = 0; j < idx.Length; j++)
						{
							var k = idx[j];
							var str = $"{l.Id} Q0 {l[k].GetDescription().Replace("#", "").Trim()} {(j + 1)} {SimpleMath.Round(scores[k], 5)} indri";
							outWriter.WriteLine(str);
						}
					}
				}
			}
		}
		catch (IOException ex)
		{
			throw RankLibError.Create("Error in Evaluator::Rank(): ", ex);
		}
	}

	private int[] PrepareSplit(string sampleFile, string featureDefFile, double percentTrain, bool normalize, List<RankList> trainingData, List<RankList> testData)
	{
		var data = ReadInput(sampleFile);
		var features = ReadFeature(featureDefFile) ?? FeatureManager.GetFeatureFromSampleVector(data);

		if (normalize)
		{
			Normalize(data, features);
		}

		FeatureManager.PrepareSplit(data, percentTrain, trainingData, testData);
		return features;
	}

	public void SavePerRankListPerformanceFile(List<string> ids, List<double> scores, string prpFile)
	{
		using (var writer = new StreamWriter(prpFile))
		{
			for (var i = 0; i < ids.Count; i++)
			{
				writer.WriteLine($"{TestScorer.Name}   {ids[i]}   {scores[i]}");
			}
		}
	}
}
