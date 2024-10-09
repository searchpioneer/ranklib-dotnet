using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Parsing;
using RankLib.Utilities;

namespace RankLib.Learning.Tree;

public class RFRankerParameters
{
	// Parameters
	// [a] general bagging parameters
	public int nBag { get; set; } = 300;
	public float subSamplingRate { get; set; } = 1.0f; // sampling of samples (*WITH* replacement)
	public float featureSamplingRate { get; set; } = 0.3f; // sampling of features (*WITHOUT* replacement)

	// [b] what to do in each bag
	public RankerType rType { get; set; } = RankerType.MART; // which algorithm to bag
	public int nTrees { get; set; } = 1; // how many trees in each bag
	public int nTreeLeaves { get; set; } = 100;
	public float learningRate { get; set; } = 0.1F; // or shrinkage, only matters if nTrees > 1
	public int nThreshold { get; set; } = 256;
	public int minLeafSupport { get; set; } = 1;
}

public class RFRanker : Ranker
{
	private readonly ILoggerFactory _loggerFactory;
	private readonly ILogger<RFRanker> _logger;

	// Parameters
	// [a] general bagging parameters
	public static int nBag = 300;
	public static float subSamplingRate = 1.0f; // sampling of samples (*WITH* replacement)
	public static float featureSamplingRate = 0.3f; // sampling of features (*WITHOUT* replacement)

	// [b] what to do in each bag
	public static RankerType rType = RankerType.MART; // which algorithm to bag
	public static int nTrees = 1; // how many trees in each bag
	public static int nTreeLeaves = 100;
	public static float learningRate = 0.1F; // or shrinkage, only matters if nTrees > 1
	public static int nThreshold = 256;
	public static int minLeafSupport = 1;

	// Variables
	protected Ensemble[] ensembles = []; // bag of ensembles
	private LambdaMARTParameters _lambdaMARTParameters;

	public RFRanker(ILoggerFactory? loggerFactory = null) : base((loggerFactory ?? NullLoggerFactory.Instance)
		.CreateLogger<RFRanker>())
	{
		_loggerFactory = loggerFactory ?? NullLoggerFactory.Instance;
		_logger = _loggerFactory.CreateLogger<RFRanker>();
	}

	public RFRanker(List<RankList> samples, int[] features, MetricScorer scorer, ILoggerFactory? loggerFactory = null)
		: base(samples, features, scorer, (loggerFactory ?? NullLoggerFactory.Instance).CreateLogger<RFRanker>())
	{
		_loggerFactory = loggerFactory ?? NullLoggerFactory.Instance;
		_logger = _loggerFactory.CreateLogger<RFRanker>();
	}

	public override void Init()
	{
		_logger.LogInformation("Initializing...");
		ensembles = new Ensemble[nBag];

		_lambdaMARTParameters = new LambdaMARTParameters
		{
			nTrees = nTrees,
			nTreeLeaves = nTreeLeaves,
			learningRate = learningRate,
			nThreshold = nThreshold,
			minLeafSupport = minLeafSupport,
			nRoundToStopEarly = -1, // no early stopping since we're doing bagging
		};

		// Turn on feature sampling
		FeatureHistogram.samplingRate = featureSamplingRate;
	}

	public override void Learn()
	{
		var rankerFactory = new RankerFactory(_loggerFactory);
		_logger.LogInformation("Training starts...");
		PrintLogLn(new int[] { 9, 9, 11 }, new string[] { "bag", Scorer.Name + "-B", Scorer.Name + "-OOB" });

		double[] impacts = null;

		// Start the bagging process
		for (var i = 0; i < nBag; i++)
		{
			var sp = new Sampler();
			// Create a "bag" of samples by random sampling from the training set
			var bag = sp.Sample(Samples, subSamplingRate, true);
			var r = (LambdaMART)rankerFactory.CreateRanker(rType, bag, Features, Scorer);

			r.Parameters = _lambdaMARTParameters;
			r.Init();
			r.Learn();

			// Accumulate impacts
			if (impacts == null)
			{
				impacts = r.impacts;
			}
			else
			{
				for (var ftr = 0; ftr < impacts.Length; ftr++)
				{
					impacts[ftr] += r.impacts[ftr];
				}
			}
			PrintLogLn(new int[] { 9, 9 }, new string[] { "b[" + (i + 1) + "]", SimpleMath.Round(r.GetScoreOnTrainingData(), 4).ToString() });
			ensembles[i] = r.GetEnsemble();
		}

		// Finishing up
		ScoreOnTrainingData = Scorer.Score(Rank(Samples));
		_logger.LogInformation("Finished successfully.");
		_logger.LogInformation(Scorer.Name + " on training data: " + SimpleMath.Round(ScoreOnTrainingData, 4));

		if (ValidationSamples != null)
		{
			BestScoreOnValidationData = Scorer.Score(Rank(ValidationSamples));
			_logger.LogInformation(Scorer.Name + " on validation data: " + SimpleMath.Round(BestScoreOnValidationData, 4));
		}

		// Print feature impacts
		_logger.LogInformation("-- FEATURE IMPACTS");
		if (_logger.IsEnabled(LogLevel.Information))
		{
			var ftrsSorted = MergeSorter.Sort(impacts, false);
			foreach (var ftr in ftrsSorted)
			{
				_logger.LogInformation(" Feature " + Features[ftr] + " reduced error " + impacts[ftr]);
			}
		}
	}

	public override double Eval(DataPoint dp)
	{
		double s = 0;
		foreach (var ensemble in ensembles)
		{
			s += ensemble.Eval(dp);
		}
		return s / ensembles.Length;
	}

	public override Ranker CreateNew() => new RFRanker(_loggerFactory);

	public override string ToString()
	{
		var builder = new StringBuilder();
		for (var i = 0; i < nBag; i++)
		{
			builder.Append(ensembles[i]).Append('\n');
		}
		return builder.ToString();
	}

	public override string Model
	{
		get
		{
			var output = new StringBuilder();
			output.Append("## " + Name + "\n");
			output.Append("## No. of bags = " + nBag + "\n");
			output.Append("## Sub-sampling = " + subSamplingRate + "\n");
			output.Append("## Feature-sampling = " + featureSamplingRate + "\n");
			output.Append("## No. of trees = " + nTrees + "\n");
			output.Append("## No. of leaves = " + nTreeLeaves + "\n");
			output.Append("## No. of threshold candidates = " + nThreshold + "\n");
			output.Append("## Learning rate = " + learningRate + "\n\n");
			output.Append(ToString());
			return output.ToString();
		}
	}

	public override void LoadFromString(string fullText)
	{
		var ens = new List<Ensemble>();
		var lineByLine = new ModelLineProducer();

		lineByLine.Parse(fullText, (model, maybeEndEns) =>
		{
			if (maybeEndEns && model.ToString().EndsWith("</ensemble>"))
			{
				ens.Add(new Ensemble(model.ToString()));
				model.Clear();
			}
		});

		var uniqueFeatures = new HashSet<int>();
		ensembles = new Ensemble[ens.Count];
		for (var i = 0; i < ens.Count; i++)
		{
			ensembles[i] = ens[i];

			// Obtain used features
			var fids = ens[i].Features;
			foreach (var fid in fids)
			{
				uniqueFeatures.Add(fid);
			}
		}

		Features = uniqueFeatures.ToArray();
	}

	public override void PrintParameters()
	{
		_logger.LogInformation("No. of bags: " + nBag);
		_logger.LogInformation("Sub-sampling: " + subSamplingRate);
		_logger.LogInformation("Feature-sampling: " + featureSamplingRate);
		_logger.LogInformation("No. of trees: " + nTrees);
		_logger.LogInformation("No. of leaves: " + nTreeLeaves);
		_logger.LogInformation("No. of threshold candidates: " + nThreshold);
		_logger.LogInformation("Learning rate: " + learningRate);
	}

	public override string Name => "Random Forests";

	public Ensemble[] Ensembles => ensembles;
}
