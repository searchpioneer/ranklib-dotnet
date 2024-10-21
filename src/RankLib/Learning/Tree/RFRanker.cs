using System.Globalization;
using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Parsing;
using RankLib.Utilities;

namespace RankLib.Learning.Tree;

public class RFRankerParameters : IRankerParameters
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
	public void Log(ILogger logger)
	{
		logger.LogInformation("No. of bags: " + nBag);
		logger.LogInformation("Sub-sampling: " + subSamplingRate);
		logger.LogInformation("Feature-sampling: " + featureSamplingRate);
		logger.LogInformation("No. of trees: " + nTrees);
		logger.LogInformation("No. of leaves: " + nTreeLeaves);
		logger.LogInformation("No. of threshold candidates: " + nThreshold);
		logger.LogInformation("Learning rate: " + learningRate);
	}
}

public class RFRanker : Ranker<RFRankerParameters>
{
	internal const string RankerName = "Random Forests";

	private readonly ILoggerFactory _loggerFactory;
	private readonly ILogger<RFRanker> _logger;
	private LambdaMARTParameters _lambdaMARTParameters;

	public Ensemble[] Ensembles { get; private set; } = [];

	public override string Name => RankerName;

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

	public override Task Init()
	{
		_logger.LogInformation("Initializing...");
		Ensembles = new Ensemble[Parameters.nBag];
		_lambdaMARTParameters = new LambdaMARTParameters
		{
			nTrees = Parameters.nTrees,
			nTreeLeaves = Parameters.nTreeLeaves,
			learningRate = Parameters.learningRate,
			nThreshold = Parameters.nThreshold,
			minLeafSupport = Parameters.minLeafSupport,
			nRoundToStopEarly = -1, // no early stopping since we're doing bagging
		};

		// Turn on feature sampling
		FeatureHistogram.samplingRate = Parameters.featureSamplingRate;

		return Task.CompletedTask;
	}

	public override async Task Learn()
	{
		var rankerFactory = new RankerFactory(_loggerFactory);
		_logger.LogInformation("Training starts...");
		PrintLogLn([9, 9, 11], ["bag", Scorer.Name + "-B", Scorer.Name + "-OOB"]);

		double[]? impacts = null;

		// Start the bagging process
		for (var i = 0; i < Parameters.nBag; i++)
		{
			// Create a "bag" of samples by random sampling from the training set
			var (bag, _) = Sampler.Sample(Samples, Parameters.subSamplingRate, true);
			var r = (LambdaMART)rankerFactory.CreateRanker(Parameters.rType, bag, Features, Scorer);

			r.Parameters = _lambdaMARTParameters;
			await r.Init();
			await r.Learn();

			// Accumulate impacts
			if (impacts == null)
			{
				impacts = r.Impacts;
			}
			else
			{
				for (var ftr = 0; ftr < impacts.Length; ftr++)
				{
					impacts[ftr] += r.Impacts[ftr];
				}
			}
			PrintLogLn([9, 9], ["b[" + (i + 1) + "]", SimpleMath.Round(r.GetScoreOnTrainingData(), 4).ToString(CultureInfo.InvariantCulture)]);
			Ensembles[i] = r.GetEnsemble();
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
			var ftrsSorted = MergeSorter.Sort(impacts!, false);
			foreach (var ftr in ftrsSorted)
			{
				_logger.LogInformation(" Feature " + Features[ftr] + " reduced error " + impacts![ftr]);
			}
		}
	}

	public override double Eval(DataPoint dataPoint)
	{
		double s = 0;
		foreach (var ensemble in Ensembles)
		{
			s += ensemble.Eval(dataPoint);
		}
		return s / Ensembles.Length;
	}

	public override string ToString()
	{
		var builder = new StringBuilder();
		for (var i = 0; i < Parameters.nBag; i++)
		{
			builder.Append(Ensembles[i]).Append('\n');
		}
		return builder.ToString();
	}

	public override string Model
	{
		get
		{
			var output = new StringBuilder();
			output.Append("## " + Name + "\n");
			output.Append("## No. of bags = " + Parameters.nBag + "\n");
			output.Append("## Sub-sampling = " + Parameters.subSamplingRate + "\n");
			output.Append("## Feature-sampling = " + Parameters.featureSamplingRate + "\n");
			output.Append("## No. of trees = " + Parameters.nTrees + "\n");
			output.Append("## No. of leaves = " + Parameters.nTreeLeaves + "\n");
			output.Append("## No. of threshold candidates = " + Parameters.nThreshold + "\n");
			output.Append("## Learning rate = " + Parameters.learningRate + "\n\n");
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
		Ensembles = new Ensemble[ens.Count];
		for (var i = 0; i < ens.Count; i++)
		{
			Ensembles[i] = ens[i];

			// Obtain used features
			var fids = ens[i].Features;
			foreach (var fid in fids)
			{
				uniqueFeatures.Add(fid);
			}
		}

		Features = uniqueFeatures.ToArray();
	}
}
