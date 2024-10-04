using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Parsing;
using RankLib.Utilities;

namespace RankLib.Learning.Tree;

public class RFRanker : Ranker
{
	private static readonly ILogger<RFRanker> logger = NullLogger<RFRanker>.Instance;

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
	protected Ensemble[] ensembles = null; // bag of ensembles

	public RFRanker() : base() { }

	public RFRanker(List<RankList> samples, int[] features, MetricScorer scorer)
		: base(samples, features, scorer) { }

	public override void Init()
	{
		logger.LogInformation("Initializing...");
		ensembles = new Ensemble[nBag];

		// Initialize parameters for the tree(s) built in each bag
		LambdaMART.nTrees = nTrees;
		LambdaMART.nTreeLeaves = nTreeLeaves;
		LambdaMART.learningRate = learningRate;
		LambdaMART.nThreshold = nThreshold;
		LambdaMART.minLeafSupport = minLeafSupport;
		LambdaMART.nRoundToStopEarly = -1; // no early stopping since we're doing bagging

		// Turn on feature sampling
		FeatureHistogram.samplingRate = featureSamplingRate;
	}

	public override void Learn()
	{
		var rf = new RankerFactory();
		logger.LogInformation("Training starts...");
		PrintLogLn(new int[] { 9, 9, 11 }, new string[] { "bag", Scorer.Name() + "-B", Scorer.Name() + "-OOB" });

		double[] impacts = null;

		// Start the bagging process
		for (var i = 0; i < nBag; i++)
		{
			var sp = new Sampler();
			// Create a "bag" of samples by random sampling from the training set
			var bag = sp.DoSampling(Samples, subSamplingRate, true);
			var r = (LambdaMART)rf.CreateRanker(rType, bag, Features, Scorer);

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
		logger.LogInformation("Finished successfully.");
		logger.LogInformation(Scorer.Name() + " on training data: " + SimpleMath.Round(ScoreOnTrainingData, 4));

		if (ValidationSamples != null)
		{
			BestScoreOnValidationData = Scorer.Score(Rank(ValidationSamples));
			logger.LogInformation(Scorer.Name() + " on validation data: " + SimpleMath.Round(BestScoreOnValidationData, 4));
		}

		// Print feature impacts
		logger.LogInformation("-- FEATURE IMPACTS");
		if (logger.IsEnabled(LogLevel.Information))
		{
			var ftrsSorted = MergeSorter.Sort(impacts, false);
			foreach (var ftr in ftrsSorted)
			{
				logger.LogInformation(" Feature " + Features[ftr] + " reduced error " + impacts[ftr]);
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

	public override Ranker CreateNew() => new RFRanker();

	public override string ToString()
	{
		var str = new StringBuilder();
		for (var i = 0; i < nBag; i++)
		{
			str.Append(ensembles[i]).Append("\n");
		}
		return str.ToString();
	}

	public override string Model()
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

		ISet<int> uniqueFeatures = new HashSet<int>();
		ensembles = new Ensemble[ens.Count];
		for (var i = 0; i < ens.Count; i++)
		{
			ensembles[i] = ens[i];

			// Obtain used features
			var fids = ens[i].GetFeatures();
			foreach (var fid in fids)
			{
				uniqueFeatures.Add(fid);
			}
		}

		Features = uniqueFeatures.ToArray();
	}

	public override void PrintParameters()
	{
		logger.LogInformation("No. of bags: " + nBag);
		logger.LogInformation("Sub-sampling: " + subSamplingRate);
		logger.LogInformation("Feature-sampling: " + featureSamplingRate);
		logger.LogInformation("No. of trees: " + nTrees);
		logger.LogInformation("No. of leaves: " + nTreeLeaves);
		logger.LogInformation("No. of threshold candidates: " + nThreshold);
		logger.LogInformation("Learning rate: " + learningRate);
	}

	public override string Name => "Random Forests";

	public Ensemble[] GetEnsembles() => ensembles;
}
