using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Parsing;
using RankLib.Utilities;

namespace RankLib.Learning.Tree;

public class LambdaMART : Ranker
{
	private readonly ILogger<LambdaMART> _logger;

	// Parameters
	public static int nTrees = 1000; // number of trees
	public static float learningRate = 0.1F; // shrinkage
	public static int nThreshold = 256;
	public static int nRoundToStopEarly = 100;
	public static int nTreeLeaves = 10;
	public static int minLeafSupport = 1;

	// Local variables
	protected float[][] thresholds = null;
	protected Ensemble ensemble = null;
	protected double[] modelScores = null;
	protected double[][] modelScoresOnValidation = null;
	protected int bestModelOnValidation = int.MaxValue - 2;

	protected DataPoint[] martSamples = null;
	protected int[][] sortedIdx = null;
	protected FeatureHistogram hist = null;
	protected double[] pseudoResponses = null;
	protected double[] weights = null;
	protected internal double[] impacts = null;

	public LambdaMART(ILogger<LambdaMART>? logger = null) : base(logger) =>
		_logger = logger ?? NullLogger<LambdaMART>.Instance;

	public LambdaMART(List<RankList> samples, int[] features, MetricScorer scorer, ILogger<LambdaMART>? logger = null)
		: base(samples, features, scorer, logger) =>
		_logger = logger ?? NullLogger<LambdaMART>.Instance;

	public override void Init()
	{
		_logger.LogInformation("Initializing...");

		var dpCount = Samples.Sum(rl => rl.Count);
		var current = 0;
		martSamples = new DataPoint[dpCount];
		modelScores = new double[dpCount];
		pseudoResponses = new double[dpCount];
		impacts = new double[Features.Length];
		weights = new double[dpCount];

		for (var i = 0; i < Samples.Count; i++)
		{
			var rl = Samples[i];
			for (var j = 0; j < rl.Count; j++)
			{
				martSamples[current + j] = rl[j];
				modelScores[current + j] = 0.0F;
				pseudoResponses[current + j] = 0.0F;
				weights[current + j] = 0;
			}
			current += rl.Count;
		}

		// Sort samples by each feature
		sortedIdx = new int[Features.Length][];
		var threadPool = MyThreadPool.GetInstance();
		if (threadPool.Size() == 1)
		{
			SortSamplesByFeature(0, Features.Length - 1);
		}
		else
		{
			var partition = threadPool.Partition(Features.Length);
			for (var i = 0; i < partition.Length - 1; i++)
			{
				threadPool.Execute(new SortWorker(this, partition[i], partition[i + 1] - 1));
			}
			threadPool.Await();
		}

		thresholds = new float[Features.Length][];
		for (var f = 0; f < Features.Length; f++)
		{
			var values = new List<float>();
			var fmax = float.MinValue;
			var fmin = float.MaxValue;
			for (var i = 0; i < martSamples.Length; i++)
			{
				var k = sortedIdx[f][i];
				var fv = martSamples[k].GetFeatureValue(Features[f]);
				values.Add(fv);
				if (fmax < fv)
					fmax = fv;
				if (fmin > fv)
					fmin = fv;

				var j = i + 1;
				while (j < martSamples.Length && martSamples[sortedIdx[f][j]].GetFeatureValue(Features[f]) <= fv)
				{
					j++;
				}
				i = j - 1;
			}

			if (values.Count <= nThreshold || nThreshold == -1)
			{
				thresholds[f] = values.ToArray();
				thresholds[f] = thresholds[f].Concat(new float[] { float.MaxValue }).ToArray();
			}
			else
			{
				var step = Math.Abs(fmax - fmin) / nThreshold;
				thresholds[f] = new float[nThreshold + 1];
				thresholds[f][0] = fmin;
				for (var j = 1; j < nThreshold; j++)
				{
					thresholds[f][j] = thresholds[f][j - 1] + step;
				}
				thresholds[f][nThreshold] = float.MaxValue;
			}
		}

		if (ValidationSamples != null)
		{
			modelScoresOnValidation = new double[ValidationSamples.Count][];
			for (var i = 0; i < ValidationSamples.Count; i++)
			{
				modelScoresOnValidation[i] = new double[ValidationSamples[i].Count];
				Array.Fill(modelScoresOnValidation[i], 0);
			}
		}

		hist = new FeatureHistogram();
		hist.Construct(martSamples, pseudoResponses, sortedIdx, Features, thresholds, impacts);
		sortedIdx = null;
	}

	public override void Learn()
	{
		ensemble = new Ensemble();
		_logger.LogInformation("Training starts...");

		if (ValidationSamples != null)
		{
			PrintLogLn(new int[] { 7, 9, 9 }, new string[] { "#iter", Scorer.Name + "-T", Scorer.Name + "-V" });
		}
		else
		{
			PrintLogLn(new int[] { 7, 9 }, new string[] { "#iter", Scorer.Name + "-T" });
		}

		for (var m = 0; m < nTrees; m++)
		{
			PrintLog(new int[] { 7 }, new string[] { (m + 1).ToString() });
			ComputePseudoResponses();
			hist.Update(pseudoResponses);
			var rt = new RegressionTree(nTreeLeaves, martSamples, pseudoResponses, hist, minLeafSupport);
			rt.Fit();
			ensemble.Add(rt, learningRate);
			UpdateTreeOutput(rt);

			var leaves = rt.Leaves();
			for (var i = 0; i < leaves.Count; i++)
			{
				var s = leaves[i];
				var idx = s.GetSamples();
				for (var j = 0; j < idx.Length; j++)
				{
					modelScores[idx[j]] += learningRate * s.GetOutput();
				}
			}
			rt.ClearSamples();

			ScoreOnTrainingData = ComputeModelScoreOnTraining();
			PrintLog(new int[] { 9 }, new string[] { SimpleMath.Round(ScoreOnTrainingData, 4).ToString() });

			if (ValidationSamples != null)
			{
				for (var i = 0; i < modelScoresOnValidation.Length; i++)
				{
					for (var j = 0; j < modelScoresOnValidation[i].Length; j++)
					{
						var tempQualifier = ValidationSamples[i];
						modelScoresOnValidation[i][j] += learningRate * rt.Eval(tempQualifier[j]);
					}
				}
				double score = ComputeModelScoreOnValidation();
				PrintLog(new int[] { 9 }, new string[] { SimpleMath.Round(score, 4).ToString() });
				if (score > BestScoreOnValidationData)
				{
					BestScoreOnValidationData = score;
					bestModelOnValidation = ensemble.TreeCount() - 1;
				}
			}
			FlushLog();

			if (m - bestModelOnValidation > nRoundToStopEarly)
			{
				break;
			}
		}

		while (ensemble.TreeCount() > bestModelOnValidation + 1)
		{
			ensemble.Remove(ensemble.TreeCount() - 1);
		}

		ScoreOnTrainingData = Scorer.Score(Rank(Samples));
		_logger.LogInformation($"Finished successfully. {Scorer.Name} on training data: {SimpleMath.Round(ScoreOnTrainingData, 4)}");

		if (ValidationSamples != null)
		{
			BestScoreOnValidationData = Scorer.Score(Rank(ValidationSamples));
			_logger.LogInformation($"{Scorer.Name} on validation data: {SimpleMath.Round(BestScoreOnValidationData, 4)}");
		}

		_logger.LogInformation("-- FEATURE IMPACTS");
		var ftrsSorted = MergeSorter.Sort(impacts, false);
		foreach (var ftr in ftrsSorted)
		{
			_logger.LogInformation($"Feature {Features[ftr]} reduced error {impacts[ftr]}");
		}
	}

	public override double Eval(DataPoint dp) => ensemble.Eval(dp);

	public override Ranker CreateNew() => new LambdaMART(_logger);

	public override string ToString() => ensemble.ToString();

	public override string Model()
	{
		var output = new System.Text.StringBuilder();
		output.AppendLine($"## {Name}");
		output.AppendLine($"## No. of trees = {nTrees}");
		output.AppendLine($"## No. of leaves = {nTreeLeaves}");
		output.AppendLine($"## No. of threshold candidates = {nThreshold}");
		output.AppendLine($"## Learning rate = {learningRate}");
		output.AppendLine($"## Stop early = {nRoundToStopEarly}");
		output.AppendLine();
		output.AppendLine(ToString());
		return output.ToString();
	}

	public override void LoadFromString(string fullText)
	{
		var lineByLine = new ModelLineProducer();
		lineByLine.Parse(fullText, (model, endEns) => { });
		ensemble = new Ensemble(lineByLine.GetModel().ToString());
		Features = ensemble.GetFeatures();
	}

	public override void PrintParameters()
	{
		_logger.LogInformation($"No. of trees: {nTrees}");
		_logger.LogInformation($"No. of leaves: {nTreeLeaves}");
		_logger.LogInformation($"No. of threshold candidates: {nThreshold}");
		_logger.LogInformation($"Min leaf support: {minLeafSupport}");
		_logger.LogInformation($"Learning rate: {learningRate}");
		_logger.LogInformation($"Stop early: {nRoundToStopEarly} rounds without performance gain on validation data");
	}

	public override string Name => "LambdaMART";

	public Ensemble GetEnsemble() => ensemble;

	// Helper Methods
	protected virtual void ComputePseudoResponses()
	{
		Array.Fill(pseudoResponses, 0);
		Array.Fill(weights, 0);
		var p = MyThreadPool.GetInstance();
		if (p.Size() == 1)
		{
			ComputePseudoResponses(0, Samples.Count - 1, 0);
		}
		else
		{
			var workers = new List<LambdaComputationWorker>();
			var partition = p.Partition(Samples.Count);
			var current = 0;
			for (var i = 0; i < partition.Length - 1; i++)
			{
				var worker = new LambdaComputationWorker(this, partition[i], partition[i + 1] - 1, current);
				workers.Add(worker);
				p.Execute(worker);
				if (i < partition.Length - 2)
				{
					for (var j = partition[i]; j <= partition[i + 1] - 1; j++)
					{
						current += Samples[j].Count;
					}
				}
			}
			p.Await();
		}
	}

	protected void ComputePseudoResponses(int start, int end, int current)
	{
		var cutoff = Scorer.K;
		for (var i = start; i <= end; i++)
		{
			var orig = Samples[i];
			var idx = MergeSorter.Sort(modelScores, current, current + orig.Count - 1, false);
			var rl = new RankList(orig, idx, current);
			var changes = Scorer.SwapChange(rl);
			for (var j = 0; j < rl.Count; j++)
			{
				var p1 = rl[j];
				var mj = idx[j];
				for (var k = 0; k < rl.Count; k++)
				{
					if (j > cutoff && k > cutoff)
					{
						break;
					}
					var p2 = rl[k];
					var mk = idx[k];
					if (p1.Label > p2.Label)
					{
						var deltaNDCG = Math.Abs(changes[j][k]);
						if (deltaNDCG > 0)
						{
							var rho = 1.0 / (1 + Math.Exp(modelScores[mj] - modelScores[mk]));
							var lambda = rho * deltaNDCG;
							pseudoResponses[mj] += lambda;
							pseudoResponses[mk] -= lambda;
							var delta = rho * (1.0 - rho) * deltaNDCG;
							weights[mj] += delta;
							weights[mk] += delta;
						}
					}
				}
			}
			current += orig.Count;
		}
	}

	protected virtual void UpdateTreeOutput(RegressionTree rt)
	{
		var leaves = rt.Leaves();
		foreach (var s in leaves)
		{
			var s1 = 0F;
			var s2 = 0F;
			var idx = s.GetSamples();
			foreach (var k in idx)
			{
				s1 += Convert.ToSingle(pseudoResponses[k]);
				s2 += Convert.ToSingle(weights[k]);
			}
			if (s2 == 0)
				s.SetOutput(0);
			else
				s.SetOutput(s1 / s2);
		}
	}

	protected int[] SortSamplesByFeature(DataPoint[] samples, int fid)
	{
		var score = new double[samples.Length];
		for (var i = 0; i < samples.Length; i++)
		{
			score[i] = samples[i].GetFeatureValue(fid);
		}
		var idx = MergeSorter.Sort(score, true);
		return idx;
	}

	protected RankList Rank(int rankListIndex, int current)
	{
		var orig = Samples[rankListIndex];
		var scores = new double[orig.Count];
		for (var i = 0; i < scores.Length; i++)
		{
			scores[i] = modelScores[current + i];
		}
		var idx = MergeSorter.Sort(scores, false);
		return new RankList(orig, idx);
	}

	protected float ComputeModelScoreOnTraining() => ComputeModelScoreOnTraining(0, Samples.Count - 1, 0) / Samples.Count;

	protected float ComputeModelScoreOnTraining(int start, int end, int current)
	{
		float s = 0;
		var c = current;
		for (var i = start; i <= end; i++)
		{
			s += Convert.ToSingle(Scorer.Score(Rank(i, c)));
			c += Samples[i].Count;
		}
		return s;
	}

	protected float ComputeModelScoreOnValidation() => ComputeModelScoreOnValidation(0, ValidationSamples.Count - 1) / ValidationSamples.Count;

	protected float ComputeModelScoreOnValidation(int start, int end)
	{
		float score = 0;
		for (var i = start; i <= end; i++)
		{
			var idx = MergeSorter.Sort(modelScoresOnValidation[i], false);
			score += Convert.ToSingle(Scorer.Score(new RankList(ValidationSamples[i], idx)));
		}
		return score;
	}

	protected void SortSamplesByFeature(int fStart, int fEnd)
	{
		for (var i = fStart; i <= fEnd; i++)
		{
			sortedIdx[i] = SortSamplesByFeature(martSamples, Features[i]);
		}
	}

	private class SortWorker : RunnableTask
	{
		private readonly LambdaMART ranker;
		private readonly int start;
		private readonly int end;

		public SortWorker(LambdaMART ranker, int start, int end)
		{
			this.ranker = ranker;
			this.start = start;
			this.end = end;
		}

		public override void Run() => ranker.SortSamplesByFeature(start, end);
	}

	private class LambdaComputationWorker : RunnableTask
	{
		private readonly LambdaMART ranker;
		private readonly int rlStart;
		private readonly int rlEnd;
		private readonly int martStart;

		public LambdaComputationWorker(LambdaMART ranker, int rlStart, int rlEnd, int martStart)
		{
			this.ranker = ranker;
			this.rlStart = rlStart;
			this.rlEnd = rlEnd;
			this.martStart = martStart;
		}

		public override void Run() => ranker.ComputePseudoResponses(rlStart, rlEnd, martStart);
	}
}
