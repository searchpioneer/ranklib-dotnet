using System.Globalization;
using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Parsing;
using RankLib.Utilities;

namespace RankLib.Learning.Tree;

public class LambdaMARTParameters : IRankerParameters
{
	public int nTrees { get; set; } = 1000; // number of trees
	public float learningRate { get; set; } = 0.1F; // shrinkage
	public int nThreshold { get; set; } = 256;
	public int nRoundToStopEarly { get; set; } = 100;
	public int nTreeLeaves { get; set; } = 10;
	public int minLeafSupport { get; set; } = 1;
	public float SamplingRate { get; set; } = 1;
	public int MaxDegreeOfParallelism { get; set; } = Environment.ProcessorCount;

	public void Log(ILogger logger)
	{
		logger.LogInformation($"No. of trees: {nTrees}");
		logger.LogInformation($"No. of leaves: {nTreeLeaves}");
		logger.LogInformation($"No. of threshold candidates: {nThreshold}");
		logger.LogInformation($"Min leaf support: {minLeafSupport}");
		logger.LogInformation($"Learning rate: {learningRate}");
		logger.LogInformation($"Stop early: {nRoundToStopEarly} rounds without performance gain on validation data");
	}
}

public class LambdaMART : Ranker, IRanker<LambdaMARTParameters>
{
	internal const string RankerName = "LambdaMART";

	private readonly ILogger<LambdaMART> _logger;

	private float[][] _thresholds = [];
	private Ensemble _ensemble = null;
	private double[][] _modelScoresOnValidation = [];
	private int _bestModelOnValidation = int.MaxValue - 2;
	private int[][] _sortedIdx = [];
	private FeatureHistogram _hist;
	private double[] _weights = [];

	protected double[] ModelScores = [];
	protected DataPoint[] MARTSamples = [];
	protected internal double[] Impacts = [];
	protected double[] PseudoResponses = [];

	public LambdaMARTParameters Parameters { get; set; }

	IRankerParameters IRanker.Parameters
	{
		get => Parameters;
		set => Parameters = (LambdaMARTParameters)value;
	}

	public override string Name => RankerName;

	public LambdaMART(ILogger<LambdaMART>? logger = null) : this(new LambdaMARTParameters(), logger)
	{
	}

	public LambdaMART(LambdaMARTParameters parameters, ILogger<LambdaMART>? logger = null) : base(logger)
	{
		Parameters = parameters;
		_logger = logger ?? NullLogger<LambdaMART>.Instance;
	}

	public LambdaMART(LambdaMARTParameters parameters, List<RankList> samples, int[] features, MetricScorer scorer, ILogger<LambdaMART>? logger = null)
		: base(samples, features, scorer, logger)
	{
		Parameters = parameters;
		_logger = logger ?? NullLogger<LambdaMART>.Instance;
	}

	public LambdaMART(List<RankList> samples, int[] features, MetricScorer scorer, ILogger<LambdaMART>? logger = null) :
		this(new LambdaMARTParameters(), samples, features, scorer, logger)
	{
	}

	public override async Task Init()
	{
		_logger.LogInformation("Initializing...");

		var dpCount = Samples.Sum(rl => rl.Count);
		var current = 0;
		MARTSamples = new DataPoint[dpCount];
		ModelScores = new double[dpCount];
		PseudoResponses = new double[dpCount];
		Impacts = new double[Features.Length];
		_weights = new double[dpCount];

		for (var i = 0; i < Samples.Count; i++)
		{
			var rl = Samples[i];
			for (var j = 0; j < rl.Count; j++)
			{
				MARTSamples[current + j] = rl[j];
				ModelScores[current + j] = 0.0F;
				PseudoResponses[current + j] = 0.0F;
				_weights[current + j] = 0;
			}
			current += rl.Count;
		}

		// Sort samples by each feature
		_sortedIdx = new int[Features.Length][];
		if (Parameters.MaxDegreeOfParallelism == 1)
			SortSamplesByFeature(0, Features.Length - 1);
		else
		{
			var tasks = ParallelExecutor
				.PartitionEnumerable(Features.Length, Parameters.MaxDegreeOfParallelism)
				.Select(range => new SortWorker(this, range.Start.Value, range.End.Value));
			await ParallelExecutor.ExecuteAsync(tasks, Parameters.MaxDegreeOfParallelism);
		}

		_thresholds = new float[Features.Length][];
		for (var f = 0; f < Features.Length; f++)
		{
			var values = new List<float>();
			var fmax = float.MinValue;
			var fmin = float.MaxValue;
			for (var i = 0; i < MARTSamples.Length; i++)
			{
				var k = _sortedIdx[f][i];
				var fv = MARTSamples[k].GetFeatureValue(Features[f]);
				values.Add(fv);
				if (fmax < fv)
					fmax = fv;
				if (fmin > fv)
					fmin = fv;

				var j = i + 1;
				while (j < MARTSamples.Length && MARTSamples[_sortedIdx[f][j]].GetFeatureValue(Features[f]) <= fv)
				{
					j++;
				}
				i = j - 1;
			}

			if (values.Count <= Parameters.nThreshold || Parameters.nThreshold == -1)
			{
				_thresholds[f] = values.ToArray();
				_thresholds[f] = _thresholds[f].Concat([float.MaxValue]).ToArray();
			}
			else
			{
				var step = Math.Abs(fmax - fmin) / Parameters.nThreshold;
				_thresholds[f] = new float[Parameters.nThreshold + 1];
				_thresholds[f][0] = fmin;
				for (var j = 1; j < Parameters.nThreshold; j++)
				{
					_thresholds[f][j] = _thresholds[f][j - 1] + step;
				}
				_thresholds[f][Parameters.nThreshold] = float.MaxValue;
			}
		}

		if (ValidationSamples != null)
		{
			_modelScoresOnValidation = new double[ValidationSamples.Count][];
			for (var i = 0; i < ValidationSamples.Count; i++)
			{
				_modelScoresOnValidation[i] = new double[ValidationSamples[i].Count];
				Array.Fill(_modelScoresOnValidation[i], 0);
			}
		}

		_hist = new FeatureHistogram(Parameters.SamplingRate, Parameters.MaxDegreeOfParallelism);
		await _hist.Construct(MARTSamples, PseudoResponses, _sortedIdx, Features, _thresholds, Impacts);
		_sortedIdx = [];
	}

	public override async Task Learn()
	{
		_ensemble = new Ensemble();
		_logger.LogInformation("Training starts...");

		if (ValidationSamples != null)
		{
			PrintLogLn([7, 9, 9], ["#iter", Scorer.Name + "-T", Scorer.Name + "-V"]);
		}
		else
		{
			PrintLogLn([7, 9], ["#iter", Scorer.Name + "-T"]);
		}

		for (var m = 0; m < Parameters.nTrees; m++)
		{
			PrintLog([7], [(m + 1).ToString()]);
			await ComputePseudoResponses();
			await _hist.Update(PseudoResponses);
			var tree = new RegressionTree(Parameters.nTreeLeaves, MARTSamples, PseudoResponses, _hist, Parameters.minLeafSupport);
			await tree.Fit();
			_ensemble.Add(tree, Parameters.learningRate);
			UpdateTreeOutput(tree);

			var leaves = tree.Leaves;
			for (var i = 0; i < leaves.Count; i++)
			{
				var s = leaves[i];
				var idx = s.GetSamples();
				for (var j = 0; j < idx.Length; j++)
				{
					ModelScores[idx[j]] += Parameters.learningRate * s.GetOutput();
				}
			}
			tree.ClearSamples();

			ScoreOnTrainingData = ComputeModelScoreOnTraining();
			PrintLog([9], [SimpleMath.Round(ScoreOnTrainingData, 4).ToString(CultureInfo.InvariantCulture)]);

			if (ValidationSamples != null)
			{
				for (var i = 0; i < _modelScoresOnValidation.Length; i++)
				{
					for (var j = 0; j < _modelScoresOnValidation[i].Length; j++)
					{
						var tempQualifier = ValidationSamples[i];
						_modelScoresOnValidation[i][j] += Parameters.learningRate * tree.Eval(tempQualifier[j]);
					}
				}
				double score = ComputeModelScoreOnValidation();
				PrintLog([9], [SimpleMath.Round(score, 4).ToString(CultureInfo.InvariantCulture)]);
				if (score > BestScoreOnValidationData)
				{
					BestScoreOnValidationData = score;
					_bestModelOnValidation = _ensemble.TreeCount - 1;
				}
			}
			FlushLog();

			if (m - _bestModelOnValidation > Parameters.nRoundToStopEarly)
			{
				break;
			}
		}

		while (_ensemble.TreeCount > _bestModelOnValidation + 1)
		{
			_ensemble.Remove(_ensemble.TreeCount - 1);
		}

		ScoreOnTrainingData = Scorer.Score(Rank(Samples));
		_logger.LogInformation($"Finished successfully. {Scorer.Name} on training data: {SimpleMath.Round(ScoreOnTrainingData, 4)}");

		if (ValidationSamples != null)
		{
			BestScoreOnValidationData = Scorer.Score(Rank(ValidationSamples));
			_logger.LogInformation($"{Scorer.Name} on validation data: {SimpleMath.Round(BestScoreOnValidationData, 4)}");
		}

		_logger.LogInformation("-- FEATURE IMPACTS");
		var ftrsSorted = MergeSorter.Sort(Impacts, false);
		foreach (var ftr in ftrsSorted)
		{
			_logger.LogInformation($"Feature {Features[ftr]} reduced error {Impacts[ftr]}");
		}
	}

	public override double Eval(DataPoint dataPoint) => _ensemble.Eval(dataPoint);

	public override string ToString() => _ensemble.ToString();

	public override string Model
	{
		get
		{
			var output = new StringBuilder();
			output.AppendLine($"## {Name}");
			output.AppendLine($"## No. of trees = {Parameters.nTrees}");
			output.AppendLine($"## No. of leaves = {Parameters.nTreeLeaves}");
			output.AppendLine($"## No. of threshold candidates = {Parameters.nThreshold}");
			output.AppendLine($"## Learning rate = {Parameters.learningRate}");
			output.AppendLine($"## Stop early = {Parameters.nRoundToStopEarly}");
			output.AppendLine();
			output.AppendLine(ToString());
			return output.ToString();
		}
	}

	public override void LoadFromString(string fullText)
	{
		var lineByLine = new ModelLineProducer();
		lineByLine.Parse(fullText, (model, endEns) => { });
		_ensemble = new Ensemble(lineByLine.Model.ToString());
		Features = _ensemble.Features;
	}

	public Ensemble GetEnsemble() => _ensemble;

	// Helper Methods
	protected virtual async Task ComputePseudoResponses()
	{
		Array.Fill(PseudoResponses, 0);
		Array.Fill(_weights, 0);
		if (Parameters.MaxDegreeOfParallelism == 1)
			ComputePseudoResponses(0, Samples.Count - 1, 0);
		else
		{
			var partition = ParallelExecutor.Partition(Samples.Count, Parameters.MaxDegreeOfParallelism);
			var current = 0;
			var parallelOptions = new ParallelOptions
			{
				MaxDegreeOfParallelism = Parameters.MaxDegreeOfParallelism,
				CancellationToken = default
			};

			var e = Enumerable.Range(0, partition.Length - 1)
				.Select(i =>
				{
					var start = partition[i];
					var end = partition[i + 1] - 1;
					var thisCurrent = current;
					if (i > 0 && i < partition.Length - 2)
					{
						for (var j = partition[i]; j <= partition[i + 1] - 1; j++)
						{
							thisCurrent += Samples[j].Count;
						}
					}

					return (start, end, thisCurrent);
				});

			await Parallel.ForEachAsync(e, parallelOptions, async (i, token) =>
			{
				await Task.Run(() =>
				{
					ComputePseudoResponses(i.start, i.end, i.thisCurrent);
				}, token);
			});
		}
	}

	protected void ComputePseudoResponses(int start, int end, int current)
	{
		var cutoff = Scorer.K;
		for (var i = start; i <= end; i++)
		{
			var orig = Samples[i];
			var idx = MergeSorter.Sort(ModelScores, current, current + orig.Count - 1, false);
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
							var rho = 1.0 / (1 + Math.Exp(ModelScores[mj] - ModelScores[mk]));
							var lambda = rho * deltaNDCG;
							PseudoResponses[mj] += lambda;
							PseudoResponses[mk] -= lambda;
							var delta = rho * (1.0 - rho) * deltaNDCG;
							_weights[mj] += delta;
							_weights[mk] += delta;
						}
					}
				}
			}
			current += orig.Count;
		}
	}

	protected virtual void UpdateTreeOutput(RegressionTree tree)
	{
		var leaves = tree.Leaves;
		foreach (var split in leaves)
		{
			var s1 = 0F;
			var s2 = 0F;
			var idx = split.GetSamples();
			foreach (var k in idx)
			{
				s1 += Convert.ToSingle(PseudoResponses[k]);
				s2 += Convert.ToSingle(_weights[k]);
			}
			if (s2 == 0)
				split.SetOutput(0);
			else
				split.SetOutput(s1 / s2);
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
			scores[i] = ModelScores[current + i];
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
			var idx = MergeSorter.Sort(_modelScoresOnValidation[i], false);
			score += Convert.ToSingle(Scorer.Score(new RankList(ValidationSamples[i], idx)));
		}
		return score;
	}

	protected void SortSamplesByFeature(int start, int end)
	{
		for (var i = start; i <= end; i++)
		{
			_sortedIdx[i] = SortSamplesByFeature(MARTSamples, Features[i]);
		}
	}

	private class SortWorker : RunnableTask
	{
		private readonly LambdaMART _ranker;
		private readonly int _start;
		private readonly int _end;

		public SortWorker(LambdaMART ranker, int start, int end)
		{
			_ranker = ranker;
			_start = start;
			_end = end;
		}

		public override Task RunAsync() => Task.Run(() => _ranker.SortSamplesByFeature(_start, _end));
	}
}
