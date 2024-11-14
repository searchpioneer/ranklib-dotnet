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
/// Parameters for <see cref="LambdaMART"/>
/// </summary>
public class LambdaMARTParameters : IRankerParameters
{
    // Constants for default property values

    /// <summary>
    /// The default number of trees used in the model.
    /// </summary>
    public const int DefaultTreeCount = 1000;

    /// <summary>
    /// The default learning rate for the model.
    /// </summary>
    public const float DefaultLearningRate = 0.1f;

    /// <summary>
    /// The default number of threshold candidates for splits.
    /// </summary>
    public const int DefaultThreshold = 256;

    /// <summary>
    /// The default number of rounds to stop early without performance gain on validation data.
    /// </summary>
    public const int DefaultStopEarlyRoundCount = 100;

    /// <summary>
    /// The default number of leaves per tree.
    /// </summary>
    public const int DefaultTreeLeavesCount = 10;

    /// <summary>
    /// The default minimum leaf support, determining the minimum number of instances per leaf.
    /// </summary>
    public const int DefaultMinimumLeafSupport = 1;

    /// <summary>
    /// The default sampling rate for training data.
    /// </summary>
    public const float DefaultSamplingRate = 1f;

    // Properties

    /// <summary>
    /// Gets or sets the number of trees. Defaults to <see cref="DefaultTreeCount"/>.
    /// </summary>
    public int TreeCount { get; set; } = DefaultTreeCount;

    /// <summary>
    /// Gets or sets the learning rate. Defaults to <see cref="DefaultLearningRate"/>.
    /// </summary>
    public float LearningRate { get; set; } = DefaultLearningRate;

    /// <summary>
    /// Gets or sets the number of threshold candidates. Defaults to <see cref="DefaultThreshold"/>.
    /// </summary>
    public int Threshold { get; set; } = DefaultThreshold;

    /// <summary>
    /// Gets or sets the number of rounds to stop early without performance gain on validation data.
    /// Defaults to <see cref="DefaultStopEarlyRoundCount"/>.
    /// </summary>
    public int StopEarlyRoundCount { get; set; } = DefaultStopEarlyRoundCount;

    /// <summary>
    /// Gets or sets the number of leaves per tree. Defaults to <see cref="DefaultTreeLeavesCount"/>.
    /// </summary>
    public int TreeLeavesCount { get; set; } = DefaultTreeLeavesCount;

    /// <summary>
    /// Gets or sets the minimum leaf support, determining the minimum number of instances per leaf.
    /// Defaults to <see cref="DefaultMinimumLeafSupport"/>.
    /// </summary>
    public int MinimumLeafSupport { get; set; } = DefaultMinimumLeafSupport;

    /// <summary>
    /// Gets or sets the sampling rate for training data. Defaults to <see cref="DefaultSamplingRate"/>.
    /// </summary>
    public float SamplingRate { get; set; } = DefaultSamplingRate;

    /// <summary>
    /// Gets or sets the maximum number of concurrent tasks allowed when splitting up workloads
    /// that can be run on multiple threads. Defaults to the count of all available processors.
    /// </summary>
    public int MaxDegreeOfParallelism { get; set; } = Environment.ProcessorCount;

    /// <inheritdoc />
    public override string ToString()
    {
        var builder = new StringBuilder();
        builder.AppendLine($"No. of trees: {TreeCount}");
        builder.AppendLine($"No. of leaves: {TreeLeavesCount}");
        builder.AppendLine($"No. of threshold candidates: {Threshold}");
        builder.AppendLine($"Min leaf support: {MinimumLeafSupport}");
        builder.AppendLine($"Learning rate: {LearningRate}");
        builder.AppendLine($"Stop early: {StopEarlyRoundCount} rounds without performance gain on validation data");
        return builder.ToString();
    }
}


/// <summary>
/// LambdaMART is a ranking algorithm that combines LambdaRank and gradient-boosted decision trees (GBDT).
/// It optimizes for ranking metrics like NDCG (Normalized Discounted Cumulative Gain) by adjusting the model's weights
/// based on the relative order of items, rather than just classification or regression errors.
/// </summary>
/// <remarks>
/// <a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/LambdaMART_Final.pdf">
/// Q. Wu, C.J.C. Burges, K. Svore and J. Gao. Adapting Boosting for Information Retrieval Measures.
/// Journal of Information Retrieval, 2007.
/// </a>
/// </remarks>
public class LambdaMART : Ranker<LambdaMARTParameters>
{
	internal const string RankerName = "LambdaMART";

	private readonly ILogger<LambdaMART> _logger;

	private float[][] _thresholds = [];
	private Ensemble _ensemble;
	private double[][] _modelScoresOnValidation = [];
	private int _bestModelOnValidation = int.MaxValue - 2;
	private int[][] _sortedIdx = [];
	private FeatureHistogram _hist;
	private double[] _weights = [];

	/// <summary>
	/// The model scores calculated through learning
	/// </summary>
	protected double[] ModelScores = [];

	/// <summary>
	/// The samples for training
	/// </summary>
	protected DataPoint[] MARTSamples = [];

	/// <summary>
	/// The impacts of features calculated through learning
	/// </summary>
	protected internal double[] Impacts = [];

	/// <summary>
	/// The pseudo responses used in learning
	/// </summary>
	protected double[] PseudoResponses = [];

	/// <summary>
	/// Initializes a new instance of <see cref="LambdaMART"/>
	/// </summary>
	/// <param name="logger">logger to log messages</param>
	public LambdaMART(ILogger<LambdaMART>? logger = null) : this(new LambdaMARTParameters(), logger)
	{
	}

	/// <summary>
	/// Initializes a new instance of <see cref="LambdaMART"/>
	/// </summary>
	/// <param name="parameters">the parameters for training this instance</param>
	/// <param name="logger">logger to log messages</param>
	public LambdaMART(LambdaMARTParameters parameters, ILogger<LambdaMART>? logger = null)
	{
		Parameters = parameters;
		_logger = logger ?? NullLogger<LambdaMART>.Instance;
	}

	/// <summary>
	/// Initializes a new instance of <see cref="LambdaMART"/>
	/// </summary>
	/// <param name="parameters">the parameters for training this instance</param>
	/// <param name="samples">the training samples</param>
	/// <param name="features">the features</param>
	/// <param name="scorer">the scorer used to measure the effectiveness of the ranker</param>
	/// <param name="logger">logger to log messages</param>
	public LambdaMART(LambdaMARTParameters parameters, List<RankList> samples, int[] features, MetricScorer scorer, ILogger<LambdaMART>? logger = null)
		: base(samples, features, scorer)
	{
		Parameters = parameters;
		_logger = logger ?? NullLogger<LambdaMART>.Instance;
	}

	/// <summary>
	/// Initializes a new instance of <see cref="LambdaMART"/>
	/// </summary>
	/// <param name="samples">the training samples</param>
	/// <param name="features">the features</param>
	/// <param name="scorer">the scorer used to measure the effectiveness of the ranker</param>
	/// <param name="logger">logger to log messages</param>
	public LambdaMART(List<RankList> samples, int[] features, MetricScorer scorer, ILogger<LambdaMART>? logger = null) :
		this(new LambdaMARTParameters(), samples, features, scorer, logger)
	{
	}

	/// <inheritdoc />
	public override string Name => RankerName;

	/// <summary>
	/// Gets the ensemble
	/// </summary>
	public Ensemble Ensemble => _ensemble;

	/// <inheritdoc />
	public override async Task InitAsync()
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
				ModelScores[current + j] = 0.0f;
				PseudoResponses[current + j] = 0.0f;
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
			var partitions =
				Partitioner.PartitionEnumerable(Features.Length, Parameters.MaxDegreeOfParallelism);
			await Parallel.ForEachAsync(
				partitions,
				new ParallelOptions { MaxDegreeOfParallelism = Parameters.MaxDegreeOfParallelism },
				async (range, cancellationToken) =>
			{
				await Task.Run(() =>
					SortSamplesByFeature(range.Start.Value, range.End.Value), cancellationToken).ConfigureAwait(false);
			}).ConfigureAwait(false);
		}

		//Create a table of candidate thresholds (for each feature). Later on, we will select the best tree split from these candidates
		_thresholds = new float[Features.Length][];
		for (var f = 0; f < Features.Length; f++)
		{
			//For this feature, keep track of the list of unique values and the max/min
			var values = new List<float>(MARTSamples.Length);
			var fMax = float.NegativeInfinity;
			var fMin = float.MaxValue;
			for (var i = 0; i < MARTSamples.Length; i++)
			{
				//get samples sorted with respect to this feature
				var k = _sortedIdx[f][i];
				var fv = MARTSamples[k].GetFeatureValue(Features[f]);
				values.Add(fv);

				if (fMax < fv)
					fMax = fv;

				if (fMin > fv)
					fMin = fv;

				var j = i + 1;
				while (j < MARTSamples.Length)
				{
					if (MARTSamples[_sortedIdx[f][j]].GetFeatureValue(Features[f]) > fv)
						break;

					j++;
				}

				//[i, j] gives the range of samples with the same feature value
				i = j - 1;
			}

			if (values.Count <= Parameters.Threshold || Parameters.Threshold == -1)
			{
				_thresholds[f] = new float[values.Count + 1];
				for (var i = 0; i < values.Count; i++)
					_thresholds[f][i] = values[i];

				_thresholds[f][values.Count] = float.MaxValue;
			}
			else
			{
				var step = Math.Abs(fMax - fMin) / Parameters.Threshold;
				_thresholds[f] = new float[Parameters.Threshold + 1];
				_thresholds[f][0] = fMin;
				for (var j = 1; j < Parameters.Threshold; j++)
					_thresholds[f][j] = _thresholds[f][j - 1] + step;

				_thresholds[f][Parameters.Threshold] = float.MaxValue;
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
		await _hist.ConstructAsync(MARTSamples, PseudoResponses, _sortedIdx, Features, _thresholds, Impacts).ConfigureAwait(false);

		//we no longer need the sorted indexes of samples
		_sortedIdx = [];
	}

	/// <inheritdoc />
	public override async Task LearnAsync()
	{
		_ensemble = new Ensemble();
		_logger.LogInformation("Training starts...");

		if (ValidationSamples != null)
			_logger.PrintLog([7, 9, 9], ["#iter", Scorer.Name + "-T", Scorer.Name + "-V"]);
		else
			_logger.PrintLog([7, 9], ["#iter", Scorer.Name + "-T"]);

		var bufferedLogger = new BufferedLogger(_logger, new StringBuilder());
		for (var m = 0; m < Parameters.TreeCount; m++)
		{
			bufferedLogger.PrintLog([7], [(m + 1).ToString()]);

			//Compute lambdas (which act as the "pseudo responses")
			//Create training instances for MART:
			//  - Each document is a training sample
			//	- The lambda for this document serves as its training label
			await ComputePseudoResponsesAsync().ConfigureAwait(false);

			//update the histogram with these training labels (the feature histogram will be used to find the best tree split)
			await _hist.UpdateAsync(PseudoResponses).ConfigureAwait(false);

			//Fit a regression tree
			var rt = new RegressionTree(Parameters.TreeLeavesCount, MARTSamples, PseudoResponses, _hist, Parameters.MinimumLeafSupport);
			await rt.FitAsync().ConfigureAwait(false);

			//Add this tree to the ensemble (our model)
			_ensemble.Add(rt, Parameters.LearningRate);

			//update the outputs of the tree (with gamma computed using the Newton-Raphson method)
			UpdateTreeOutput(rt);

			//Update the model's outputs on all training samples
			var leaves = rt.Leaves;
			for (var i = 0; i < leaves.Count; i++)
			{
				var s = leaves[i];
				var idx = s.GetSamples();
				for (var j = 0; j < idx.Length; j++)
					ModelScores[idx[j]] += Parameters.LearningRate * s.Output;
			}

			//clear references to data that is no longer used
			rt.ClearSamples();

			//Evaluate the current model
			TrainingDataScore = ComputeModelScoreOnTraining();

			bufferedLogger.PrintLog([9], [SimpleMath.Round(TrainingDataScore, 4).ToString(CultureInfo.InvariantCulture)]);

			if (ValidationSamples != null)
			{
				for (var i = 0; i < _modelScoresOnValidation.Length; i++)
				{
					for (var j = 0; j < _modelScoresOnValidation[i].Length; j++)
						_modelScoresOnValidation[i][j] += Parameters.LearningRate * rt.Eval(ValidationSamples[i][j]);
				}

				double score = ComputeModelScoreOnValidation();
				bufferedLogger.PrintLog([9], [SimpleMath.Round(score, 4).ToString(CultureInfo.InvariantCulture)]);
				if (score > ValidationDataScore)
				{
					ValidationDataScore = score;
					_bestModelOnValidation = _ensemble.TreeCount - 1;
				}
			}

			bufferedLogger.FlushLog();

			if (m - _bestModelOnValidation > Parameters.StopEarlyRoundCount)
				break;
		}

		while (_ensemble.TreeCount > _bestModelOnValidation + 1)
			_ensemble.RemoveAt(_ensemble.TreeCount - 1);

		TrainingDataScore = Scorer.Score(Rank(Samples));
		_logger.LogInformation($"Finished successfully. {Scorer.Name} on training data: {SimpleMath.Round(TrainingDataScore, 4)}");

		if (ValidationSamples != null)
		{
			ValidationDataScore = Scorer.Score(Rank(ValidationSamples));
			_logger.LogInformation($"{Scorer.Name} on validation data: {SimpleMath.Round(ValidationDataScore, 4)}");
		}

		_logger.LogInformation("-- FEATURE IMPACTS");
		var ftrsSorted = MergeSorter.Sort(Impacts, false);
		for (var index = 0; index < ftrsSorted.Length; index++)
		{
			var ftr = ftrsSorted[index];
			_logger.LogInformation($"Feature {Features[ftr]} reduced error {Impacts[ftr]}");
		}
	}

	/// <inheritdoc />
	public override double Eval(DataPoint dataPoint) => _ensemble.Eval(dataPoint);

	/// <inheritdoc />
	public override string GetModel()
	{
		var output = new StringBuilder();
		output.AppendLine($"## {Name}");
		output.AppendLine($"## No. of trees = {Parameters.TreeCount}");
		output.AppendLine($"## No. of leaves = {Parameters.TreeLeavesCount}");
		output.AppendLine($"## No. of threshold candidates = {Parameters.Threshold}");
		output.AppendLine($"## Learning rate = {Parameters.LearningRate}");
		output.AppendLine($"## Stop early = {Parameters.StopEarlyRoundCount}");
		output.AppendLine();
		output.AppendLine(_ensemble.ToString());
		return output.ToString();
	}

	/// <inheritdoc />
	public override void LoadFromString(string model)
	{
		var lineByLine = new ModelLineProducer();
		lineByLine.Parse(model, (_, _) => { });
		_ensemble = Ensemble.Parse(lineByLine.Model.ToString());
		Features = _ensemble.Features;
	}

	/// <summary>
	/// Computes the pseudo responses for an iteration of learning.
	/// </summary>
	protected virtual async Task ComputePseudoResponsesAsync()
	{
		Array.Fill(PseudoResponses, 0);
		Array.Fill(_weights, 0);
		if (Parameters.MaxDegreeOfParallelism == 1)
			ComputePseudoResponses(0, Samples.Count - 1, 0);
		else
		{
			var partition = Partitioner.Partition(Samples.Count, Parameters.MaxDegreeOfParallelism);
			var current = 0;
			var tuples = Enumerable.Range(0, partition.Length - 1)
				.Select(i =>
				{
					var start = partition[i];
					var end = partition[i + 1] - 1;
					var values = (start, end, current);
					if (i < partition.Length - 2)
					{
						for (var j = partition[i]; j <= partition[i + 1] - 1; j++)
							current += Samples[j].Count;
					}

					return values;
				})
				.ToList();

			var parallelOptions = new ParallelOptions
			{
				MaxDegreeOfParallelism = Parameters.MaxDegreeOfParallelism,
				CancellationToken = default
			};

			await Parallel.ForEachAsync(tuples, parallelOptions, async (values, cancellationToken) =>
				await Task.Run(() => ComputePseudoResponses(values.start, values.end, values.current), cancellationToken).ConfigureAwait(false))
				.ConfigureAwait(false);
		}
	}

	// compute pseudo responses based on the current model's error (another name for pseudo response -- 'force' or 'gradient')
	// "pseudo responses" is the error currently in the model (that we'll attempt to model with features)
	// How do we get that error?
	//  Let's say we have two training samples (aka docs) for a query. Docs k and j, where k is more relevant than j
	//  (ie label of k is 4, label for j is 0)
	// Then we want two values to help compute a pseudo response to help build a model for the remaining error
	//  1. rho -- a weight for how wrong the previous model is. Higher rho is, the more the prev model is wrong
	//			  at dealing with docs k and j by just predicting scores that don't make sense
	//  2. deltaNDCG -- what swapping k and j means for the NDCG for this query.
	//					even though the variable is called 'NDCG' it really uses whatever relevance metric you
	//					specify (MAP, precision, ERR,... whatever)
	//
	// We update pseudoResponse[k] += rho * deltaNDCG (higher gradient/force when
	//													(a) -- rho high: previous models are more wrong
	//													(b) -- deltaNDCG high: these two docs being swapped
	//														   would be really bad for this particular query
	//     aka pseudoResponse[k] += current error * importance

	// We also update down j (which remember should be left relevant than k) by subtracting out the same val:
	//         pseudoResponse[j] -= current error * importance
	private void ComputePseudoResponses(int start, int end, int current)
	{
		var cutoff = Scorer.K;
		// compute the lambda for each document (a.k.a "pseudo response")
		for (var i = start; i <= end; i++)
		{
			var orig = Samples[i];
			// sort based on current model's relevance scores
			var idx = MergeSorter.Sort(ModelScores, current, current + orig.Count - 1, false);
			var rl = new RankList(orig, idx, current);

			// a table of possible rearrangements of rl
			var changes = Scorer.SwapChange(rl);
			//NOTE: j, k are indices in the sorted (by modelScore) list, not the original
			// ==> need to map back with idx[j] and idx[k]
			for (var j = 0; j < rl.Count; j++)
			{
				var pointJ = rl[j];
				var mj = idx[j];
				for (var k = 0; k < rl.Count; k++)
				{
					//swapping these pair won't result in any change in target measures since they're below the cut-off point
					if (j > cutoff && k > cutoff)
						break;

					var pointK = rl[k];
					var mk = idx[k];
					if (pointJ.Label > pointK.Label)
					{
						// ReSharper disable once InconsistentNaming
						var deltaNDCG = Math.Abs(changes[j][k]);
						if (deltaNDCG > 0)
						{
							// rho weighs the delta ndcg by the current model score
							// in this way, this is acting as a gradient
							// rho mj's score
							// if the model scores are close (say 100 for j, k for 99)
							//   rho is smaller
							// if model scores are far
							var rho = 1.0 / (1 + Math.Exp(ModelScores[mj] - ModelScores[mk]));
							var lambda = rho * deltaNDCG;

							// response of DataPoint j in original list
							// which is better than k in original list
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

	/// <summary>
	/// Updates the outputs of the tree
	/// </summary>
	/// <param name="tree">The tree to update</param>
	protected virtual void UpdateTreeOutput(RegressionTree tree)
	{
		var leaves = tree.Leaves;
		for (var i = 0; i < leaves.Count; i++)
		{
			var s1 = 0f;
			var s2 = 0f;
			var split = leaves[i];
			var idx = split.GetSamples();
			for (var j = 0; j < idx.Length; j++)
			{
				var k = idx[j];
				s1 = (float)(s1 + PseudoResponses[k]);
				s2 = (float)(s2 + _weights[k]);
			}

			if (s2 == 0)
				split.Output = (float)0;
			else
				split.Output = s1 / s2;
		}
	}

	private int[] SortSamplesByFeature(DataPoint[] samples, int fid)
	{
		var score = new double[samples.Length];
		for (var i = 0; i < samples.Length; i++)
			score[i] = samples[i].GetFeatureValue(fid);

		var idx = MergeSorter.Sort(score, true);
		return idx;
	}

	private RankList Rank(int rankListIndex, int current)
	{
		var orig = Samples[rankListIndex];
		var scores = new double[orig.Count];
		for (var i = 0; i < scores.Length; i++)
			scores[i] = ModelScores[current + i];

		var idx = MergeSorter.Sort(scores, false);
		return new RankList(orig, idx);
	}

	private float ComputeModelScoreOnTraining() => ComputeModelScoreOnTraining(0, Samples.Count - 1, 0) / Samples.Count;

	private float ComputeModelScoreOnTraining(int start, int end, int current)
	{
		float s = 0;
		var c = current;
		for (var i = start; i <= end; i++)
		{
			s = (float)(s + Scorer.Score(Rank(i, c)));
			c += Samples[i].Count;
		}
		return s;
	}

	private float ComputeModelScoreOnValidation() =>
		ComputeModelScoreOnValidation(0, ValidationSamples!.Count - 1) / ValidationSamples.Count;

	private float ComputeModelScoreOnValidation(int start, int end)
	{
		float score = 0;
		for (var i = start; i <= end; i++)
		{
			var idx = MergeSorter.Sort(_modelScoresOnValidation[i], false);
			score = (float)(score + Scorer.Score(new RankList(ValidationSamples![i], idx)));
		}
		return score;
	}

	private void SortSamplesByFeature(int start, int end)
	{
		for (var i = start; i <= end; i++)
			_sortedIdx[i] = SortSamplesByFeature(MARTSamples, Features[i]);
	}
}
