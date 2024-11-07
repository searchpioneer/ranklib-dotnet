using System.Text;
using RankLib.Eval;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning;

/// <summary>
/// Base class for a generic ranker with typed ranker parameters.
/// </summary>
/// <typeparam name="TRankerParameters">The type of ranker parameters</typeparam>
public abstract class Ranker<TRankerParameters> : Ranker, IRanker<TRankerParameters>
	where TRankerParameters : IRankerParameters, new()
{
	/// <summary>
	/// Initializes a new instance of <see cref="Ranker{TRankerParameters}"/>
	/// </summary>
	protected Ranker()
	{
	}

	/// <summary>
	/// Initializes a new instance of <see cref="Ranker{TRankerParameters}"/>
	/// </summary>
	/// <param name="samples">The training samples</param>
	/// <param name="features">The features</param>
	/// <param name="scorer">The scorer to use for training</param>
	protected Ranker(List<RankList> samples, int[] features, MetricScorer scorer)
	: base(samples, features, scorer)
	{
	}

	/// <summary>
	/// Gets or sets the ranker parameters used to train the ranker
	/// </summary>
	public TRankerParameters Parameters { get; set; } = new();

	/// <inheritdoc />
	IRankerParameters IRanker.Parameters
	{
		get => Parameters;
		set => Parameters = (TRankerParameters)value;
	}
}

/// <summary>
/// Base class for a ranker
/// </summary>
public abstract class Ranker : IRanker
{
	private MetricScorer? _scorer;

	protected double TrainingDataScore = 0.0;
	protected double ValidationDataScore = 0.0;

	/// <inheritdoc />
	public List<RankList> Samples { get; set; } = [];

	/// <inheritdoc />
	public List<RankList>? ValidationSamples { get; set; }

	/// <inheritdoc />
	public int[] Features { get; set; } = [];

	/// <summary>
	/// Gets or sets the metric scorer
	/// </summary>
	/// <remarks>
	/// If no scorer is assigned, a new instance of <see cref="APScorer"/> is instantiated on first get
	/// </remarks>
	public MetricScorer Scorer
	{
		get => _scorer ?? new APScorer();
		set => _scorer = value;
	}

	/// <inheritdoc />
	IRankerParameters IRanker.Parameters { get; set; } = default!;

	/// <summary>
	/// Initializes a new instance of <see cref="Ranker"/>
	/// </summary>
	protected Ranker()
	{
	}

	/// <summary>
	/// Initializes a new instance of <see cref="Ranker"/>
	/// </summary>
	/// <param name="samples">The training samples</param>
	/// <param name="features">The features</param>
	/// <param name="scorer">The scorer to use for training</param>
	protected Ranker(List<RankList> samples, int[] features, MetricScorer scorer)
	{
		Samples = samples;
		Features = features;
		Scorer = scorer;
	}

	/// <inheritdoc />
	public double GetTrainingDataScore() => TrainingDataScore;

	/// <inheritdoc />
	public double GetValidationDataScore() => ValidationDataScore;

	/// <inheritdoc />
	public virtual RankList Rank(RankList rankList)
	{
		var scores = new double[rankList.Count];
		for (var i = 0; i < rankList.Count; i++)
			scores[i] = Eval(rankList[i]);

		var idx = MergeSorter.Sort(scores, false);
		return new RankList(rankList, idx);
	}

	/// <inheritdoc />
	public List<RankList> Rank(List<RankList> rankLists)
	{
		var rankedRankLists = new List<RankList>(rankLists.Count);
		for (var i = 0; i < rankLists.Count; i++)
			rankedRankLists.Add(Rank(rankLists[i]));

		return rankedRankLists;
	}

	/// <inheritdoc />
	public async Task SaveAsync(string modelFile)
	{
		var directory = Path.GetDirectoryName(Path.GetFullPath(modelFile));
		Directory.CreateDirectory(directory!);
		await File.WriteAllTextAsync(modelFile, GetModel(), Encoding.ASCII);
	}

	/// <inheritdoc />
	public abstract Task InitAsync();

	/// <inheritdoc />
	public abstract Task LearnAsync();

	/// <inheritdoc />
	public abstract double Eval(DataPoint dataPoint);

	/// <inheritdoc />
	public abstract string GetModel();

	/// <inheritdoc />
	public abstract void LoadFromString(string model);

	/// <inheritdoc />
	public abstract string Name { get; }
}
