using System.Text;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning;

/// <summary>
/// Generic base class for a ranker with typed ranker parameters.
/// </summary>
/// <typeparam name="TRankerParameters">The type of ranker parameters</typeparam>
public abstract class Ranker<TRankerParameters> : Ranker, IRanker<TRankerParameters>
	where TRankerParameters : IRankerParameters, new()
{
	protected Ranker()
	{
	}

	protected Ranker(List<RankList> samples, int[] features, MetricScorer scorer)
	: base(samples, features, scorer)
	{
	}

	/// <summary>
	/// Gets or sets the ranker parameters
	/// </summary>
	public TRankerParameters Parameters { get; set; } = new();

	IRankerParameters IRanker.Parameters
	{
		get => Parameters;
		set => Parameters = (TRankerParameters)value;
	}
}

/// <summary>
/// Base class for a ranker with ranker parameters
/// </summary>
public abstract class Ranker : IRanker
{
	private MetricScorer? _scorer;

	protected double TrainingDataScore = 0.0;
	protected double ValidationDataScore = 0.0;

	/// <summary>
	/// Gets or sets the training samples.
	/// </summary>
	public List<RankList> Samples { get; set; } = []; // training samples

	/// <summary>
	/// Gets or sets the validation samples.
	/// </summary>
	public List<RankList>? ValidationSamples { get; set; }

	/// <summary>
	/// Gets or sets the features.
	/// </summary>
	public int[] Features { get; set; } = [];

	/// <summary>
	/// Gets or sets the scorer
	/// </summary>
	/// <remarks>
	/// If no scorer is assigned, a new instance of <see cref="APScorer"/> is instantiated on first get
	/// </remarks>
	public MetricScorer Scorer
	{
		get => _scorer ?? new APScorer();
		set => _scorer = value;
	}

	IRankerParameters IRanker.Parameters { get; set; } = default!;

	protected Ranker()
	{
	}

	protected Ranker(List<RankList> samples, int[] features, MetricScorer scorer)
	{
		Samples = samples;
		Features = features;
		Scorer = scorer;
	}

	/// <summary>
	/// Gets the score from the training data.
	/// </summary>
	/// <returns>The training data score.</returns>
	public double GetTrainingDataScore() => TrainingDataScore;

	/// <summary>
	/// Gets the score from the validation data.
	/// </summary>
	/// <returns>The validation data score.</returns>
	public double GetValidationDataScore() => ValidationDataScore;

	public virtual RankList Rank(RankList rankList)
	{
		var scores = new double[rankList.Count];
		for (var i = 0; i < rankList.Count; i++)
			scores[i] = Eval(rankList[i]);

		var idx = MergeSorter.Sort(scores, false);
		return new RankList(rankList, idx);
	}

	public List<RankList> Rank(List<RankList> rankLists)
	{
		var rankedRankLists = new List<RankList>(rankLists.Count);
		for (var i = 0; i < rankLists.Count; i++)
			rankedRankLists.Add(Rank(rankLists[i]));

		return rankedRankLists;
	}

	/// <summary>
	/// Saves the model to file.
	/// </summary>
	/// <param name="modelFile">The file path to save the model to.</param>
	public async Task SaveAsync(string modelFile)
	{
		var directory = Path.GetDirectoryName(Path.GetFullPath(modelFile));
		Directory.CreateDirectory(directory!);
		await File.WriteAllTextAsync(modelFile, Model, Encoding.ASCII);
	}

	/// <summary>
	/// Initializes the ranker for training.
	/// </summary>
	/// <returns></returns>
	public abstract Task InitAsync();

	/// <summary>
	/// Trains the ranker to learn from the training samples.
	/// </summary>
	/// <returns></returns>
	public abstract Task LearnAsync();

	/// <summary>
	/// Evaluates a datapoint.
	/// </summary>
	/// <param name="dataPoint">The data point.</param>
	/// <returns>The score for the data point</returns>
	public abstract double Eval(DataPoint dataPoint);

	/// <inheritdoc />
	public abstract override string ToString();

	/// <summary>
	/// Gets the model for the ranker.
	/// </summary>
	public abstract string Model { get; }

	/// <summary>
	/// Loads a ranker from a model.
	/// </summary>
	/// <param name="model">The model for the ranker.</param>
	public abstract void LoadFromString(string model);

	/// <summary>
	/// Gets the name of the ranker.
	/// </summary>
	public abstract string Name { get; }
}
