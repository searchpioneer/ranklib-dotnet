using RankLib.Eval;
using RankLib.Metric;

namespace RankLib.Learning;

/// <summary>
/// Parameters for a <see cref="IRanker"/>
/// </summary>
/// <remarks>
/// Implementations should override <see cref="object.ToString"/> to allow parameters to be logged.
/// </remarks>
public interface IRankerParameters
{
}

/// <summary>
/// A ranker
/// </summary>
public interface IRanker
{
	/// <summary>
	/// Gets or sets the training samples.
	/// </summary>
	public List<RankList> Samples { get; set; }

	/// <summary>
	/// Gets or sets the validation samples.
	/// </summary>
	public List<RankList>? ValidationSamples { get; set; }

	/// <summary>
	/// Gets or sets the features.
	/// </summary>
	public int[] Features { get; set; }

	/// <summary>
	/// Gets or sets the metric scorer
	/// </summary>
	public MetricScorer Scorer { get; set; }

	/// <summary>
	/// Initializes the ranker for training.
	/// </summary>
	/// <returns>a new instance of <see cref="Task"/> that can be awaited.</returns>
	Task InitAsync();

	/// <summary>
	/// Trains the ranker to learn from the training samples.
	/// </summary>
	/// <returns>a new instance of <see cref="Task"/> that can be awaited.</returns>
	Task LearnAsync();

	/// <summary>
	/// Evaluates a datapoint.
	/// </summary>
	/// <param name="dataPoint">The data point.</param>
	/// <returns>The score for the data point</returns>
	double Eval(DataPoint dataPoint);

	/// <summary>
	/// Gets the model for the ranker.
	/// </summary>
	/// <remarks>
	/// When a ranker is loaded from a string with <see cref="LoadFromString"/>, contains the ranker model.
	/// For a ranker trained using an <see cref="Evaluator"/>, contains the model <b>after</b> training.
	/// </remarks>
	string GetModel();

	/// <summary>
	/// Loads a ranker from a model.
	/// </summary>
	/// <param name="model">The model for the ranker.</param>
	void LoadFromString(string model);

	/// <summary>
	/// Gets the name of the ranker.
	/// </summary>
	string Name { get; }

	/// <summary>
	/// Gets or sets the ranker parameters used to train the ranker
	/// </summary>
	public IRankerParameters Parameters { get; set; }

	/// <summary>
	/// Ranks a rank list by evaluating and scoring data points using the ranker's model.
	/// </summary>
	/// <param name="rankList">The rank list to rank.</param>
	/// <returns>A new instance of <see cref="RankList"/> ranked using the ranker's model.</returns>
	RankList Rank(RankList rankList);

	/// <summary>
	/// Ranks a list of rank lists by evaluating and scoring data points using the ranker's model.
	/// </summary>
	/// <param name="rankLists">The list of rank list to rank.</param>
	/// <returns>
	/// A list of new instances of <see cref="RankList"/> ranked using the ranker's model.
	/// The order of rank lists matches the input order.
	/// </returns>
	List<RankList> Rank(List<RankList> rankLists);

	/// <summary>
	/// Saves the model to file.
	/// </summary>
	/// <param name="modelFile">The file path to save the model to.</param>
	/// <returns>a new instance of <see cref="Task"/> that can be awaited.</returns>
	Task SaveAsync(string modelFile);

	/// <summary>
	/// Gets the score from evaluation on the training data.
	/// </summary>
	/// <returns>The training data score</returns>
	double GetTrainingDataScore();

	/// <summary>
	/// Gets the score from evaluation on the validation data.
	/// </summary>
	/// <returns>The validation data score.</returns>
	double GetValidationDataScore();
}

/// <summary>
/// A generic ranker
/// </summary>
/// <typeparam name="TParameters">The type of rank parameters</typeparam>
public interface IRanker<TParameters> : IRanker
	where TParameters : IRankerParameters
{
	/// <summary>
	/// Gets or sets the parameters for the ranker.
	/// The ranker uses parameters for training
	/// </summary>
	public new TParameters Parameters { get; set; }
}
