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
/// A ranker that can score <see cref="DataPoint"/> and rank data points in a <see cref="RankList"/>.
/// A ranker can be trained by a <see cref="RankerTrainer"/>, or a trained ranker model can be loaded
/// from file
/// </summary>
/// <remarks>
/// Use <see cref="EvaluatorFactory"/> to create an <see cref="Evaluator"/> for training and evaluation.
/// </remarks>
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
	/// <param name="cancellationToken">Token used to cancel the operation</param>
	/// <returns>a new instance of <see cref="Task"/> that can be awaited.</returns>
	Task InitAsync(CancellationToken cancellationToken = default);

	/// <summary>
	/// Trains the ranker to learn from the training samples.
	/// </summary>
	/// <param name="cancellationToken">Token used to cancel the operation</param>
	/// <returns>a new instance of <see cref="Task"/> that can be awaited.</returns>
	Task LearnAsync(CancellationToken cancellationToken = default);

	/// <summary>
	/// Evaluates a data point.
	/// </summary>
	/// <param name="dataPoint">The data point.</param>
	/// <returns>The score for the data point</returns>
	double Eval(DataPoint dataPoint);

	/// <summary>
	/// Gets the model for the ranker.
	/// </summary>
	/// <remarks>
	/// When a ranker is loaded from a string with <see cref="LoadFromString"/>, this returns the ranker model.
	/// For a ranker trained using an <see cref="Evaluator"/>, this returns the model <b>after training</b>.
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
	/// <param name="cancellationToken">Token used to cancel the operation</param>
	/// <returns>a new instance of <see cref="Task"/> that can be awaited.</returns>
	Task SaveAsync(string modelFile, CancellationToken cancellationToken = default);

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
/// A generic ranker that can score <see cref="DataPoint"/> and rank data points in a <see cref="RankList"/>.
/// A ranker can be trained by a <see cref="RankerTrainer"/>, or a trained ranker model can be loaded
/// from file
/// </summary>
/// <typeparam name="TRankerParameters">The type of ranker parameters</typeparam>
/// <remarks>
/// Use <see cref="EvaluatorFactory"/> to create an <see cref="Evaluator"/> for training and evaluation.
/// </remarks>
public interface IRanker<TRankerParameters> : IRanker
	where TRankerParameters : IRankerParameters
{
	/// <summary>
	/// Gets or sets the parameters for the ranker.
	/// The ranker uses parameters for training
	/// </summary>
	new TRankerParameters Parameters { get; set; }
}
