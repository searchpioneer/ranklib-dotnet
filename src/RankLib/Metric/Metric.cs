namespace RankLib.Metric;

/// <summary>
/// The type of metric
/// </summary>
public enum Metric
{
	/// <summary>
	/// Mean Average Precision
	/// </summary>
	MAP,
	/// <summary>
	/// Normalized Discounted Cumulative Gain
	/// </summary>
	NDCG,
	/// <summary>
	/// Discounted Cumulative Gain
	/// </summary>
	DCG,
	/// <summary>
	/// Precision at K
	/// </summary>
	Precision,
	/// <summary>
	/// Reciprocal Rank
	/// </summary>
	Reciprocal,
	/// <summary>
	/// Best at K
	/// </summary>
	Best,
	/// <summary>
	/// Expected Reciprocal Rank at K
	/// </summary>
	ERR
}
