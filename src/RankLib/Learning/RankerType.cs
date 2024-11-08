using RankLib.Learning.Boosting;
using RankLib.Learning.NeuralNet;
using RankLib.Learning.Tree;

namespace RankLib.Learning;

/// <summary>
/// The ranker type
/// </summary>
public enum RankerType
{
	/// <inheritdoc cref="RankLib.Learning.Tree.MART"/>
	MART = 0,
	/// <inheritdoc cref="RankLib.Learning.NeuralNet.RankNet"/>
	RankNet = 1,
	/// <inheritdoc cref="RankLib.Learning.Boosting.RankBoost"/>
	RankBoost = 2,
	/// <inheritdoc cref="RankLib.Learning.Boosting.AdaRank"/>
	AdaRank = 3,
	/// <inheritdoc cref="RankLib.Learning.CoordinateAscent"/>
	CoordinateAscent = 4,
	/// <inheritdoc cref="RankLib.Learning.NeuralNet.LambdaRank"/>
	LambdaRank = 5,
	/// <inheritdoc cref="RankLib.Learning.Tree.LambdaMART"/>
	LambdaMART = 6,
	/// <inheritdoc cref="RankLib.Learning.NeuralNet.ListNet"/>
	ListNet = 7,
	/// <inheritdoc cref="RankLib.Learning.Tree.RandomForests"/>
	RandomForests = 8,
	/// <inheritdoc cref="RankLib.Learning.LinearRegression"/>
	LinearRegression = 9,
}

/// <summary>
/// Extension methods for <see cref="RankerType"/>
/// </summary>
public static class RankerTypeExtensions
{
	/// <summary>
	/// Gets the ranker <see cref="Type"/> from the <see cref="RankerType"/>.
	/// </summary>
	/// <param name="rankerType">The ranker type</param>
	/// <returns>The ranker type</returns>
	/// <exception cref="ArgumentOutOfRangeException">
	/// <paramref name="rankerType"/> is outside of the range of valid values.
	/// </exception>
	public static Type GetRankerType(this RankerType rankerType) =>
		rankerType switch
		{
			RankerType.MART => typeof(MART),
			RankerType.RankBoost => typeof(RankBoost),
			RankerType.RankNet => typeof(RankNet),
			RankerType.AdaRank => typeof(AdaRank),
			RankerType.CoordinateAscent => typeof(CoordinateAscent),
			RankerType.LambdaRank => typeof(LambdaRank),
			RankerType.LambdaMART => typeof(LambdaMART),
			RankerType.ListNet => typeof(ListNet),
			RankerType.RandomForests => typeof(RandomForests),
			RankerType.LinearRegression => typeof(LinearRegression),
			_ => throw new ArgumentOutOfRangeException(nameof(rankerType), rankerType, "Unknown ranker type")
		};
}
