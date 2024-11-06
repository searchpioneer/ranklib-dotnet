using RankLib.Learning.Boosting;
using RankLib.Learning.NeuralNet;
using RankLib.Learning.Tree;

namespace RankLib.Learning;

/// <summary>
/// The ranker type
/// </summary>
public enum RankerType
{
	MART = 0,
	RankNet = 1,
	RankBoost = 2,
	AdaRank = 3,
	CoordinateAscent = 4,
	LambdaRank = 5,
	LambdaMART = 6,
	ListNet = 7,
	RandomForests = 8,
	LinearRegression = 9,
}

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
