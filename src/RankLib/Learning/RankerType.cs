using RankLib.Learning.Boosting;
using RankLib.Learning.NeuralNet;
using RankLib.Learning.Tree;

namespace RankLib.Learning;

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
	public static Type GetRankerType(this RankerType rankerType) =>
		rankerType switch
		{
			RankerType.MART => typeof(MART),
			RankerType.RankBoost => typeof(RankBoost),
			RankerType.RankNet => typeof(RankNet),
			RankerType.AdaRank => typeof(AdaRank),
			RankerType.CoordinateAscent => typeof(CoorAscent),
			RankerType.LambdaRank => typeof(LambdaRank),
			RankerType.LambdaMART => typeof(LambdaMART),
			RankerType.ListNet => typeof(ListNet),
			RankerType.RandomForests => typeof(RFRanker),
			RankerType.LinearRegression => typeof(LinearRegRank),
			_ => throw new ArgumentOutOfRangeException(nameof(rankerType), rankerType, "Unknown ranker type")
		};
}
