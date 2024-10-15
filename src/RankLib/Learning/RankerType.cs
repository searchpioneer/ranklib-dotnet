using RankLib.Learning.Boosting;
using RankLib.Learning.NeuralNet;
using RankLib.Learning.Tree;

namespace RankLib.Learning;

public enum RankerType
{
	MART = 0,
	RANKBOOST,
	RANKNET,
	ADARANK,
	COOR_ASCENT,
	LAMBDARANK,
	LAMBDAMART,
	LISTNET,
	RANDOM_FOREST,
	LINEAR_REGRESSION
}

public static class RankerTypeExtensions
{
	public static Type GetRankerType(this RankerType rankerType) =>
		rankerType switch
		{
			RankerType.MART => typeof(MART),
			RankerType.RANKBOOST => typeof(RankBoost),
			RankerType.RANKNET => typeof(RankNet),
			RankerType.ADARANK => typeof(AdaRank),
			RankerType.COOR_ASCENT => typeof(CoorAscent),
			RankerType.LAMBDARANK => typeof(LambdaRank),
			RankerType.LAMBDAMART => typeof(LambdaMART),
			RankerType.LISTNET => typeof(ListNet),
			RankerType.RANDOM_FOREST => typeof(RFRanker),
			RankerType.LINEAR_REGRESSION => typeof(LinearRegRank),
			_ => throw new ArgumentOutOfRangeException(nameof(rankerType), rankerType, null)
		};
}
