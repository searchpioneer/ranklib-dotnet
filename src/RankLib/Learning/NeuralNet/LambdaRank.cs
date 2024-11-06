using Microsoft.Extensions.Logging;
using RankLib.Metric;

namespace RankLib.Learning.NeuralNet;

/// <summary>
/// LambdaRank is a learning-to-rank algorithm that optimizes a neural network using
/// gradient descent to directly maximize NDCG (Normalized Discounted Cumulative Gain).
/// It extends <see cref="RankNet"/> by incorporating a weighted lambda gradient that accounts for
/// the position-dependent NDCG gains between document pairs.
/// </summary>
public class LambdaRank : RankNet
{
	internal new const string RankerName = "LambdaRank";

	private float[][] _targetValue = [];

	public override string Name => RankerName;

	public LambdaRank(ILogger<LambdaRank>? logger = null) : base(logger)
	{
	}

	public LambdaRank(List<RankList> samples, int[] features, MetricScorer scorer, ILogger<LambdaRank>? logger = null)
		: base(samples, features, scorer, logger)
	{
	}

	protected override int[][] BatchFeedForward(RankList rankList)
	{
		var pairMap = new int[rankList.Count][];
		_targetValue = new float[rankList.Count][];

		for (var i = 0; i < rankList.Count; i++)
		{
			AddInput(rankList[i]);
			Propagate(i);

			var count = 0;
			for (var j = 0; j < rankList.Count; j++)
			{
				if (rankList[i].Label > rankList[j].Label || rankList[i].Label < rankList[j].Label)
					count++;
			}

			pairMap[i] = new int[count];
			_targetValue[i] = new float[count];

			var k = 0;
			for (var j = 0; j < rankList.Count; j++)
			{
				if (rankList[i].Label > rankList[j].Label || rankList[i].Label < rankList[j].Label)
				{
					pairMap[i][k] = j;
					_targetValue[i][k] = rankList[i].Label > rankList[j].Label ? 1 : 0;
					k++;
				}
			}
		}

		return pairMap;
	}

	protected override void BatchBackPropagate(int[][] pairMap, float[][] pairWeight)
	{
		for (var i = 0; i < pairMap.Length; i++)
		{
			var p = new PropParameter(i, pairMap, pairWeight, _targetValue);

			// Back-propagate
			OutputLayer.ComputeDelta(p); // Starting at the output layer
			for (var j = _layers.Count - 2; j >= 1; j--)
				_layers[j].UpdateDelta(p);

			// Weight update
			OutputLayer.UpdateWeight(p);
			for (var j = _layers.Count - 2; j >= 1; j--)
				_layers[j].UpdateWeight(p);
		}
	}

	protected override RankList InternalReorder(RankList rl) => Rank(rl);

	protected override float[][] ComputePairWeight(int[][] pairMap, RankList rl)
	{
		var changes = Scorer.SwapChange(rl);
		var weight = new float[pairMap.Length][];

		for (var i = 0; i < weight.Length; i++)
		{
			weight[i] = new float[pairMap[i].Length];
			for (var j = 0; j < pairMap[i].Length; j++)
			{
				var k = pairMap[i][j];
				var sign = rl[i].Label > rl[k].Label ? 1 : -1;
				weight[i][j] = (float)(Math.Abs(changes[i][pairMap[i][j]]) * sign);
			}
		}

		return weight;
	}

	protected override void EstimateLoss()
	{
		_misorderedPairs = 0;

		for (var j = 0; j < Samples.Count; j++)
		{
			var rl = Samples[j];

			for (var k = 0; k < rl.Count - 1; k++)
			{
				var o1 = Eval(rl[k]);
				for (var l = k + 1; l < rl.Count; l++)
				{
					if (rl[k].Label > rl[l].Label)
					{
						var o2 = Eval(rl[l]);
						if (o1 < o2)
							_misorderedPairs++;
					}
				}
			}
		}

		_error = 1.0 - TrainingDataScore;

		if (_error > _lastError)
			_straightLoss++;
		else
			_straightLoss = 0;

		_lastError = _error;
	}
}
