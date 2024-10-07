using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;

namespace RankLib.Learning.NeuralNet;

public class LambdaRank : RankNet
{
	private readonly ILogger<LambdaRank>? _logger;
	protected float[][]? targetValue = null;

	public LambdaRank(ILogger<LambdaRank>? logger = null) : base(logger) =>
		_logger = logger ?? NullLogger<LambdaRank>.Instance;

	public LambdaRank(List<RankList> samples, int[] features, MetricScorer scorer, ILogger<LambdaRank>? logger = null)
		: base(samples, features, scorer, logger) =>
		_logger = logger ?? NullLogger<LambdaRank>.Instance;

	protected override int[][] BatchFeedForward(RankList rl)
	{
		var pairMap = new int[rl.Count][];
		targetValue = new float[rl.Count][];

		for (var i = 0; i < rl.Count; i++)
		{
			AddInput(rl[i]);
			Propagate(i);

			var count = 0;
			for (var j = 0; j < rl.Count; j++)
			{
				if (rl[i].Label > rl[j].Label || rl[i].Label < rl[j].Label)
				{
					count++;
				}
			}

			pairMap[i] = new int[count];
			targetValue[i] = new float[count];

			var k = 0;
			for (var j = 0; j < rl.Count; j++)
			{
				if (rl[i].Label > rl[j].Label || rl[i].Label < rl[j].Label)
				{
					pairMap[i][k] = j;
					targetValue[i][k] = rl[i].Label > rl[j].Label ? 1 : 0;
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
			var p = new PropParameter(i, pairMap, pairWeight, targetValue);

			// Back-propagate
			_outputLayer.ComputeDelta(p); // Starting at the output layer
			for (var j = _layers.Count - 2; j >= 1; j--)
			{
				_layers[j].UpdateDelta(p);
			}

			// Weight update
			_outputLayer.UpdateWeight(p);
			for (var j = _layers.Count - 2; j >= 1; j--)
			{
				_layers[j].UpdateWeight(p);
			}
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
				weight[i][j] = (float)Math.Abs(changes[i][pairMap[i][j]]) * sign;
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
						{
							_misorderedPairs++;
						}
					}
				}
			}
		}

		_error = 1.0 - ScoreOnTrainingData;

		if (_error > _lastError)
		{
			_straightLoss++;
		}
		else
		{
			_straightLoss = 0;
		}

		_lastError = _error;
	}

	public override Ranker CreateNew() => new LambdaRank(_logger);

	public override string Name => "LambdaRank";
}
