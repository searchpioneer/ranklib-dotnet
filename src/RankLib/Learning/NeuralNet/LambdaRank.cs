using RankLib.Metric;

namespace RankLib.Learning.NeuralNet;

public class LambdaRank : RankNet
{
    protected float[][]? targetValue = null;

    public LambdaRank() { }

    public LambdaRank(List<RankList> samples, int[] features, MetricScorer scorer)
        : base(samples, features, scorer)
    { }

    protected override int[][] BatchFeedForward(RankList rl)
    {
        int[][] pairMap = new int[rl.Size()][];
        targetValue = new float[rl.Size()][];
        
        for (int i = 0; i < rl.Size(); i++)
        {
            AddInput(rl.Get(i));
            Propagate(i);

            int count = 0;
            for (int j = 0; j < rl.Size(); j++)
            {
                if (rl.Get(i).GetLabel() > rl.Get(j).GetLabel() || rl.Get(i).GetLabel() < rl.Get(j).GetLabel())
                {
                    count++;
                }
            }

            pairMap[i] = new int[count];
            targetValue[i] = new float[count];

            int k = 0;
            for (int j = 0; j < rl.Size(); j++)
            {
                if (rl.Get(i).GetLabel() > rl.Get(j).GetLabel() || rl.Get(i).GetLabel() < rl.Get(j).GetLabel())
                {
                    pairMap[i][k] = j;
                    targetValue[i][k] = rl.Get(i).GetLabel() > rl.Get(j).GetLabel() ? 1 : 0;
                    k++;
                }
            }
        }

        return pairMap;
    }

    protected override void BatchBackPropagate(int[][] pairMap, float[][] pairWeight)
    {
        for (int i = 0; i < pairMap.Length; i++)
        {
            var p = new PropParameter(i, pairMap, pairWeight, targetValue);

            // Back-propagate
            _outputLayer.ComputeDelta(p); // Starting at the output layer
            for (int j = _layers.Count - 2; j >= 1; j--)
            {
                _layers[j].UpdateDelta(p);
            }

            // Weight update
            _outputLayer.UpdateWeight(p);
            for (int j = _layers.Count - 2; j >= 1; j--)
            {
                _layers[j].UpdateWeight(p);
            }
        }
    }

    protected override RankList InternalReorder(RankList rl)
    {
        return Rank(rl);
    }

    protected override float[][] ComputePairWeight(int[][] pairMap, RankList rl)
    {
        double[][] changes = _scorer.SwapChange(rl);
        float[][] weight = new float[pairMap.Length][];

        for (int i = 0; i < weight.Length; i++)
        {
            weight[i] = new float[pairMap[i].Length];
            for (int j = 0; j < pairMap[i].Length; j++)
            {
                int sign = rl.Get(i).GetLabel() > rl.Get(pairMap[i][j]).GetLabel() ? 1 : -1;
                weight[i][j] = (float)Math.Abs(changes[i][pairMap[i][j]]) * sign;
            }
        }

        return weight;
    }

    protected override void EstimateLoss()
    {
        _misorderedPairs = 0;

        for (int j = 0; j < _samples.Count; j++)
        {
            RankList rl = _samples[j];

            for (int k = 0; k < rl.Size() - 1; k++)
            {
                double o1 = Eval(rl.Get(k));
                for (int l = k + 1; l < rl.Size(); l++)
                {
                    if (rl.Get(k).GetLabel() > rl.Get(l).GetLabel())
                    {
                        double o2 = Eval(rl.Get(l));
                        if (o1 < o2)
                        {
                            _misorderedPairs++;
                        }
                    }
                }
            }
        }

        _error = 1.0 - _scoreOnTrainingData;

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

    public override Ranker CreateNew()
    {
        return new LambdaRank();
    }

    public override string Name()
    {
        return "LambdaRank";
    }
}
