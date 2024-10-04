using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Utilities;
using Microsoft.Extensions.Logging;
using RankLib.Metric;
using RankLib.Parsing;

namespace RankLib.Learning.Tree;

public class LambdaMART : Ranker
{
    private static ILogger<LambdaMART> logger = NullLogger<LambdaMART>.Instance;

    // Parameters
    public static int nTrees = 1000; // number of trees
    public static float learningRate = 0.1F; // shrinkage
    public static int nThreshold = 256;
    public static int nRoundToStopEarly = 100;
    public static int nTreeLeaves = 10;
    public static int minLeafSupport = 1;

    // Local variables
    protected float[][] thresholds = null;
    protected Ensemble ensemble = null;
    protected double[] modelScores = null;
    protected double[][] modelScoresOnValidation = null;
    protected int bestModelOnValidation = int.MaxValue - 2;

    protected DataPoint[] martSamples = null;
    protected int[][] sortedIdx = null;
    protected FeatureHistogram hist = null;
    protected double[] pseudoResponses = null;
    protected double[] weights = null;
    protected internal double[] impacts = null;

    public LambdaMART()
    {
    }

    public LambdaMART(List<RankList> samples, int[] features, MetricScorer scorer)
        : base(samples, features, scorer)
    {
    }

    public override void Init()
    {
        logger.LogInformation("Initializing...");

        int dpCount = _samples.Sum(rl => rl.Size());
        int current = 0;
        martSamples = new DataPoint[dpCount];
        modelScores = new double[dpCount];
        pseudoResponses = new double[dpCount];
        impacts = new double[_features.Length];
        weights = new double[dpCount];

        for (int i = 0; i < _samples.Count; i++)
        {
            RankList rl = _samples[i];
            for (int j = 0; j < rl.Size(); j++)
            {
                martSamples[current + j] = rl.Get(j);
                modelScores[current + j] = 0.0F;
                pseudoResponses[current + j] = 0.0F;
                weights[current + j] = 0;
            }
            current += rl.Size();
        }

        // Sort samples by each feature
        sortedIdx = new int[_features.Length][];
        var threadPool = MyThreadPool.GetInstance();
        if (threadPool.Size() == 1)
        {
            SortSamplesByFeature(0, _features.Length - 1);
        }
        else
        {
            var partition = threadPool.Partition(_features.Length);
            for (int i = 0; i < partition.Length - 1; i++)
            {
                threadPool.Execute(new SortWorker(this, partition[i], partition[i + 1] - 1));
            }
            threadPool.Await();
        }

        thresholds = new float[_features.Length][];
        for (int f = 0; f < _features.Length; f++)
        {
            List<float> values = new List<float>();
            float fmax = float.MinValue;
            float fmin = float.MaxValue;
            for (int i = 0; i < martSamples.Length; i++)
            {
                int k = sortedIdx[f][i];
                float fv = martSamples[k].GetFeatureValue(_features[f]);
                values.Add(fv);
                if (fmax < fv) fmax = fv;
                if (fmin > fv) fmin = fv;

                int j = i + 1;
                while (j < martSamples.Length && martSamples[sortedIdx[f][j]].GetFeatureValue(_features[f]) <= fv)
                {
                    j++;
                }
                i = j - 1;
            }

            if (values.Count <= nThreshold || nThreshold == -1)
            {
                thresholds[f] = values.ToArray();
                thresholds[f] = thresholds[f].Concat(new float[] { float.MaxValue }).ToArray();
            }
            else
            {
                float step = Math.Abs(fmax - fmin) / nThreshold;
                thresholds[f] = new float[nThreshold + 1];
                thresholds[f][0] = fmin;
                for (int j = 1; j < nThreshold; j++)
                {
                    thresholds[f][j] = thresholds[f][j - 1] + step;
                }
                thresholds[f][nThreshold] = float.MaxValue;
            }
        }

        if (_validationSamples != null)
        {
            modelScoresOnValidation = new double[_validationSamples.Count][];
            for (int i = 0; i < _validationSamples.Count; i++)
            {
                modelScoresOnValidation[i] = new double[_validationSamples[i].Size()];
                Array.Fill(modelScoresOnValidation[i], 0);
            }
        }

        hist = new FeatureHistogram();
        hist.Construct(martSamples, pseudoResponses, sortedIdx, _features, thresholds, impacts);
        sortedIdx = null;
    }

    public override void Learn()
    {
        ensemble = new Ensemble();
        logger.LogInformation("Training starts...");

        if (_validationSamples != null)
        {
            PrintLogLn(new int[] { 7, 9, 9 }, new string[] { "#iter", _scorer.Name() + "-T", _scorer.Name() + "-V" });
        }
        else
        {
            PrintLogLn(new int[] { 7, 9 }, new string[] { "#iter", _scorer.Name() + "-T" });
        }

        for (int m = 0; m < nTrees; m++)
        {
            PrintLog(new int[] { 7 }, new string[] { (m + 1).ToString() });
            ComputePseudoResponses();
            hist.Update(pseudoResponses);
            var rt = new RegressionTree(nTreeLeaves, martSamples, pseudoResponses, hist, minLeafSupport);
            rt.Fit();
            ensemble.Add(rt, learningRate);
            UpdateTreeOutput(rt);

            List<Split> leaves = rt.Leaves();
            for (int i = 0; i < leaves.Count; i++)
            {
                Split s = leaves[i];
                int[] idx = s.GetSamples();
                for (int j = 0; j < idx.Length; j++)
                {
                    modelScores[idx[j]] += learningRate * s.GetOutput();
                }
            }
            rt.ClearSamples();

            _scoreOnTrainingData = ComputeModelScoreOnTraining();
            PrintLog(new int[] { 9 }, new string[] { SimpleMath.Round(_scoreOnTrainingData, 4).ToString() });

            if (_validationSamples != null)
            {
                for (int i = 0; i < modelScoresOnValidation.Length; i++)
                {
                    for (int j = 0; j < modelScoresOnValidation[i].Length; j++)
                    {
                        modelScoresOnValidation[i][j] += learningRate * rt.Eval(_validationSamples[i].Get(j));
                    }
                }
                double score = ComputeModelScoreOnValidation();
                PrintLog(new int[] { 9 }, new string[] { SimpleMath.Round(score, 4).ToString() });
                if (score > _bestScoreOnValidationData)
                {
                    _bestScoreOnValidationData = score;
                    bestModelOnValidation = ensemble.TreeCount() - 1;
                }
            }
            FlushLog();

            if (m - bestModelOnValidation > nRoundToStopEarly)
            {
                break;
            }
        }

        while (ensemble.TreeCount() > bestModelOnValidation + 1)
        {
            ensemble.Remove(ensemble.TreeCount() - 1);
        }

        _scoreOnTrainingData = _scorer.Score(Rank(_samples));
        logger.LogInformation($"Finished successfully. {_scorer.Name()} on training data: {SimpleMath.Round(_scoreOnTrainingData, 4)}");

        if (_validationSamples != null)
        {
            _bestScoreOnValidationData = _scorer.Score(Rank(_validationSamples));
            logger.LogInformation($"{_scorer.Name()} on validation data: {SimpleMath.Round(_bestScoreOnValidationData, 4)}");
        }

        logger.LogInformation("-- FEATURE IMPACTS");
        int[] ftrsSorted = MergeSorter.Sort(impacts, false);
        foreach (int ftr in ftrsSorted)
        {
            logger.LogInformation($"Feature {_features[ftr]} reduced error {impacts[ftr]}");
        }
    }

    public override double Eval(DataPoint dp)
    {
        return ensemble.Eval(dp);
    }

    public override Ranker CreateNew()
    {
        return new LambdaMART();
    }

    public override string ToString()
    {
        return ensemble.ToString();
    }

    public override string Model()
    {
        var output = new System.Text.StringBuilder();
        output.AppendLine($"## {Name()}");
        output.AppendLine($"## No. of trees = {nTrees}");
        output.AppendLine($"## No. of leaves = {nTreeLeaves}");
        output.AppendLine($"## No. of threshold candidates = {nThreshold}");
        output.AppendLine($"## Learning rate = {learningRate}");
        output.AppendLine($"## Stop early = {nRoundToStopEarly}");
        output.AppendLine();
        output.AppendLine(ToString());
        return output.ToString();
    }

    public override void LoadFromString(string fullText)
    {
        var lineByLine = new ModelLineProducer();
        lineByLine.Parse(fullText, (model, endEns) => { });
        ensemble = new Ensemble(lineByLine.GetModel().ToString());
        _features = ensemble.GetFeatures();
    }

    public override void PrintParameters()
    {
        logger.LogInformation($"No. of trees: {nTrees}");
        logger.LogInformation($"No. of leaves: {nTreeLeaves}");
        logger.LogInformation($"No. of threshold candidates: {nThreshold}");
        logger.LogInformation($"Min leaf support: {minLeafSupport}");
        logger.LogInformation($"Learning rate: {learningRate}");
        logger.LogInformation($"Stop early: {nRoundToStopEarly} rounds without performance gain on validation data");
    }

    public override string Name()
    {
        return "LambdaMART";
    }

    public Ensemble GetEnsemble()
    {
        return ensemble;
    }

    // Helper Methods
    protected virtual void ComputePseudoResponses()
    {
        Array.Fill(pseudoResponses, 0);
        Array.Fill(weights, 0);
        var p = MyThreadPool.GetInstance();
        if (p.Size() == 1)
        {
            ComputePseudoResponses(0, _samples.Count - 1, 0);
        }
        else
        {
            var workers = new List<LambdaComputationWorker>();
            var partition = p.Partition(_samples.Count);
            int current = 0;
            for (int i = 0; i < partition.Length - 1; i++)
            {
                var worker = new LambdaComputationWorker(this, partition[i], partition[i + 1] - 1, current);
                workers.Add(worker);
                p.Execute(worker);
                if (i < partition.Length - 2)
                {
                    for (int j = partition[i]; j <= partition[i + 1] - 1; j++)
                    {
                        current += _samples[j].Size();
                    }
                }
            }
            p.Await();
        }
    }

    protected void ComputePseudoResponses(int start, int end, int current)
    {
        int cutoff = _scorer.GetK();
        for (int i = start; i <= end; i++)
        {
            RankList orig = _samples[i];
            int[] idx = MergeSorter.Sort(modelScores, current, current + orig.Size() - 1, false);
            RankList rl = new RankList(orig, idx, current);
            double[][] changes = _scorer.SwapChange(rl);
            for (int j = 0; j < rl.Size(); j++)
            {
                DataPoint p1 = rl.Get(j);
                int mj = idx[j];
                for (int k = 0; k < rl.Size(); k++)
                {
                    if (j > cutoff && k > cutoff)
                    {
                        break;
                    }
                    DataPoint p2 = rl.Get(k);
                    int mk = idx[k];
                    if (p1.GetLabel() > p2.GetLabel())
                    {
                        double deltaNDCG = Math.Abs(changes[j][k]);
                        if (deltaNDCG > 0)
                        {
                            double rho = 1.0 / (1 + Math.Exp(modelScores[mj] - modelScores[mk]));
                            double lambda = rho * deltaNDCG;
                            pseudoResponses[mj] += lambda;
                            pseudoResponses[mk] -= lambda;
                            double delta = rho * (1.0 - rho) * deltaNDCG;
                            weights[mj] += delta;
                            weights[mk] += delta;
                        }
                    }
                }
            }
            current += orig.Size();
        }
    }

    protected virtual void UpdateTreeOutput(RegressionTree rt)
    {
        List<Split> leaves = rt.Leaves();
        foreach (var s in leaves)
        {
            float s1 = 0F;
            float s2 = 0F;
            int[] idx = s.GetSamples();
            foreach (var k in idx)
            {
                s1 += Convert.ToSingle(pseudoResponses[k]);
                s2 += Convert.ToSingle(weights[k]);
            }
            if (s2 == 0) s.SetOutput(0);
            else s.SetOutput(s1 / s2);
        }
    }

    protected int[] SortSamplesByFeature(DataPoint[] samples, int fid)
    {
        double[] score = new double[samples.Length];
        for (int i = 0; i < samples.Length; i++)
        {
            score[i] = samples[i].GetFeatureValue(fid);
        }
        int[] idx = MergeSorter.Sort(score, true);
        return idx;
    }

    protected RankList Rank(int rankListIndex, int current)
    {
        RankList orig = _samples[rankListIndex];
        double[] scores = new double[orig.Size()];
        for (int i = 0; i < scores.Length; i++)
        {
            scores[i] = modelScores[current + i];
        }
        int[] idx = MergeSorter.Sort(scores, false);
        return new RankList(orig, idx);
    }

    protected float ComputeModelScoreOnTraining()
    {
        return ComputeModelScoreOnTraining(0, _samples.Count - 1, 0) / _samples.Count;
    }

    protected float ComputeModelScoreOnTraining(int start, int end, int current)
    {
        float s = 0;
        int c = current;
        for (int i = start; i <= end; i++)
        {
            s += Convert.ToSingle(_scorer.Score(Rank(i, c)));
            c += _samples[i].Size();
        }
        return s;
    }

    protected float ComputeModelScoreOnValidation()
    {
        return ComputeModelScoreOnValidation(0, _validationSamples.Count - 1) / _validationSamples.Count;
    }

    protected float ComputeModelScoreOnValidation(int start, int end)
    {
        float score = 0;
        for (int i = start; i <= end; i++)
        {
            int[] idx = MergeSorter.Sort(modelScoresOnValidation[i], false);
            score += Convert.ToSingle(_scorer.Score(new RankList(_validationSamples[i], idx)));
        }
        return score;
    }

    protected void SortSamplesByFeature(int fStart, int fEnd)
    {
        for (int i = fStart; i <= fEnd; i++)
        {
            sortedIdx[i] = SortSamplesByFeature(martSamples, _features[i]);
        }
    }

    class SortWorker : RunnableTask
    {
        LambdaMART ranker;
        int start;
        int end;

        public SortWorker(LambdaMART ranker, int start, int end)
        {
            this.ranker = ranker;
            this.start = start;
            this.end = end;
        }

        public override void Run()
        {
            ranker.SortSamplesByFeature(start, end);
        }
    }

    class LambdaComputationWorker : RunnableTask
    {
        LambdaMART ranker;
        int rlStart;
        int rlEnd;
        int martStart;

        public LambdaComputationWorker(LambdaMART ranker, int rlStart, int rlEnd, int martStart)
        {
            this.ranker = ranker;
            this.rlStart = rlStart;
            this.rlEnd = rlEnd;
            this.martStart = martStart;
        }

        public override void Run()
        {
            ranker.ComputePseudoResponses(rlStart, rlEnd, martStart);
        }
    }
}
