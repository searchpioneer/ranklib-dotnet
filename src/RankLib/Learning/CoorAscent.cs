using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Utilities;
using System.Text;
using Microsoft.Extensions.Logging;
using RankLib.Metric;
using KeyValuePair = RankLib.Utilities.KeyValuePair;

namespace RankLib.Learning;

public class CoorAscent : Ranker
{
    private static readonly ILogger<CoorAscent> logger = NullLogger<CoorAscent>.Instance;

    // Parameters
    public static int nRestart = 5;
    public static int nMaxIteration = 25;
    public static double stepBase = 0.05;
    public static double stepScale = 2.0;
    public static double tolerance = 0.001;
    public static bool regularized = false;
    public static double slack = 0.001;

    // Local variables
    public double[] weight = null;

    protected int currentFeature = -1; // Used only during learning
    protected double weightChange = -1.0; // Used only during learning

    public CoorAscent() { }

    public CoorAscent(List<RankList> samples, int[] features, MetricScorer scorer)
        : base(samples, features, scorer) { }

    public override void Init()
    {
        logger.LogInformation("Initializing...");
        weight = new double[_features.Length];
        for (int i = 0; i < weight.Length; i++)
        {
            weight[i] = 1.0 / _features.Length;
        }
    }

    public override void Learn()
    {
        var regVector = new double[weight.Length];
        Array.Copy(weight, regVector, weight.Length); // Uniform weight distribution

        double[]? bestModel = null;
        double bestModelScore = 0.0;
        int[] sign = new int[] { 1, -1, 0 };

        logger.LogInformation("Training starts...");

        for (int r = 0; r < nRestart; r++)
        {
            logger.LogInformation($"[+] Random restart #{r + 1}/{nRestart}...");
            int consecutiveFails = 0;

            for (int i = 0; i < weight.Length; i++)
            {
                weight[i] = 1.0f / _features.Length;
            }

            currentFeature = -1;
            double startScore = _scorer.Score(Rank(_samples));

            double bestScore = startScore;
            var bestWeight = new double[weight.Length];
            Array.Copy(weight, bestWeight, weight.Length);

            while ((weight.Length > 1 && consecutiveFails < weight.Length - 1) || (weight.Length == 1 && consecutiveFails == 0))
            {
                logger.LogInformation("Shuffling features' order...");
                logger.LogInformation("Optimizing weight vector... ");
                PrintLogLn(new[] { 7, 8, 7 }, new[] { "Feature", "weight", _scorer.Name() });

                int[] shuffledFeatures = GetShuffledFeatures();

                for (int i = 0; i < shuffledFeatures.Length; i++)
                {
                    currentFeature = shuffledFeatures[i];
                    double origWeight = weight[shuffledFeatures[i]];
                    double bestTotalStep = 0;
                    bool succeeds = false;

                    for (int s = 0; s < sign.Length; s++)
                    {
                        int dir = sign[s];
                        double step = 0.001 * dir;
                        if (origWeight != 0.0 && Math.Abs(step) > 0.5 * Math.Abs(origWeight))
                        {
                            step = stepBase * Math.Abs(origWeight);
                        }

                        double totalStep = step;
                        int numIter = dir == 0 ? 1 : nMaxIteration;

                        for (int j = 0; j < numIter; j++)
                        {
                            double newWeight = origWeight + totalStep;
                            weightChange = step;
                            weight[shuffledFeatures[i]] = newWeight;

                            double score = _scorer.Score(Rank(_samples));
                            if (regularized)
                            {
                                double penalty = slack * GetDistance(weight, regVector);
                                score -= penalty;
                            }

                            if (score > bestScore)
                            {
                                bestScore = score;
                                bestTotalStep = totalStep;
                                succeeds = true;
                                string bw = weight[shuffledFeatures[i]] > 0 ? "+" : "";
                                PrintLogLn(new[] { 7, 8, 7 }, new[] { _features[shuffledFeatures[i]].ToString(), $"{bw}{Math.Round(weight[shuffledFeatures[i]], 4)}", Math.Round(bestScore, 4).ToString() });
                            }

                            if (j < nMaxIteration - 1)
                            {
                                step *= stepScale;
                                totalStep += step;
                            }
                        }

                        if (succeeds) break;

                        if (s < sign.Length - 1)
                        {
                            weightChange = -totalStep;
                            UpdateCached();
                            weight[shuffledFeatures[i]] = origWeight;
                        }
                    }

                    if (succeeds)
                    {
                        weightChange = bestTotalStep - weight[shuffledFeatures[i]];
                        UpdateCached();
                        weight[shuffledFeatures[i]] = origWeight + bestTotalStep;
                        consecutiveFails = 0;

                        double sum = Normalize(weight);
                        ScaleCached(sum);
                        Array.Copy(weight, bestWeight, weight.Length);
                    }
                    else
                    {
                        consecutiveFails++;
                        weightChange = -weight[shuffledFeatures[i]];
                        UpdateCached();
                        weight[shuffledFeatures[i]] = origWeight;
                    }
                }

                if (bestScore - startScore < tolerance) break;
            }

            if (_validationSamples != null)
            {
                currentFeature = -1;
                bestScore = _scorer.Score(Rank(_validationSamples));
            }

            if (bestModel == null || bestScore > bestModelScore)
            {
                bestModelScore = bestScore;
                bestModel = new double[bestWeight.Length];
                Array.Copy(bestWeight, bestModel, bestWeight.Length);
            }
        }

        Array.Copy(bestModel, weight, bestModel.Length);
        currentFeature = -1;
        _scoreOnTrainingData = Math.Round(_scorer.Score(Rank(_samples)), 4);
        logger.LogInformation("Finished successfully.");
        logger.LogInformation($"{_scorer.Name()} on training data: {_scoreOnTrainingData}");

        if (_validationSamples != null)
        {
            _bestScoreOnValidationData = _scorer.Score(Rank(_validationSamples));
            logger.LogInformation($"{_scorer.Name()} on validation data: {Math.Round(_bestScoreOnValidationData, 4)}");
        }
    }

    public override RankList Rank(RankList rl)
    {
        double[] score = new double[rl.Size()];
        if (currentFeature == -1)
        {
            for (int i = 0; i < rl.Size(); i++)
            {
                for (int j = 0; j < _features.Length; j++)
                {
                    score[i] += weight[j] * rl.Get(i).GetFeatureValue(_features[j]);
                }
                rl.Get(i).SetCached(score[i]);
            }
        }
        else
        {
            for (int i = 0; i < rl.Size(); i++)
            {
                score[i] = rl.Get(i).GetCached() + weightChange * rl.Get(i).GetFeatureValue(_features[currentFeature]);
                rl.Get(i).SetCached(score[i]);
            }
        }

        int[] idx = MergeSorter.Sort(score, false);
        return new RankList(rl, idx);
    }

    public override double Eval(DataPoint p)
    {
        double score = 0.0;
        for (int i = 0; i < _features.Length; i++)
        {
            score += weight[i] * p.GetFeatureValue(_features[i]);
        }
        return score;
    }

    public override Ranker CreateNew()
    {
        return new CoorAscent();
    }

    public override string ToString()
    {
        StringBuilder output = new StringBuilder();
        for (int i = 0; i < weight.Length; i++)
        {
            output.Append($"{_features[i]}:{weight[i]}{(i == weight.Length - 1 ? "" : " ")}");
        }
        return output.ToString();
    }

    public override string Model()
    {
        StringBuilder output = new StringBuilder();
        output.AppendLine($"## {Name()}");
        output.AppendLine($"## Restart = {nRestart}");
        output.AppendLine($"## MaxIteration = {nMaxIteration}");
        output.AppendLine($"## StepBase = {stepBase}");
        output.AppendLine($"## StepScale = {stepScale}");
        output.AppendLine($"## Tolerance = {tolerance}");
        output.AppendLine($"## Regularized = {regularized}");
        output.AppendLine($"## Slack = {slack}");
        output.AppendLine(ToString());
        return output.ToString();
    }

    public override void LoadFromString(string fullText)
    {
        using var reader = new StringReader(fullText);
        while (reader.ReadLine() is { } line)
        {
            if (line.StartsWith("##")) continue;
            var kvp = new KeyValuePair(line);
            var keys = kvp.Keys();
            var values = kvp.Values();
            weight = new double[keys.Count];
            _features = new int[keys.Count];
            for (int i = 0; i < keys.Count; i++)
            {
                _features[i] = int.Parse(keys[i]);
                weight[i] = double.Parse(values[i]);
            }
            break;
        }
    }

    public override void PrintParameters()
    {
        logger.LogInformation($"No. of random restarts: {nRestart}");
        logger.LogInformation($"No. of iterations to search in each direction: {nMaxIteration}");
        logger.LogInformation($"Tolerance: {tolerance}");
        if (regularized)
        {
            logger.LogInformation($"Reg. param: {slack}");
        }
        else
        {
            logger.LogInformation("Regularization: No");
        }
    }

    public override string Name()
    {
        return "Coordinate Ascent";
    }

    // Private helper methods
    private void UpdateCached()
    {
        for (int j = 0; j < _samples.Count; j++)
        {
            RankList rl = _samples[j];
            for (int i = 0; i < rl.Size(); i++)
            {
                double score = rl.Get(i).GetCached() + weightChange * rl.Get(i).GetFeatureValue(_features[currentFeature]);
                rl.Get(i).SetCached(score);
            }
        }
    }

    private void ScaleCached(double sum)
    {
        for (int j = 0; j < _samples.Count; j++)
        {
            RankList rl = _samples[j];
            for (int i = 0; i < rl.Size(); i++)
            {
                rl.Get(i).SetCached(rl.Get(i).GetCached() / sum);
            }
        }
    }

    private int[] GetShuffledFeatures()
    {
        var indices = Enumerable.Range(0, _features.Length).ToList();
        indices = indices.OrderBy(x => Guid.NewGuid()).ToList(); // Shuffle
        return indices.ToArray();
    }

    private double GetDistance(double[] w1, double[] w2)
    {
        double s1 = w1.Sum(Math.Abs);
        double s2 = w2.Sum(Math.Abs);
        double dist = 0.0;
        for (int i = 0; i < w1.Length; i++)
        {
            double t = w1[i] / s1 - w2[i] / s2;
            dist += t * t;
        }
        return Math.Sqrt(dist);
    }

    private double Normalize(double[] weights)
    {
        double sum = weights.Sum(Math.Abs);
        if (sum > 0)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] /= sum;
            }
        }
        else
        {
            sum = 1;
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = 1.0 / weights.Length;
            }
        }
        return sum;
    }

    public void CopyModel(CoorAscent ranker) {
        weight = new double[_features.Length];
        if (ranker.weight.Length != weight.Length) {
            throw RankLibError.Create("These two models use different feature set!!");
        }
        Copy(ranker.weight, weight);
        logger.LogInformation("Model loaded.");
    }

    public double Distance(CoorAscent ca) {
        return GetDistance(weight, ca.weight);
    }
}
