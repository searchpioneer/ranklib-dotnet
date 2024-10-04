using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Utilities;
using KeyValuePair = RankLib.Utilities.KeyValuePair;

namespace RankLib.Learning;

public class LinearRegRank : Ranker
{
    private static readonly ILogger<LinearRegRank> logger = NullLogger<LinearRegRank>.Instance;
    public static double lambda = 1E-10; // L2-norm regularization parameter

    // Local variables
    protected double[] weight = null;

    public LinearRegRank() { }

    public LinearRegRank(List<RankList> samples, int[] features, MetricScorer scorer)
        : base(samples, features, scorer) { }

    public override void Init()
    {
        logger.LogInformation("Initializing...");
    }

    public override void Learn()
    {
        logger.LogInformation("Training starts...");
        logger.LogInformation("Learning the least square model...");

        // closed form solution: beta = ((xTx - lambda*I)^(-1)) * (xTy)
        int nVar = 0;
        foreach (var rl in _samples)
        {
            int c = rl.GetFeatureCount();
            if (c > nVar)
                nVar = c;
        }

        double[][] xTx = new double[nVar][];
        for (int i = 0; i < nVar; i++)
        {
            xTx[i] = new double[nVar];
            Array.Fill(xTx[i], 0.0);
        }

        double[] xTy = new double[nVar];
        Array.Fill(xTy, 0.0);

        for (int s = 0; s < _samples.Count; s++)
        {
            RankList rl = _samples[s];
            for (int i = 0; i < rl.Size(); i++)
            {
                DataPoint point = rl.Get(i);
                xTy[nVar - 1] += point.GetLabel();
                for (int j = 0; j < nVar - 1; j++)
                {
                    xTy[j] += point.GetFeatureValue(j + 1) * point.GetLabel();
                    for (int k = 0; k < nVar; k++)
                    {
                        double t = (k < nVar - 1) ? point.GetFeatureValue(k + 1) : 1f;
                        xTx[j][k] += point.GetFeatureValue(j + 1) * t;
                    }
                }

                for (int k = 0; k < nVar - 1; k++)
                    xTx[nVar - 1][k] += point.GetFeatureValue(k + 1);

                xTx[nVar - 1][nVar - 1] += 1f;
            }
        }

        if (lambda != 0.0)
        {
            for (int i = 0; i < xTx.Length; i++)
                xTx[i][i] += lambda;
        }

        weight = Solve(xTx, xTy);

        _scoreOnTrainingData = SimpleMath.Round(_scorer.Score(Rank(_samples)), 4);
        logger.LogInformation("Finished successfully.");
        logger.LogInformation($"{_scorer.Name()} on training data: {_scoreOnTrainingData}");

        if (_validationSamples != null)
        {
            _bestScoreOnValidationData = _scorer.Score(Rank(_validationSamples));
            logger.LogInformation($"{_scorer.Name()} on validation data: {SimpleMath.Round(_bestScoreOnValidationData, 4)}");
        }
    }

    public override double Eval(DataPoint p)
    {
        double score = weight[weight.Length - 1];
        for (int i = 0; i < _features.Length; i++)
            score += weight[i] * p.GetFeatureValue(_features[i]);
        return score;
    }

    public override Ranker CreateNew()
    {
        return new LinearRegRank();
    }

    public override string ToString()
    {
        StringBuilder output = new StringBuilder();
        output.Append("0:" + weight[0] + " ");
        for (int i = 0; i < _features.Length; i++)
        {
            output.Append(_features[i] + ":" + weight[i]);
            if (i != weight.Length - 1)
                output.Append(" ");
        }
        return output.ToString();
    }

    public override string Model()
    {
        StringBuilder output = new StringBuilder();
        output.Append("## " + Name() + "\n");
        output.Append("## Lambda = " + lambda + "\n");
        output.Append(ToString());
        return output.ToString();
    }

    public override void LoadFromString(string fullText)
    {
        try
        {
            using (StringReader reader = new StringReader(fullText))
            {
                string content;
                KeyValuePair kvp = null;
                while ((content = reader.ReadLine()) != null)
                {
                    content = content.Trim();
                    if (content.Length == 0 || content.StartsWith("##"))
                        continue;

                    kvp = new KeyValuePair(content);
                    break;
                }

                if (kvp == null) return;

                List<string> keys = kvp.Keys();
                List<string> values = kvp.Values();

                weight = new double[keys.Count];
                _features = new int[keys.Count - 1];

                int idx = 0;
                for (int i = 0; i < keys.Count; i++)
                {
                    int fid = int.Parse(keys[i]);
                    if (fid > 0)
                    {
                        _features[idx] = fid;
                        weight[idx] = double.Parse(values[i]);
                        idx++;
                    }
                    else
                    {
                        weight[weight.Length - 1] = double.Parse(values[i]);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            throw RankLibError.Create("Error in LinearRegRank::load(): ", ex);
        }
    }

    public override void PrintParameters()
    {
        logger.LogInformation("L2-norm regularization: lambda = " + lambda);
    }

    public override string Name()
    {
        return "Linear Regression";
    }

    protected double[] Solve(double[][] A, double[] B)
    {
        if (A.Length == 0 || B.Length == 0)
            throw RankLibError.Create("Error: some of the input arrays is empty.");
        if (A[0].Length == 0)
            throw RankLibError.Create("Error: some of the input arrays is empty.");
        if (A.Length != B.Length)
            throw RankLibError.Create("Error: Solving Ax=B: A and B have different dimensions.");

        double[][] a = new double[A.Length][];
        double[] b = new double[B.Length];
        Array.Copy(B, b, B.Length);
        for (int i = 0; i < a.Length; i++)
        {
            a[i] = new double[A[i].Length];
            if (i > 0 && a[i].Length != a[i - 1].Length)
                throw RankLibError.Create("Error: Solving Ax=B: A is NOT a square matrix.");
            Array.Copy(A[i], a[i], A[i].Length);
        }

        for (int j = 0; j < b.Length - 1; j++)
        {
            double pivot = a[j][j];
            for (int i = j + 1; i < b.Length; i++)
            {
                double multiplier = a[i][j] / pivot;
                for (int k = j + 1; k < b.Length; k++)
                    a[i][k] -= a[j][k] * multiplier;
                b[i] -= b[j] * multiplier;
            }
        }

        double[] x = new double[b.Length];
        int n = b.Length;
        x[n - 1] = b[n - 1] / a[n - 1][n - 1];
        for (int i = n - 2; i >= 0; i--)
        {
            double val = b[i];
            for (int j = i + 1; j < n; j++)
                val -= a[i][j] * x[j];
            x[i] = val / a[i][i];
        }

        return x;
    }
}
