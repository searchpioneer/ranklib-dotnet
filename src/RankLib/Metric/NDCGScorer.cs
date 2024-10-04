using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Learning;
using RankLib.Utilities;

namespace RankLib.Metric;

public class NDCGScorer : DCGScorer
{
    private static readonly ILogger<NDCGScorer> logger = NullLogger<NDCGScorer>.Instance;
    protected Dictionary<string, double> idealGains = new();

    public NDCGScorer()
    {
    }

    public NDCGScorer(int k) : base(k)
    {
    }

    public override MetricScorer Copy()
    {
        return new NDCGScorer();
    }

    public override void LoadExternalRelevanceJudgment(string qrelFile)
    {
        // Queries with external relevance judgment will have their cached ideal gain value overridden
        try
        {
            using (var reader = new StreamReader(qrelFile))
            {
                string content;
                string lastQID = string.Empty;
                var rel = new List<int>();
                int nQueries = 0;

                while ((content = reader.ReadLine()) != null)
                {
                    content = content.Trim();
                    if (string.IsNullOrEmpty(content))
                    {
                        continue;
                    }

                    var parts = content.Split(' ');
                    string qid = parts[0].Trim();
                    int label = (int)Math.Round(double.Parse(parts[3].Trim()));

                    if (!string.IsNullOrEmpty(lastQID) && !lastQID.Equals(qid, StringComparison.Ordinal))
                    {
                        int size = (rel.Count > _k) ? _k : rel.Count;
                        int[] r = rel.ToArray();
                        double ideal = GetIdealDCG(r, size);
                        idealGains[lastQID] = ideal;
                        rel.Clear();
                        nQueries++;
                    }

                    lastQID = qid;
                    rel.Add(label);
                }

                if (rel.Count > 0)
                {
                    int size = (rel.Count > _k) ? _k : rel.Count;
                    int[] r = rel.ToArray();
                    double ideal = GetIdealDCG(r, size);
                    idealGains[lastQID] = ideal;
                    rel.Clear();
                    nQueries++;
                }

                logger.LogInformation($"Relevance judgment file loaded. [#q={nQueries}]");
            }
        }
        catch (IOException ex)
        {
            throw RankLibError.Create($"Error in NDCGScorer::loadExternalRelevanceJudgment(): {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Compute NDCG at k. NDCG(k) = DCG(k) / DCG_{perfect}(k).
    /// </summary>
    public override double Score(RankList rl)
    {
        if (rl.Size() == 0)
        {
            return 0;
        }

        int size = _k;
        if (_k > rl.Size() || _k <= 0)
        {
            size = rl.Size();
        }

        int[] rel = GetRelevanceLabels(rl);

        double ideal = 0;
        if (idealGains.TryGetValue(rl.GetID(), out var cachedIdeal))
        {
            ideal = cachedIdeal;
        }
        else
        {
            ideal = GetIdealDCG(rel, size);
            idealGains[rl.GetID()] = ideal;
        }

        if (ideal <= 0.0)
        {
            return 0.0;
        }

        return GetDCG(rel, size) / ideal;
    }

    public override double[][] SwapChange(RankList rl)
    {
        int size = (rl.Size() > _k) ? _k : rl.Size();
        int[] rel = GetRelevanceLabels(rl);

        double ideal = 0;
        if (idealGains.TryGetValue(rl.GetID(), out var cachedIdeal))
        {
            ideal = cachedIdeal;
        }
        else
        {
            ideal = GetIdealDCG(rel, size);
        }

        var changes = new double[rl.Size()][];
        for (int i = 0; i < rl.Size(); i++)
        {
            changes[i] = new double[rl.Size()];
            Array.Fill(changes[i], 0);
        }

        for (int i = 0; i < size; i++)
        {
            for (int j = i + 1; j < rl.Size(); j++)
            {
                if (ideal > 0)
                {
                    changes[j][i] = changes[i][j] = (Discount(i) - Discount(j)) * (Gain(rel[i]) - Gain(rel[j])) / ideal;
                }
            }
        }

        return changes;
    }

    public override string Name()
    {
        return "NDCG@" + _k;
    }

    private double GetIdealDCG(int[] rel, int topK)
    {
        int[] idx = Sorter.Sort(rel, false);
        double dcg = 0;

        for (int i = 0; i < topK; i++)
        {
            dcg += Gain(rel[idx[i]]) * Discount(i);
        }

        return dcg;
    }
}