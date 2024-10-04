using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Learning;
using RankLib.Utilities;

namespace RankLib.Metric;

public class APScorer : MetricScorer
{
    private static readonly ILogger<APScorer> logger = NullLogger<APScorer>.Instance;

    // This class computes MAP from the *WHOLE* ranked list. "K" will be completely ignored.
    // The reason is, if you want MAP@10, you really should be using NDCG@10 or ERR@10 instead.

    protected Dictionary<string, int>? relDocCount = null;

    public APScorer()
    {
        _k = 0; // consider the whole list
    }

    public override MetricScorer Copy()
    {
        return new APScorer();
    }

    public override void LoadExternalRelevanceJudgment(string qrelFile)
    {
        relDocCount = new Dictionary<string, int>();
        try
        {
            using (var reader = new StreamReader(qrelFile))
            {
                while (reader.ReadLine() is { } content)
                {
                    content = content.Trim();
                    if (content.Length == 0) continue;

                    var parts = content.Split(' ');
                    string qid = parts[0].Trim();
                    int label = (int)Math.Round(double.Parse(parts[3].Trim()));

                    if (label > 0)
                    {
                        relDocCount.TryAdd(qid, 0);
                        relDocCount[qid] += 1;
                    }
                }
            }

            logger.LogInformation($"Relevance judgment file loaded. [#q={relDocCount.Count}]");
        }
        catch (IOException ex)
        {
            throw RankLibError.Create("Error in APScorer::LoadExternalRelevanceJudgment(): ", ex);
        }
    }

    /// <summary>
    /// Compute Average Precision (AP) of the list. 
    /// AP of a list is the average of precision evaluated at ranks where a relevant document is observed.
    /// </summary>
    public override double Score(RankList rl)
    {
        double ap = 0.0;
        int count = 0;

        for (int i = 0; i < rl.Size(); i++)
        {
            if (rl.Get(i).GetLabel() > 0.0) // relevant
            {
                count++;
                ap += ((double)count) / (i + 1);
            }
        }

        int rdCount = 0;

        if (relDocCount != null && relDocCount.TryGetValue(rl.GetID(), out int relCount))
        {
            rdCount = relCount;
        }
        else
        {
            rdCount = count;
        }

        if (rdCount == 0) return 0.0;

        return ap / rdCount;
    }

    public override string Name()
    {
        return "MAP";
    }

    public override double[][] SwapChange(RankList rl)
    {
        // NOTE: Compute swap-change *IGNORING* K (consider the entire ranked list)
        int[] relCount = new int[rl.Size()];
        int[] labels = new int[rl.Size()];
        int count = 0;

        for (int i = 0; i < rl.Size(); i++)
        {
            if (rl.Get(i).GetLabel() > 0) // relevant
            {
                labels[i] = 1;
                count++;
            }
            else
            {
                labels[i] = 0;
            }
            relCount[i] = count;
        }

        int rdCount = 0; // total number of relevant documents

        if (relDocCount != null && relDocCount.TryGetValue(rl.GetID(), out int relCountInList))
        {
            rdCount = relCountInList;
        }
        else
        {
            rdCount = count;
        }

        double[][] changes = new double[rl.Size()][];
        for (int i = 0; i < rl.Size(); i++)
        {
            changes[i] = new double[rl.Size()];
            Array.Fill(changes[i], 0);
        }

        if (rdCount == 0 || count == 0)
        {
            return changes; // all "0"
        }

        for (int i = 0; i < rl.Size() - 1; i++)
        {
            for (int j = i + 1; j < rl.Size(); j++)
            {
                double change = 0;
                if (labels[i] != labels[j])
                {
                    int diff = labels[j] - labels[i];
                    change += ((double)((relCount[i] + diff) * labels[j] - relCount[i] * labels[i])) / (i + 1);

                    for (int k = i + 1; k <= j - 1; k++)
                    {
                        if (labels[k] > 0)
                        {
                            change += ((double)diff) / (k + 1);
                        }
                    }

                    change += ((double)(-relCount[j] * diff)) / (j + 1);
                }
                changes[j][i] = changes[i][j] = change / rdCount;
            }
        }

        return changes;
    }
}
