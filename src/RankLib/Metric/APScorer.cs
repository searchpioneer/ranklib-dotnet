using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Learning;
using RankLib.Utilities;

namespace RankLib.Metric;

public class APScorer : MetricScorer
{
	private readonly ILogger<APScorer> _logger;

	// This class computes MAP from the *WHOLE* ranked list. "K" will be completely ignored.
	// The reason is, if you want MAP@10, you really should be using NDCG@10 or ERR@10 instead.
	protected Dictionary<string, int>? relDocCount;

	public APScorer(ILogger<APScorer>? logger)
	{
		_logger = logger ?? NullLogger<APScorer>.Instance;

		// consider the whole list
		K = 0;
	}

	public override MetricScorer Copy() => new APScorer(_logger);

	public override void LoadExternalRelevanceJudgment(string queryRelevanceFile)
	{
		relDocCount = new Dictionary<string, int>();
		try
		{
			using (var reader = new StreamReader(queryRelevanceFile))
			{
				while (reader.ReadLine() is { } content)
				{
					content = content.Trim();
					if (content.Length == 0)
						continue;

					var parts = content.Split(' ');
					var qid = parts[0].Trim();
					var label = (int)Math.Round(double.Parse(parts[3].Trim()));

					if (label > 0)
					{
						relDocCount.TryAdd(qid, 0);
						relDocCount[qid] += 1;
					}
				}
			}

			_logger.LogInformation("Relevance judgment file loaded. [#q={RelDocCount}]", relDocCount.Count);
		}
		catch (IOException ex)
		{
			throw RankLibException.Create("Error in APScorer::LoadExternalRelevanceJudgment(): ", ex);
		}
	}

	/// <summary>
	/// Compute Average Precision (AP) of the list.
	/// AP of a list is the average of precision evaluated at ranks where a relevant document is observed.
	/// </summary>
	public override double Score(RankList rankList)
	{
		var ap = 0.0;
		var count = 0;

		for (var i = 0; i < rankList.Count; i++)
		{
			if (rankList[i].Label > 0.0) // relevant
			{
				count++;
				ap += ((double)count) / (i + 1);
			}
		}

		var rdCount = 0;

		if (relDocCount != null && relDocCount.TryGetValue(rankList.Id, out var relCount))
		{
			rdCount = relCount;
		}
		else
		{
			rdCount = count;
		}

		if (rdCount == 0)
			return 0.0;

		return ap / rdCount;
	}

	public override string Name => "MAP";

	public override double[][] SwapChange(RankList rankList)
	{
		// NOTE: Compute swap-change *IGNORING* K (consider the entire ranked list)
		var relCount = new int[rankList.Count];
		var labels = new int[rankList.Count];
		var count = 0;

		for (var i = 0; i < rankList.Count; i++)
		{
			if (rankList[i].Label > 0) // relevant
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

		var rdCount = 0; // total number of relevant documents

		if (relDocCount != null && relDocCount.TryGetValue(rankList.Id, out var relCountInList))
		{
			rdCount = relCountInList;
		}
		else
		{
			rdCount = count;
		}

		var changes = new double[rankList.Count][];
		for (var i = 0; i < rankList.Count; i++)
		{
			changes[i] = new double[rankList.Count];
			Array.Fill(changes[i], 0);
		}

		if (rdCount == 0 || count == 0)
		{
			return changes; // all "0"
		}

		for (var i = 0; i < rankList.Count - 1; i++)
		{
			for (var j = i + 1; j < rankList.Count; j++)
			{
				double change = 0;
				if (labels[i] != labels[j])
				{
					var diff = labels[j] - labels[i];
					change += ((double)((relCount[i] + diff) * labels[j] - relCount[i] * labels[i])) / (i + 1);

					for (var k = i + 1; k <= j - 1; k++)
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
