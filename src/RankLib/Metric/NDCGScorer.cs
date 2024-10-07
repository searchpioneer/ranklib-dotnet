using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Learning;
using RankLib.Utilities;

namespace RankLib.Metric;

public class NDCGScorer : DCGScorer
{
	private readonly ILogger<NDCGScorer> _logger;
	protected Dictionary<string, double> idealGains = new();

	public NDCGScorer(ILogger<NDCGScorer>? logger = null) =>
		_logger = logger ?? NullLogger<NDCGScorer>.Instance;

	public NDCGScorer(int k, ILogger<NDCGScorer>? logger = null) : base(k) =>
		_logger = logger ?? NullLogger<NDCGScorer>.Instance;

	public override MetricScorer Copy() => new NDCGScorer(_logger);

	public override void LoadExternalRelevanceJudgment(string qrelFile)
	{
		// Queries with external relevance judgment will have their cached ideal gain value overridden
		try
		{
			using var reader = new StreamReader(qrelFile);
			var lastQID = string.Empty;
			var rel = new List<int>();
			var nQueries = 0;

			while (reader.ReadLine() is { } content)
			{
				content = content.Trim();
				if (string.IsNullOrEmpty(content))
				{
					continue;
				}

				var parts = content.Split(' ');
				var qid = parts[0].Trim();
				var label = (int)Math.Round(double.Parse(parts[3].Trim()));

				if (!string.IsNullOrEmpty(lastQID) && !lastQID.Equals(qid, StringComparison.Ordinal))
				{
					var size = (rel.Count > K) ? K : rel.Count;
					var r = rel.ToArray();
					var ideal = GetIdealDCG(r, size);
					idealGains[lastQID] = ideal;
					rel.Clear();
					nQueries++;
				}

				lastQID = qid;
				rel.Add(label);
			}

			if (rel.Count > 0)
			{
				var size = (rel.Count > K) ? K : rel.Count;
				var r = rel.ToArray();
				var ideal = GetIdealDCG(r, size);
				idealGains[lastQID] = ideal;
				rel.Clear();
				nQueries++;
			}

			_logger.LogInformation("Relevance judgment file loaded. [#q={NumberOfQueries}]", nQueries);
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
		if (rl.Count == 0)
		{
			return 0;
		}

		var size = K;
		if (K > rl.Count || K <= 0)
		{
			size = rl.Count;
		}

		var rel = GetRelevanceLabels(rl);

		double ideal = 0;
		if (idealGains.TryGetValue(rl.Id, out var cachedIdeal))
		{
			ideal = cachedIdeal;
		}
		else
		{
			ideal = GetIdealDCG(rel, size);
			idealGains[rl.Id] = ideal;
		}

		if (ideal <= 0.0)
		{
			return 0.0;
		}

		return GetDCG(rel, size) / ideal;
	}

	public override double[][] SwapChange(RankList rl)
	{
		var size = (rl.Count > K) ? K : rl.Count;
		var rel = GetRelevanceLabels(rl);

		double ideal = 0;
		if (idealGains.TryGetValue(rl.Id, out var cachedIdeal))
		{
			ideal = cachedIdeal;
		}
		else
		{
			ideal = GetIdealDCG(rel, size);
		}

		var changes = new double[rl.Count][];
		for (var i = 0; i < rl.Count; i++)
		{
			changes[i] = new double[rl.Count];
			Array.Fill(changes[i], 0);
		}

		for (var i = 0; i < size; i++)
		{
			for (var j = i + 1; j < rl.Count; j++)
			{
				if (ideal > 0)
				{
					changes[j][i] = changes[i][j] = (Discount(i) - Discount(j)) * (Gain(rel[i]) - Gain(rel[j])) / ideal;
				}
			}
		}

		return changes;
	}

	public override string Name => "NDCG@" + K;

	private double GetIdealDCG(int[] rel, int topK)
	{
		var idx = Sorter.Sort(rel, false);
		double dcg = 0;

		for (var i = 0; i < topK; i++)
		{
			dcg += Gain(rel[idx[i]]) * Discount(i);
		}

		return dcg;
	}
}
