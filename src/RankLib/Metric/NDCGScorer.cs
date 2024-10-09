using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Learning;
using RankLib.Utilities;

namespace RankLib.Metric;

/// <summary>
/// Normalized Discounted Cumulative Gain scorer
/// </summary>
/// <remarks>
/// https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG
/// </remarks>
public class NDCGScorer : DCGScorer
{
	private readonly ILogger<NDCGScorer> _logger;
	private readonly Dictionary<string, double> _idealGains = new();

	public NDCGScorer(ILogger<NDCGScorer>? logger = null) =>
		_logger = logger ?? NullLogger<NDCGScorer>.Instance;

	public NDCGScorer(int k, ILogger<NDCGScorer>? logger = null) : base(k) =>
		_logger = logger ?? NullLogger<NDCGScorer>.Instance;

	public override void LoadExternalRelevanceJudgment(string queryRelevanceFile)
	{
		// Queries with external relevance judgment will have their cached ideal gain value overridden
		try
		{
			using var reader = new StreamReader(queryRelevanceFile);
			var lastQid = string.Empty;
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

				if (!string.IsNullOrEmpty(lastQid) && !lastQid.Equals(qid, StringComparison.Ordinal))
				{
					var size = (rel.Count > K) ? K : rel.Count;
					var r = rel.ToArray();
					var ideal = GetIdealDCG(r, size);
					_idealGains[lastQid] = ideal;
					rel.Clear();
					nQueries++;
				}

				lastQid = qid;
				rel.Add(label);
			}

			if (rel.Count > 0)
			{
				var size = rel.Count > K
					? K
					: rel.Count;

				var r = rel.ToArray();
				var ideal = GetIdealDCG(r, size);
				_idealGains[lastQid] = ideal;
				rel.Clear();
				nQueries++;
			}

			_logger.LogInformation("Relevance judgment file loaded. [#q={NumberOfQueries}]", nQueries);
		}
		catch (IOException ex)
		{
			throw RankLibException.Create(ex);
		}
	}

	/// <summary>
	/// Compute NDCG at k. NDCG(k) = DCG(k) / DCG_{perfect}(k).
	/// </summary>
	public override double Score(RankList rankList)
	{
		if (rankList.Count == 0)
		{
			return 0;
		}

		var size = K > rankList.Count || K <= 0
			? rankList.Count
			: K;
		var rel = GetRelevanceLabels(rankList);

		double ideal;
		if (_idealGains.TryGetValue(rankList.Id, out var cachedIdeal))
		{
			ideal = cachedIdeal;
		}
		else
		{
			ideal = GetIdealDCG(rel, size);
			_idealGains[rankList.Id] = ideal;
		}

		if (ideal <= 0)
			return 0;

		return GetDCG(rel, size) / ideal;
	}

	public override double[][] SwapChange(RankList rankList)
	{
		var size = rankList.Count > K ? K : rankList.Count;
		var rel = GetRelevanceLabels(rankList);

		var ideal = _idealGains.TryGetValue(rankList.Id, out var cachedIdeal)
			? cachedIdeal
			: GetIdealDCG(rel, size);

		var changes = new double[rankList.Count][];
		for (var i = 0; i < rankList.Count; i++)
		{
			changes[i] = new double[rankList.Count];
			Array.Fill(changes[i], 0);
		}

		for (var i = 0; i < size; i++)
		{
			for (var j = i + 1; j < rankList.Count; j++)
			{
				if (ideal > 0)
				{
					changes[j][i] = changes[i][j] = (Discount(i) - Discount(j)) * (Gain(rel[i]) - Gain(rel[j])) / ideal;
				}
			}
		}

		return changes;
	}

	public override string Name => $"NDCG@{K}";

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
