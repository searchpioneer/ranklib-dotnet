using Microsoft.Extensions.Logging;
using RankLib.Metric;

namespace RankLib.Learning;

/// <summary>
/// Parameters for a ranker
/// </summary>
public interface IRankerParameters
{
	internal void Log(ILogger logger);
}

/// <summary>
/// A ranker
/// </summary>
public interface IRanker
{
	public List<RankList> Samples { get; set; }

	public List<RankList>? ValidationSamples { get; set; }

	public int[] Features { get; set; }

	public MetricScorer Scorer { get; set; }

	void Init();
	void Learn();
	double Eval(DataPoint dataPoint);
	string Model { get; }
	void LoadFromString(string fullText);
	string Name { get; }

	public IRankerParameters Parameters { get; set; }

	RankList Rank(RankList rankList);

	List<RankList> Rank(List<RankList> rankLists)
	{
		var rankedRankLists = new List<RankList>(rankLists.Count);
		for (var i = 0; i < rankLists.Count; i++)
		{
			rankedRankLists.Add(Rank(rankLists[i]));
		}
		return rankedRankLists;
	}

	void Save(string modelFile);

	double GetScoreOnTrainingData();
}

/// <summary>
/// A generic ranker
/// </summary>
/// <typeparam name="TParameters">The type of rank parameters</typeparam>
public interface IRanker<TParameters> : IRanker
	where TParameters : IRankerParameters
{
	public new TParameters Parameters { get; set; }
}
