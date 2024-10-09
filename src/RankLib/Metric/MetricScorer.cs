using RankLib.Learning;

namespace RankLib.Metric;

/// <summary>
/// Retrieval metric score
/// </summary>
public abstract class MetricScorer
{
	/// <summary>
	/// The depth parameter, or how deep of a ranked list to use to score the measure.
	/// </summary>
	public int K { get; set; } = 10;

	/// <summary>
	/// Loads external relevance judgments from a file.
	/// </summary>
	/// <param name="queryRelevanceFile">The file containing relevance judgments.</param>
	public virtual void LoadExternalRelevanceJudgment(string queryRelevanceFile)
	{
	}

	/// <summary>
	/// Scores a list of <see cref="RankList"/> by averaging the score of each individual rank list.
	/// </summary>
	/// <param name="rankLists">The list of rank lists to score.</param>
	/// <returns>The average score across the rank lists.</returns>
	public double Score(List<RankList> rankLists)
	{
		var score = 0.0;
		for (var i = 0; i < rankLists.Count; i++)
		{
			score += Score(rankLists[i]);
		}
		return score / rankLists.Count;
	}

	/// <summary>
	/// Extracts the relevance labels from the given RankList.
	/// </summary>
	/// <param name="rankList">The RankList to extract relevance labels from.</param>
	/// <returns>An array of relevance labels.</returns>
	protected int[] GetRelevanceLabels(RankList rankList)
	{
		var rel = new int[rankList.Count];
		for (var i = 0; i < rankList.Count; i++)
		{
			rel[i] = (int)rankList[i].Label;
		}
		return rel;
	}

	/// <summary>
	/// Score a <see cref="RankList"/>.
	/// </summary>
	/// <param name="rankList">The rank list to score.</param>
	/// <returns>The score for the rank list.</returns>
	public abstract double Score(RankList rankList);

	/// <summary>
	/// Gets the name of this MetricScorer.
	/// </summary>
	/// <value>The name of this MetricScorer.</value>
	public abstract string Name { get; }

	/// <summary>
	/// Calculates the changes in score caused by swapping elements in the RankList.
	/// </summary>
	/// <param name="rankList">The RankList to calculate score changes for.</param>
	/// <returns>A 2D array representing the change in score for each swap.</returns>
	public abstract double[][] SwapChange(RankList rankList);
}
