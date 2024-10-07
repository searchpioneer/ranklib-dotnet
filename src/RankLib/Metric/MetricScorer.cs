using RankLib.Learning;

namespace RankLib.Metric;

public abstract class MetricScorer
{
	// The depth parameter, or how deep of a ranked list to use to score the measure.
	public int K { get; set; } = 10;

	public MetricScorer() { }

	/// <summary>
	/// Loads external relevance judgments from a qrel file.
	/// </summary>
	/// <param name="qrelFile">The qrel file containing relevance judgments.</param>
	public virtual void LoadExternalRelevanceJudgment(string qrelFile)
	{
		// Can be overridden if needed, currently no implementation.
	}

	/// <summary>
	/// Scores a list of RankLists by averaging the score of each individual RankList.
	/// </summary>
	/// <param name="rl">The list of RankLists to score.</param>
	/// <returns>The average score across the RankLists.</returns>
	public double Score(List<RankList> rl)
	{
		var score = 0.0;
		for (var i = 0; i < rl.Count; i++)
		{
			score += Score(rl[i]);
		}
		return score / rl.Count;
	}

	/// <summary>
	/// Extracts the relevance labels from the given RankList.
	/// </summary>
	/// <param name="rl">The RankList to extract relevance labels from.</param>
	/// <returns>An array of relevance labels.</returns>
	protected int[] GetRelevanceLabels(RankList rl)
	{
		var rel = new int[rl.Count];
		for (var i = 0; i < rl.Count; i++)
		{
			rel[i] = (int)rl[i].Label;
		}
		return rel;
	}

	/// <summary>
	/// Abstract method to score a RankList.
	/// </summary>
	/// <param name="rl">The RankList to score.</param>
	/// <returns>The score for the RankList.</returns>
	public abstract double Score(RankList rl);

	/// <summary>
	/// Creates a copy of the current MetricScorer.
	/// </summary>
	/// <returns>A copy of this MetricScorer.</returns>
	public abstract MetricScorer Copy();

	/// <summary>
	/// Gets the name of this MetricScorer.
	/// </summary>
	/// <value>The name of this MetricScorer.</value>
	public abstract string Name { get; }

	/// <summary>
	/// Calculates the changes in score caused by swapping elements in the RankList.
	/// </summary>
	/// <param name="rl">The RankList to calculate score changes for.</param>
	/// <returns>A 2D array representing the change in score for each swap.</returns>
	public abstract double[][] SwapChange(RankList rl);
}
