using RankLib.Learning;

namespace RankLib.Metric;

public abstract class MetricScorer
{
    // The depth parameter, or how deep of a ranked list to use to score the measure.
    protected int _k = 10;

    public MetricScorer() { }

    /// <summary>
    /// Sets the depth parameter, or how deep of a ranked list to use to score the measure.
    /// </summary>
    /// <param name="k">The new depth for this measure.</param>
    public void SetK(int k)
    {
        _k = k;
    }

    /// <summary>
    /// Gets the depth parameter, or how deep of a ranked list to use to score the measure.
    /// </summary>
    /// <returns>The depth parameter k.</returns>
    public int GetK()
    {
        return _k;
    }

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
        double score = 0.0;
        for (int i = 0; i < rl.Count; i++)
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
        int[] rel = new int[rl.Size()];
        for (int i = 0; i < rl.Size(); i++)
        {
            rel[i] = (int)rl.Get(i).GetLabel();
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
    /// <returns>The name of this MetricScorer.</returns>
    public abstract string Name();

    /// <summary>
    /// Calculates the changes in score caused by swapping elements in the RankList.
    /// </summary>
    /// <param name="rl">The RankList to calculate score changes for.</param>
    /// <returns>A 2D array representing the change in score for each swap.</returns>
    public abstract double[][] SwapChange(RankList rl);
}