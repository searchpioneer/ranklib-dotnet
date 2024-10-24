using RankLib.Utilities;

namespace RankLib.Learning.Boosting;

internal sealed class AdaRankWeakRanker
{
	public int Fid { get; }

	public AdaRankWeakRanker(int fid) => Fid = fid;

	public RankList Rank(RankList l)
	{
		var score = new double[l.Count];
		for (var i = 0; i < l.Count; i++)
			score[i] = l[i].GetFeatureValue(Fid);

		var idx = Sorter.Sort(score, false);
		return new RankList(l, idx);
	}

	public List<RankList> Rank(List<RankList> rankLists)
	{
		var rankedRankLists = new List<RankList>(rankLists.Count);
		for (var i = 0; i < rankLists.Count; i++)
			rankedRankLists.Add(Rank(rankLists[i]));

		return rankedRankLists;
	}
}
