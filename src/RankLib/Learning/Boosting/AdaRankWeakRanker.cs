using RankLib.Utilities;

namespace RankLib.Learning.Boosting;

public class AdaRankWeakRanker
{
	public int Fid { get; } = -1;

	public AdaRankWeakRanker(int fid) => Fid = fid;

	public RankList Rank(RankList l)
	{
		var score = new double[l.Count];
		for (var i = 0; i < l.Count; i++)
		{
			score[i] = l[i].GetFeatureValue(Fid);
		}
		var idx = Sorter.Sort(score, false);
		return new RankList(l, idx);
	}

	public List<RankList> Rank(List<RankList> rankLists)
	{
		var rankedRankLists = new List<RankList>(rankLists.Count);
		// ReSharper disable once ForCanBeConvertedToForeach
		for (var i = 0; i < rankLists.Count; i++)
		{
			rankedRankLists.Add(Rank(rankLists[i]));
		}

		return rankedRankLists;
	}
}
