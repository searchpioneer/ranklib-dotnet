using RankLib.Utilities;

namespace RankLib.Learning.Boosting;

public class WeakRanker
{
	private readonly int fid = -1;

	public WeakRanker(int fid) => this.fid = fid;

	public int GetFID() => fid;

	public RankList Rank(RankList l)
	{
		var score = new double[l.Count];
		for (var i = 0; i < l.Count; i++)
		{
			score[i] = l[i].GetFeatureValue(fid);
		}
		var idx = Sorter.Sort(score, false);
		return new RankList(l, idx);
	}

	public List<RankList> Rank(List<RankList> l)
	{
		var ll = new List<RankList>();
		for (var i = 0; i < l.Count; i++)
		{
			ll.Add(Rank(l[i]));
		}

		return ll;
	}
}
