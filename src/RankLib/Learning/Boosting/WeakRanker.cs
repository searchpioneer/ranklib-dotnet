using RankLib.Utilities;

namespace RankLib.Learning.Boosting;

public class WeakRanker 
{
    private int fid = -1;

    public WeakRanker(int fid) {
        this.fid = fid;
    }

    public int GetFID() {
        return fid;
    }

    public RankList Rank(RankList l) { 
        double[] score = new double[l.Size()];
        for (int i = 0; i < l.Size(); i++) {
            score[i] = l.Get(i).GetFeatureValue(fid);
        }
        int[] idx = Sorter.Sort(score, false);
        return new RankList(l, idx);
    }

    public List<RankList> Rank(List<RankList> l) {
        List<RankList> ll = new List<RankList>();
        for (int i = 0; i < l.Count; i++) {
            ll.Add(Rank(l[i]));
        }
        
        return ll;
    }
}