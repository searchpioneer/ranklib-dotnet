using RankLib.Learning;

namespace RankLib.Features;

public abstract class Normalizer
{
    public abstract void Normalize(RankList rl);
    
    public void Normalize(List<RankList> samples)
    {
        foreach (RankList rankList in samples)
        {
            Normalize(rankList);
        }
    }
    
    public abstract void Normalize(RankList rl, int[] fids);
    
    public void Normalize(List<RankList> samples, int[] fids)
    {
        foreach (RankList rankList in samples)
        {
            Normalize(rankList, fids);
        }
    }
    
    public int[] RemoveDuplicateFeatures(int[] fids)
    {
        var uniqueSet = new HashSet<int>(fids);
        return uniqueSet.ToArray();
    }
    
    public abstract string Name { get; }
}