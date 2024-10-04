using RankLib.Learning;

public class RBWeakRanker
{
    private readonly int _fid = -1;
    private readonly double _threshold = 0.0;

    public RBWeakRanker(int fid, double threshold)
    {
        _fid = fid;
        _threshold = threshold;
    }

    public int Score(DataPoint p)
    {
        return p.GetFeatureValue(_fid) > _threshold ? 1 : 0;
    }

    public int GetFid()
    {
        return _fid;
    }

    public double GetThreshold()
    {
        return _threshold;
    }

    public override string ToString()
    {
        return $"{_fid}:{_threshold}";
    }
}