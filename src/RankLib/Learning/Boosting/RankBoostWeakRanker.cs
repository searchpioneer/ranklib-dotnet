namespace RankLib.Learning.Boosting;

public class RankBoostWeakRanker
{
	private readonly int _fid;
	private readonly double _threshold;

	public RankBoostWeakRanker(int fid, double threshold)
	{
		_fid = fid;
		_threshold = threshold;
	}

	public int Score(DataPoint p) => p.GetFeatureValue(_fid) > _threshold ? 1 : 0;

	public int GetFid() => _fid;

	public double GetThreshold() => _threshold;

	public override string ToString() => $"{_fid}:{_threshold}";
}
