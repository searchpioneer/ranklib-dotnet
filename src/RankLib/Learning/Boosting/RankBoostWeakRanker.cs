namespace RankLib.Learning.Boosting;

internal sealed class RankBoostWeakRanker
{
	public RankBoostWeakRanker(int fid, double threshold)
	{
		Fid = fid;
		Threshold = threshold;
	}

	public int Score(DataPoint dataPoint) => dataPoint.GetFeatureValue(Fid) > Threshold ? 1 : 0;

	public int Fid { get; }

	public double Threshold { get; }

	public override string ToString() => $"{Fid}:{Threshold}";
}
