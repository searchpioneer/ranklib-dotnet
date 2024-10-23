using RankLib.Utilities;

namespace RankLib.Learning;

public class DenseDataPoint : DataPoint
{
	public DenseDataPoint(string text) : base(text) { }

	public DenseDataPoint(ReadOnlySpan<char> span) : base(span) { }

	public DenseDataPoint(DenseDataPoint dataPoint)
	{
		Label = dataPoint.Label;
		Id = dataPoint.Id;
		Description = dataPoint.Description;
		Cached = dataPoint.Cached;
		FVals = new float[dataPoint.FVals.Length];
		Array.Copy(dataPoint.FVals, FVals, dataPoint.FVals.Length);
	}

	public override float GetFeatureValue(int fid)
	{
		if (fid <= 0 || fid >= FVals.Length)
		{
			if (MissingZero)
				return 0f;

			throw RankLibException.Create($"Error in DenseDataPoint::GetFeatureValue(): requesting unspecified feature, fid={fid}");
		}

		return IsUnknown(FVals[fid]) ? 0 : FVals[fid];
	}

	public override void SetFeatureValue(int fid, float fval)
	{
		if (fid <= 0 || fid >= FVals.Length)
		{
			throw RankLibException.Create($"Error in DenseDataPoint::SetFeatureValue(): feature (id={fid}) not found.");
		}
		FVals[fid] = fval;
	}

	protected override void SetFeatureVector(float[] dfVals) => FVals = dfVals;

	protected override float[] GetFeatureVector() => FVals;
}
