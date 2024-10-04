using RankLib.Utilities;

namespace RankLib.Learning;

public class DenseDataPoint : DataPoint
{
    public DenseDataPoint(string text) : base(text) { }

    public DenseDataPoint(DenseDataPoint dp)
    {
        _label = dp._label;
        _id = dp._id;
        _description = dp._description;
        _cached = dp._cached;
        _fVals = new float[dp._fVals.Length];
        Array.Copy(dp._fVals, _fVals, dp._fVals.Length);
    }

    public override float GetFeatureValue(int fid)
    {
        if (fid <= 0 || fid >= _fVals.Length)
        {
            if (MissingZero)
            {
                return 0f;
            }
            throw RankLibError.Create($"Error in DenseDataPoint::GetFeatureValue(): requesting unspecified feature, fid={fid}");
        }
        return IsUnknown(_fVals[fid]) ? 0 : _fVals[fid];
    }

    public override void SetFeatureValue(int fid, float fval)
    {
        if (fid <= 0 || fid >= _fVals.Length)
        {
            throw RankLibError.Create($"Error in DenseDataPoint::SetFeatureValue(): feature (id={fid}) not found.");
        }
        _fVals[fid] = fval;
    }

    public override void SetFeatureVector(float[] dfVals)
    {
        _fVals = dfVals;
    }

    public override float[] GetFeatureVector()
    {
        return _fVals;
    }
}
