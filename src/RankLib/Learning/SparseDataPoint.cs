using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Utilities;

namespace RankLib.Learning;

public class SparseDataPoint : DataPoint
{
	private static readonly ILogger<SparseDataPoint> logger = NullLogger<SparseDataPoint>.Instance;

	// Access pattern of the feature values
	private enum AccessPattern
	{
		Sequential,
		Random
	}

	private static readonly AccessPattern SearchPattern = AccessPattern.Random;

	// The feature ids for known values
	private int[] _fIds;

	// Internal search optimizers. Currently unused.
	private int _lastMinId = -1;
	private int _lastMinPos = -1;

	public SparseDataPoint(ReadOnlySpan<char> span) : base(span) { }

	public SparseDataPoint(SparseDataPoint dp)
	{
		Label = dp.Label;
		Id = dp.Id;
		Description = dp.Description;
		Cached = dp.Cached;
		_fIds = new int[dp._fIds.Length];
		FVals = new float[dp.FVals.Length];
		Array.Copy(dp._fIds, 0, _fIds, 0, dp._fIds.Length);
		Array.Copy(dp.FVals, 0, FVals, 0, dp.FVals.Length);
	}

	private int Locate(int fid)
	{
		if (SearchPattern == AccessPattern.Sequential)
		{
			if (_lastMinId > fid)
			{
				_lastMinId = -1;
				_lastMinPos = -1;
			}
			while (_lastMinPos < _knownFeatures && _lastMinId < fid)
				_lastMinId = _fIds[++_lastMinPos];
			if (_lastMinId == fid)
				return _lastMinPos;
		}
		else if (SearchPattern == AccessPattern.Random)
		{
			var pos = Array.BinarySearch(_fIds, fid);
			if (pos >= 0)
				return pos;
		}
		else
			logger.LogWarning("Invalid search pattern specified for sparse data points.");

		return -1;
	}

	public bool HasFeature(int fid) => Locate(fid) != -1;

	public override float GetFeatureValue(int fid)
	{
		if (fid <= 0 || fid > FeatureCount)
		{
			if (MissingZero)
			{
				return 0f;
			}
			throw RankLibException.Create("Error in SparseDataPoint::GetFeatureValue(): requesting unspecified feature, fid=" + fid);
		}

		var pos = Locate(fid);
		if (pos >= 0)
		{
			return FVals[pos];
		}

		return 0; // Should ideally be returning unknown?
	}

	public override void SetFeatureValue(int fid, float fval)
	{
		if (fid <= 0 || fid > FeatureCount)
		{
			throw RankLibException.Create("Error in SparseDataPoint::SetFeatureValue(): feature (id=" + fid + ") out of range.");
		}

		var pos = Locate(fid);
		if (pos >= 0)
		{
			FVals[pos] = fval;
		}
		else
		{
			throw RankLibException.Create("Error in SparseDataPoint::SetFeatureValue(): feature (id=" + fid + ") not found.");
		}
	}

	protected override void SetFeatureVector(float[] dfVals)
	{
		_fIds = new int[_knownFeatures];
		FVals = new float[_knownFeatures];
		var pos = 0;
		for (var i = 1; i < dfVals.Length; i++)
		{
			if (!IsUnknown(dfVals[i]))
			{
				_fIds[pos] = i;
				FVals[pos] = dfVals[i];
				pos++;
			}
		}
		if (pos != _knownFeatures)
		{
			throw new InvalidOperationException("Mismatch in known features count.");
		}
	}

	protected override float[] GetFeatureVector()
	{
		var dfVals = new float[_fIds[_knownFeatures - 1] + 1]; // Adjust for array length
		Array.Fill(dfVals, Unknown);
		for (var i = 0; i < _knownFeatures; i++)
		{
			dfVals[_fIds[i]] = FVals[i];
		}
		return dfVals;
	}
}
