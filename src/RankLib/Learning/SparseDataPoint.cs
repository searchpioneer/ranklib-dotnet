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
		SEQUENTIAL,
		RANDOM
	}

	private static readonly AccessPattern SearchPattern = AccessPattern.RANDOM;

	// The feature ids for known values
	private int[] fIds;

	// Internal search optimizers. Currently unused.
	private int lastMinId = -1;
	private int lastMinPos = -1;

	public SparseDataPoint(string text) : base(text)
	{
	}

	public SparseDataPoint(SparseDataPoint dp)
	{
		Label = dp.Label;
		Id = dp.Id;
		Description = dp.Description;
		Cached = dp.Cached;
		fIds = new int[dp.fIds.Length];
		_fVals = new float[dp._fVals.Length];
		Array.Copy(dp.fIds, 0, fIds, 0, dp.fIds.Length);
		Array.Copy(dp._fVals, 0, _fVals, 0, dp._fVals.Length);
	}

	private int Locate(int fid)
	{
		if (SearchPattern == AccessPattern.SEQUENTIAL)
		{
			if (lastMinId > fid)
			{
				lastMinId = -1;
				lastMinPos = -1;
			}
			while (lastMinPos < _knownFeatures && lastMinId < fid)
			{
				lastMinId = fIds[++lastMinPos];
			}
			if (lastMinId == fid)
			{
				return lastMinPos;
			}
		}
		else if (SearchPattern == AccessPattern.RANDOM)
		{
			var pos = Array.BinarySearch(fIds, fid);
			if (pos >= 0)
			{
				return pos;
			}
		}
		else
		{
			logger.LogWarning("Invalid search pattern specified for sparse data points.");
		}

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
			return _fVals[pos];
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
			_fVals[pos] = fval;
		}
		else
		{
			throw RankLibException.Create("Error in SparseDataPoint::SetFeatureValue(): feature (id=" + fid + ") not found.");
		}
	}

	public override void SetFeatureVector(float[] dfVals)
	{
		fIds = new int[_knownFeatures];
		_fVals = new float[_knownFeatures];
		var pos = 0;
		for (var i = 1; i < dfVals.Length; i++)
		{
			if (!IsUnknown(dfVals[i]))
			{
				fIds[pos] = i;
				_fVals[pos] = dfVals[i];
				pos++;
			}
		}
		if (pos != _knownFeatures)
		{
			throw new InvalidOperationException("Mismatch in known features count.");
		}
	}

	public override float[] GetFeatureVector()
	{
		var dfVals = new float[fIds[_knownFeatures - 1] + 1]; // Adjust for array length
		Array.Fill(dfVals, Unknown);
		for (var i = 0; i < _knownFeatures; i++)
		{
			dfVals[fIds[i]] = _fVals[i];
		}
		return dfVals;
	}
}
