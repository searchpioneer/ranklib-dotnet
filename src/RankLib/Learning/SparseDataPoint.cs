using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Utilities;

namespace RankLib.Learning;

public class SparseDataPoint : DataPoint
{
	// Access pattern of the feature values
	private enum AccessPattern
	{
		Sequential,
		Random
	}

	private static readonly AccessPattern SearchPattern = AccessPattern.Random;

	// The feature ids for known values
	private int[] _featureIds = [];
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
		_featureIds = new int[dp._featureIds.Length];
		FeatureValues = new float[dp.FeatureValues.Length];
		Array.Copy(dp._featureIds, 0, _featureIds, 0, dp._featureIds.Length);
		Array.Copy(dp.FeatureValues, 0, FeatureValues, 0, dp.FeatureValues.Length);
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
			while (_lastMinPos < KnownFeatures && _lastMinId < fid)
				_lastMinId = _featureIds[++_lastMinPos];
			if (_lastMinId == fid)
				return _lastMinPos;
		}
		else if (SearchPattern == AccessPattern.Random)
		{
			var pos = Array.BinarySearch(_featureIds, fid);
			if (pos >= 0)
				return pos;
		}
		else
			throw new InvalidOperationException("Invalid search pattern specified for sparse data points.");

		return -1;
	}

	public bool HasFeature(int fid) => Locate(fid) != -1;

	public override float GetFeatureValue(int featureId)
	{
		if (featureId <= 0 || featureId > FeatureCount)
		{
			if (MissingZero)
				return 0f;

			throw RankLibException.Create("Error in SparseDataPoint::GetFeatureValue(): requesting unspecified feature, fid=" + featureId);
		}

		var pos = Locate(featureId);
		if (pos >= 0)
			return FeatureValues[pos];

		return 0; // Should ideally be returning unknown?
	}

	public override void SetFeatureValue(int featureId, float featureValue)
	{
		if (featureId <= 0 || featureId > FeatureCount)
			throw RankLibException.Create("Error in SparseDataPoint::SetFeatureValue(): feature (id=" + featureId + ") out of range.");

		var pos = Locate(featureId);
		if (pos >= 0)
			FeatureValues[pos] = featureValue;
		else
			throw RankLibException.Create("Error in SparseDataPoint::SetFeatureValue(): feature (id=" + featureId + ") not found.");
	}

	protected override void SetFeatureVector(float[] featureValues)
	{
		_featureIds = new int[KnownFeatures];
		FeatureValues = new float[KnownFeatures];
		var pos = 0;
		for (var i = 1; i < featureValues.Length; i++)
		{
			if (!IsUnknown(featureValues[i]))
			{
				_featureIds[pos] = i;
				FeatureValues[pos] = featureValues[i];
				pos++;
			}
		}
		if (pos != KnownFeatures)
			throw new InvalidOperationException("Mismatch in known features count.");
	}

	protected override float[] GetFeatureVector()
	{
		var featureVector = new float[_featureIds[KnownFeatures - 1] + 1]; // Adjust for array length
		Array.Fill(featureVector, Unknown);
		for (var i = 0; i < KnownFeatures; i++)
			featureVector[_featureIds[i]] = FeatureValues[i];

		return featureVector;
	}
}
