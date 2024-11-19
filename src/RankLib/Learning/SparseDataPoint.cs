using RankLib.Utilities;

namespace RankLib.Learning;

/// <summary>
/// A sparse data point
/// </summary>
public sealed class SparseDataPoint : DataPoint
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

	/// <summary>
	/// Initializes a new instance of <see cref="SparseDataPoint"/> from the given span.
	/// </summary>
	/// <param name="span">The span containing data to initialize the instance with</param>
	public SparseDataPoint(ReadOnlySpan<char> span) : base(span) { }

	/// <summary>
	/// Initializes a new instance of <see cref="SparseDataPoint"/> from values.
	/// </summary>
	/// <param name="label">The relevance label for the data point.</param>
	/// <param name="id">The ID of the data point. The ID is typically an identifier for the query.</param>
	/// <param name="featureValues">The feature values.
	/// Feature Ids are 1-based, so this array is expected to have <see cref="DataPoint.Unknown"/> in index 0,
	/// and feature values from index 1</param>
	/// <param name="description">The optional description</param>
	public SparseDataPoint(float label, string id, float[] featureValues, string? description = null)
		: base(label, id, featureValues, description) { }

	/// <summary>
	/// Initializes a new instance of <see cref="SparseDataPoint"/> from another sparse data point.
	/// </summary>
	/// <param name="dataPoint">The data point to copy.</param>
	public SparseDataPoint(SparseDataPoint dataPoint)
	{
		Label = dataPoint.Label;
		Id = dataPoint.Id;
		Description = dataPoint.Description;
		Cached = dataPoint.Cached;
		_featureIds = new int[dataPoint._featureIds.Length];
		FeatureValues = new float[dataPoint.FeatureValues.Length];
		FeatureCount = dataPoint.FeatureCount;
		Array.Copy(dataPoint._featureIds, 0, _featureIds, 0, dataPoint._featureIds.Length);
		Array.Copy(dataPoint.FeatureValues, 0, FeatureValues, 0, dataPoint.FeatureValues.Length);
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

			throw RankLibException.Create($"Requesting unspecified feature: {featureId}");
		}

		var pos = Locate(featureId);
		if (pos >= 0)
			return FeatureValues[pos];

		return 0; // Should ideally be returning unknown?
	}

	public override void SetFeatureValue(int featureId, float featureValue)
	{
		if (featureId <= 0 || featureId > FeatureCount)
			throw RankLibException.Create($"Feature out of range: {featureId}");

		var pos = Locate(featureId);
		if (pos >= 0)
			FeatureValues[pos] = featureValue;
		else
			throw RankLibException.Create($"Feature not found: {featureId}");
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
