using RankLib.Utilities;

namespace RankLib.Learning;

/// <summary>
/// A dense data point
/// </summary>
public sealed class DenseDataPoint : DataPoint
{
	/// <summary>
	/// Initializes a new instance of <see cref="DenseDataPoint"/> from the given span.
	/// </summary>
	/// <param name="span">The span containing data to initialize the instance with</param>
	public DenseDataPoint(ReadOnlySpan<char> span) : base(span) { }

	/// <summary>
	/// Initializes a new instance of <see cref="DenseDataPoint"/> from values.
	/// </summary>
	/// <param name="label">The relevance label for the data point.</param>
	/// <param name="id">The ID of the data point. The ID is typically an identifier for the query.</param>
	/// <param name="featureValues">The feature values.
	/// Feature Ids are 1-based, so this array is expected to have <see cref="DataPoint.Unknown"/> in index 0,
	/// and feature values from index 1</param>
	/// <param name="description">The optional description</param>
	public DenseDataPoint(float label, string id, float[] featureValues, string? description = null)
		: base(label, id, featureValues, description) { }

	/// <summary>
	/// Initializes a new instance of <see cref="DenseDataPoint"/> from another dense data point.
	/// </summary>
	/// <param name="dataPoint">The data point to copy.</param>
	public DenseDataPoint(DenseDataPoint dataPoint)
	{
		Label = dataPoint.Label;
		Id = dataPoint.Id;
		Description = dataPoint.Description;
		Cached = dataPoint.Cached;
		FeatureValues = new float[dataPoint.FeatureValues.Length];
		FeatureCount = dataPoint.FeatureCount;
		Array.Copy(dataPoint.FeatureValues, FeatureValues, dataPoint.FeatureValues.Length);
	}

	public override float GetFeatureValue(int featureId)
	{
		if (featureId <= 0 || featureId >= FeatureValues.Length)
		{
			if (MissingZero)
				return 0f;

			throw RankLibException.Create($"Error in DenseDataPoint::GetFeatureValue(): requesting unspecified feature, fid={featureId}");
		}

		var featureValue = FeatureValues[featureId];
		return IsUnknown(featureValue) ? 0 : featureValue;
	}

	public override void SetFeatureValue(int featureId, float featureValue)
	{
		if (featureId <= 0 || featureId >= FeatureValues.Length)
			throw RankLibException.Create($"Error in DenseDataPoint::SetFeatureValue(): feature (id={featureId}) not found.");

		FeatureValues[featureId] = featureValue;
	}

	protected override void SetFeatureVector(float[] featureValues) => FeatureValues = featureValues;

	protected override float[] GetFeatureVector() => FeatureValues;
}
