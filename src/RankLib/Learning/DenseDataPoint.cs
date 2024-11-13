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

	public DenseDataPoint(DenseDataPoint dataPoint)
	{
		Label = dataPoint.Label;
		Id = dataPoint.Id;
		Description = dataPoint.Description;
		Cached = dataPoint.Cached;
		FeatureValues = new float[dataPoint.FeatureValues.Length];
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

		return IsUnknown(FeatureValues[featureId]) ? 0 : FeatureValues[featureId];
	}

	public override void SetFeatureValue(int featureId, float featureValue)
	{
		if (featureId <= 0 || featureId >= FeatureValues.Length)
		{
			throw RankLibException.Create($"Error in DenseDataPoint::SetFeatureValue(): feature (id={featureId}) not found.");
		}
		FeatureValues[featureId] = featureValue;
	}

	protected override void SetFeatureVector(float[] featureValues) => FeatureValues = featureValues;

	protected override float[] GetFeatureVector() => FeatureValues;
}
