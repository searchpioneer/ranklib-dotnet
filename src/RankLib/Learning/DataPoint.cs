using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using RankLib.Utilities;

namespace RankLib.Learning;

/// <summary>
/// A datapoint
/// </summary>
public abstract class DataPoint
{
	public static int MaxFeature = 51;
	public static bool MissingZero = false;
	public static int FeatureIncrease = 10;

	protected const float Unknown = float.NaN;

	// attributes
	protected float[] FeatureValues = []; // _fVals[0] is unused. Feature id MUST start from 1

	// helper attributes
	protected int KnownFeatures; // number of known feature values

	// internal to learning procedures
	public double Cached { get; set; } = -1;

	protected static bool IsUnknown(float featureValue) => float.IsNaN(featureValue);

	private static ReadOnlySpan<char> GetKey(ReadOnlySpan<char> pair) => pair.Slice(0, pair.IndexOf(':'));
	private static ReadOnlySpan<char> GetValue(ReadOnlySpan<char> pair) => pair.Slice(pair.LastIndexOf(':') + 1);

	/// <summary>
	/// Parses the given line of text to construct a dense array of feature values and reset metadata.
	/// </summary>
	/// <param name="span">The text to parse</param>
	/// <returns>Dense array of feature values</returns>
	protected float[] Parse(ReadOnlySpan<char> span)
	{
		var featureValues = new float[MaxFeature];
		Array.Fill(featureValues, Unknown);
		var lastFeature = -1;

		try
		{
			var idx = span.IndexOf('#');
			if (idx != -1)
			{
				Description = span[idx..].ToString();
				span = span[..idx].Trim();
			}

			var enumerator = span.SplitOnWhitespace();
			enumerator.MoveNext();

			Label = float.Parse(enumerator.Current);

			if (Label < 0)
				throw new InvalidOperationException("Relevance label cannot be negative.");

			enumerator.MoveNext();
			Id = GetValue(enumerator.Current).ToString();

			while (enumerator.MoveNext())
			{
				KnownFeatures++;
				var key = GetKey(enumerator.Current);
				var val = GetValue(enumerator.Current);
				var f = int.Parse(key);

				if (f <= 0)
					throw new InvalidOperationException("Cannot use feature numbering less than or equal to zero. Start your features at 1.");

				if (f >= MaxFeature)
				{
					while (f >= MaxFeature)
						MaxFeature += FeatureIncrease;

					var tmp = new float[MaxFeature];
					Array.Copy(featureValues, tmp, featureValues.Length);
					Array.Fill(tmp, Unknown, featureValues.Length, MaxFeature - featureValues.Length);
					featureValues = tmp;
				}

				featureValues[f] = float.Parse(val);

				if (f > FeatureCount)
					FeatureCount = f;

				if (f > lastFeature)
					lastFeature = f;
			}

			// shrink fVals
			var shrinkFeatureValues = new float[lastFeature + 1];
			Array.Copy(featureValues, shrinkFeatureValues, lastFeature + 1);
			featureValues = shrinkFeatureValues;
		}
		catch (Exception ex)
		{
			throw new InvalidOperationException("Error parsing data point", ex);
		}

		return featureValues;
	}

	/// <summary>
	/// Get the value of the feature with the given feature ID
	/// </summary>
	public abstract float GetFeatureValue(int featureId);

	/// <summary>
	/// Sets the value of the feature with the given feature ID
	/// </summary>
	public abstract void SetFeatureValue(int featureId, float featureValue);

	/// <summary>
	/// Sets the value of all features with the provided dense array of feature values
	/// </summary>
	protected abstract void SetFeatureVector(float[] featureValues);

	/// <summary>
	/// Gets the value of all features as a dense array of feature values.
	/// </summary>
	protected abstract float[] GetFeatureVector();

	/// <summary>
	/// Initializes a new instance of a <see cref="DataPoint"/>
	/// </summary>
	protected DataPoint() { }

	/// <summary>
	/// Initializes a new instance of a <see cref="DataPoint"/> from the given span.
	/// </summary>
	/// <param name="span">The span containing data to initialize the instance with</param>
	protected DataPoint(ReadOnlySpan<char> span)
	{
		var featureValues = Parse(span);
		// ReSharper disable once VirtualMemberCallInConstructor
		SetFeatureVector(featureValues);
	}

	/// <summary>
	/// Gets or sets the ID of the data point. The ID is typically an identifier for the query.
	/// </summary>
	public string Id { get; protected set; } = string.Empty;

	/// <summary>
	/// Gets the description. The description typically contains the document ID and query text.
	/// </summary>
	public string Description { get; protected set; } = string.Empty;

	/// <summary>
	/// Gets the label
	/// </summary>
	public float Label { get; protected set; }

	/// <summary>
	/// Gets the feature count
	/// </summary>
	public int FeatureCount { get; private set; }

	public void ResetCached() => Cached = -1;

	// Override ToString method
	public override string ToString()
	{
		var featureVector = GetFeatureVector();
		var output = new StringBuilder();
		output.Append($"{(int)Label} qid:{Id} ");

		for (var i = 1; i < featureVector.Length; i++)
		{
			if (!IsUnknown(featureVector[i]))
				output.Append($"{i}:{featureVector[i]}{(i == featureVector.Length - 1 ? "" : " ")}");
		}

		output.Append($" {Description}");
		return output.ToString();
	}
}
