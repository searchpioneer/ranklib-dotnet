using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using RankLib.Utilities;

namespace RankLib.Learning;

public abstract class DataPoint
{
	public static int MaxFeature = 51;
	public static bool MissingZero = false;
	public static int FeatureIncrease = 10;

	protected const float Unknown = float.NaN;

	// attributes
	protected float[] FVals = null; // _fVals[0] is unused. Feature id MUST start from 1

	// helper attributes
	protected int _knownFeatures; // number of known feature values

	// internal to learning procedures
	public double Cached { get; set; } = -1.0;

	protected static bool IsUnknown(float fVal) => float.IsNaN(fVal);

	private static ReadOnlySpan<char> GetKey(ReadOnlySpan<char> pair) => pair.Slice(0, pair.IndexOf(':'));
	private static ReadOnlySpan<char> GetValue(ReadOnlySpan<char> pair) => pair.Slice(pair.LastIndexOf(':') + 1);

	/// <summary>
	/// Parse the given line of text to construct a dense array of feature values and reset metadata.
	/// </summary>
	/// <param name="span">The text to parse</param>
	/// <returns>Dense array of feature values</returns>
	protected float[] Parse(ReadOnlySpan<char> span)
	{
		var fVals = new float[MaxFeature];
		Array.Fill(fVals, Unknown);
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
				_knownFeatures++;
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
					Array.Copy(fVals, tmp, fVals.Length);
					Array.Fill(tmp, Unknown, fVals.Length, MaxFeature - fVals.Length);
					fVals = tmp;
				}

				fVals[f] = float.Parse(val);

				if (f > FeatureCount)
					FeatureCount = f;

				if (f > lastFeature)
					lastFeature = f;
			}

			// shrink fVals
			var shrinkFVals = new float[lastFeature + 1];
			Array.Copy(fVals, shrinkFVals, lastFeature + 1);
			fVals = shrinkFVals;
		}
		catch (Exception ex)
		{
			throw new InvalidOperationException("Error in DataPoint::Parse", ex);
		}

		return fVals;
	}

	/// <summary>
	/// Get the value of the feature with the given feature ID
	/// </summary>
	public abstract float GetFeatureValue(int fid);

	/// <summary>
	/// Sets the value of the feature with the given feature ID
	/// </summary>
	public abstract void SetFeatureValue(int fid, float fval);

	/// <summary>
	/// Sets the value of all features with the provided dense array of feature values
	/// </summary>
	protected abstract void SetFeatureVector(float[] dfVals);

	/// <summary>
	/// Gets the value of all features as a dense array of feature values.
	/// </summary>
	protected abstract float[] GetFeatureVector();

	protected DataPoint() { }

	protected DataPoint(ReadOnlySpan<char> span)
	{
		var fVals = Parse(span);
		// ReSharper disable once VirtualMemberCallInConstructor
		SetFeatureVector(fVals);
	}

	public string Id { get; set; } = string.Empty;

	public string Description { get; set; } = string.Empty;

	public float Label { get; set; }

	public int FeatureCount { get; protected set; }

	public void ResetCached() => Cached = -100000000.0f;

	// Override ToString method
	public override string ToString()
	{
		var fval = GetFeatureVector();
		var output = new StringBuilder();
		output.Append($"{(int)Label} qid:{Id} ");

		for (var i = 1; i < fval.Length; i++)
		{
			if (!IsUnknown(fval[i]))
				output.Append($"{i}:{fval[i]}{(i == fval.Length - 1 ? "" : " ")}");
		}

		output.Append($" {Description}");
		return output.ToString();
	}
}
