using System.Text;
using System.Text.RegularExpressions;

namespace RankLib.Learning;

public abstract partial class DataPoint
{
	[GeneratedRegex("\\s+")]
	private static partial Regex WhitespaceRegex();

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

	protected static string GetKey(string pair) => pair.Substring(0, pair.IndexOf(':'));

	protected static string GetValue(string pair) => pair.Substring(pair.LastIndexOf(':') + 1);

	/// <summary>
	/// Parse the given line of text to construct a dense array of feature values and reset metadata.
	/// </summary>
	/// <param name="text">The text to parse</param>
	/// <returns>Dense array of feature values</returns>
	protected float[] Parse(string text)
	{
		// TODO: convert to parsing from Span<T>

		var fVals = new float[MaxFeature];
		Array.Fill(fVals, Unknown);
		var lastFeature = -1;

		try
		{
			var idx = text.IndexOf('#');
			if (idx != -1)
			{
				Description = text.Substring(idx);
				text = text.Substring(0, idx).Trim(); // remove the comment part at the end of the line
			}

			var fs = WhitespaceRegex().Split(text);
			Label = float.Parse(fs[0]);

			if (Label < 0)
				throw new InvalidOperationException("Relevance label cannot be negative.");

			Id = GetValue(fs[1]);

			for (var i = 2; i < fs.Length; i++)
			{
				_knownFeatures++;
				var key = GetKey(fs[i]);
				var val = GetValue(fs[i]);
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

	// Abstract methods for feature value operations
	public abstract float GetFeatureValue(int fid);
	public abstract void SetFeatureValue(int fid, float fval);
	protected abstract void SetFeatureVector(float[] dfVals);
	protected abstract float[] GetFeatureVector();

	// Default constructor
	protected DataPoint() { }

	// Constructor to initialize DataPoint from text
	protected DataPoint(string text)
	{
		var fVals = Parse(text);
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
