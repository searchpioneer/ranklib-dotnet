using System.Text;
using System.Text.RegularExpressions;

namespace RankLib.Learning;

public abstract class DataPoint
{
	public static bool MissingZero = false;
	protected const int FeatureIncrease = 10;
	protected const float Unknown = float.NaN;

	// attributes
	protected float[] _fVals = null; // _fVals[0] is unused. Feature id MUST start from 1

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

		var maxFeature = 51;
		var fval = new float[maxFeature];
		Array.Fill(fval, Unknown);
		var lastFeature = -1;

		try
		{
			var idx = text.IndexOf('#');
			if (idx != -1)
			{
				Description = text.Substring(idx);
				text = text.Substring(0, idx).Trim(); // remove the comment part at the end of the line
			}

			var fs = Regex.Split(text, "\\s+");
			Label = float.Parse(fs[0]);

			if (Label < 0)
			{
				throw new InvalidOperationException("Relevance label cannot be negative. System will now exit.");
			}

			Id = GetValue(fs[1]);

			for (var i = 2; i < fs.Length; i++)
			{
				_knownFeatures++;
				var key = GetKey(fs[i]);
				var val = GetValue(fs[i]);
				var f = int.Parse(key);

				if (f <= 0)
				{
					throw new InvalidOperationException("Cannot use feature numbering less than or equal to zero. Start your features at 1.");
				}

				if (f >= maxFeature)
				{
					while (f >= maxFeature)
					{
						maxFeature += FeatureIncrease;
					}

					var tmp = new float[maxFeature];
					Array.Copy(fval, tmp, fval.Length);
					Array.Fill(tmp, Unknown, fval.Length, maxFeature - fval.Length);
					fval = tmp;
				}

				fval[f] = float.Parse(val);

				if (f > FeatureCount)
				{
					FeatureCount = f;
				}

				if (f > lastFeature)
				{
					lastFeature = f;
				}
			}

			// shrink fVals
			var shrinkFVals = new float[lastFeature + 1];
			Array.Copy(fval, shrinkFVals, lastFeature + 1);
			fval = shrinkFVals;
		}
		catch (Exception ex)
		{
			throw new InvalidOperationException("Error in DataPoint::Parse", ex);
		}

		return fval;
	}

	// Abstract methods for feature value operations
	public abstract float GetFeatureValue(int fid);
	public abstract void SetFeatureValue(int fid, float fval);
	public abstract void SetFeatureVector(float[] dfVals);
	public abstract float[] GetFeatureVector();

	// Default constructor
	protected DataPoint() { }

	// Constructor to initialize DataPoint from text
	protected DataPoint(string text) => SetFeatureVector(Parse(text));

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
			{
				output.Append($"{i}:{fval[i]}{(i == fval.Length - 1 ? "" : " ")}");
			}
		}

		output.Append($" {Description}");
		return output.ToString();
	}
}
