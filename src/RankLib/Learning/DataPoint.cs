using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
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

	private static string GetKey(string pair) => pair.Substring(0, pair.IndexOf(':'));

	private static string GetValue(string pair) => pair.Substring(pair.LastIndexOf(':') + 1);

	private static ReadOnlySpan<char> GetKey(ReadOnlySpan<char> pair) => pair.Slice(0, pair.IndexOf(':'));
	private static ReadOnlySpan<char>  GetValue(ReadOnlySpan<char> pair) => pair.Slice(pair.LastIndexOf(':') + 1);

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
				span = span[..idx].Trim(); // remove the comment part at the end of the line
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
		var fVals = Parse(text.AsSpan());
		SetFeatureVector(fVals);
	}

	protected DataPoint(ReadOnlySpan<char> span)
	{
		var fVals = Parse(span);
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

public ref struct WhitespaceSplitEnumerator
{
	private ReadOnlySpan<char> _remaining;
	private ReadOnlySpan<char> _current;

	public WhitespaceSplitEnumerator(ReadOnlySpan<char> span)
	{
		_remaining = span;
		_current = default;
	}

	public bool MoveNext()
	{
		while (_remaining.Length > 0 && char.IsWhiteSpace(_remaining[0]))
			_remaining = _remaining[1..];

		if (_remaining.Length == 0)
			return false;

		var end = 0;
		while (end < _remaining.Length && !char.IsWhiteSpace(_remaining[end]))
			end++;

		_current = _remaining[..end];
		_remaining = _remaining[end..];
		return true;
	}

	public ReadOnlySpan<char> Current => _current;

	public WhitespaceSplitEnumerator GetEnumerator() => this;
}

public static class SpanExtensions
{
	public static WhitespaceSplitEnumerator SplitOnWhitespace(this ReadOnlySpan<char> span) => new(span);
}
