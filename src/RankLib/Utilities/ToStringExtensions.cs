using System.Globalization;

namespace RankLib.Utilities;

internal static class ToStringExtensions
{
	/// <summary>
	/// ToString implementation compatible with java RankLib
	/// </summary>
	public static string ToRankLibString(this double value) => double.IsInteger(value)
		? value.ToString("F1")
		: value.ToString(CultureInfo.InvariantCulture);

	/// <summary>
	/// ToString implementation compatible with java RankLib
	/// </summary>
	public static string ToRankLibString(this float value) => float.IsInteger(value)
		? value.ToString("F1")
		: value.ToString(CultureInfo.InvariantCulture);
}
