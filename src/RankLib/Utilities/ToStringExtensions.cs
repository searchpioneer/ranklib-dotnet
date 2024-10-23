using System.Globalization;

namespace RankLib.Utilities;

internal static class ToStringExtensions
{
	public static string ToRankLibString(this double value) => double.IsInteger(value)
		? value.ToString("F1")
		: value.ToString(CultureInfo.InvariantCulture);

	public static string ToRankLibString(this float value) => float.IsInteger(value)
		? value.ToString("F1")
		: value.ToString(CultureInfo.InvariantCulture);
}
