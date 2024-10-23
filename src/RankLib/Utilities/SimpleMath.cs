namespace RankLib.Utilities;

public static class SimpleMath
{
	private static readonly double Log2 = Math.Log(2);
	private static readonly double Log10 = Math.Log(10);
	private static readonly double Loge = Math.Log(Math.E);

	public static double LogBase2(double value) => Math.Log(value) / Log2;

	public static double LogBase10(double value) => Math.Log(value) / Log10;

	public static double Ln(double value) => Math.Log(value) / Loge;

	public static double P(long count, long total) => (count + 0.5) / (total + 1);

	public static double Round(double val, int n)
	{
		var precision = 1;
		for (var i = 0; i < n; i++)
			precision *= 10;

		return Math.Floor(val * precision + 0.5) / precision;
	}
}
