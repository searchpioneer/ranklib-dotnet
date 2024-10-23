namespace RankLib.Stats;

public static class BasicStats
{
	/// <summary>
	/// Computes the arithmetic mean for an array of values.
	/// </summary>
	/// <param name="values">The values</param>
	/// <returns>The arithmetic mean</returns>
	/// <exception cref="ArgumentException"><paramref name="values"/> is empty.</exception>
	public static double Mean(double[] values)
	{
		if (values.Length == 0)
			throw new ArgumentException("values is empty", nameof(values));

		var mean = 0.0;
		foreach (var value in values)
			mean += value;

		return mean / values.Length;
	}
}
