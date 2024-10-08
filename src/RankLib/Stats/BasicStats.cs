namespace RankLib.Stats;

public static class BasicStats
{
	public static double Mean(double[] values)
	{
		if (values.Length == 0)
		{
			throw new ArgumentException("Error in BasicStats::Mean(): Empty input array.");
		}

		var mean = 0.0;
		foreach (var value in values)
		{
			mean += value;
		}

		return mean / values.Length;
	}
}
