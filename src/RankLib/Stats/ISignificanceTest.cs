namespace RankLib.Stats;

/// <summary>
/// A test for statistical significance.
/// </summary>
public interface ISignificanceTest
{
	public double Test(Dictionary<string, double> target, Dictionary<string, double> baseline);
}
