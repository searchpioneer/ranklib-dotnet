using System.Text;
using RankLib.Utilities;

namespace RankLib.Stats;

public class RandomPermutationTest : SignificanceTest
{
	public const int DefaultPermutationCount = 10_000;
	public int PermutationCount { get; }

	public RandomPermutationTest() : this(DefaultPermutationCount)
	{
	}

	public RandomPermutationTest(int permutationCount) => PermutationCount = permutationCount;

	/// <summary>
	/// Run the randomization test
	/// </summary>
	public override double Test(Dictionary<string, double> target, Dictionary<string, double> baseline)
	{
		var b = baseline.Values.ToArray(); // Baseline
		var t = target.Values.ToArray();   // Target

		var trueDiff = Math.Abs(BasicStats.Mean(b) - BasicStats.Mean(t));
		var pValue = 0.0;
		var pb = new double[baseline.Count]; // Permutation of baseline
		var pt = new double[target.Count];   // Permutation of target

		for (var i = 0; i < PermutationCount; i++)
		{
			var bits = RandomBitVector(b.Length);
			for (var j = 0; j < b.Length; j++)
			{
				if (bits[j] == '0')
				{
					pb[j] = b[j];
					pt[j] = t[j];
				}
				else
				{
					pb[j] = t[j];
					pt[j] = b[j];
				}
			}

			var pDiff = Math.Abs(BasicStats.Mean(pb) - BasicStats.Mean(pt));
			if (pDiff >= trueDiff)
				pValue += 1.0;
		}

		return pValue / PermutationCount;
	}

	/// <summary>
	/// Generate a random bit vector of a certain size
	/// </summary>
	private static string RandomBitVector(int size)
	{
		var random = ThreadsafeSeedableRandom.Shared;
		var output = new StringBuilder(size * 2);
		for (var i = 0; i < (size / 10) + 1; i++)
		{
			var x = (int)(Math.Pow(2, 10) * random.NextDouble());
			var s = Convert.ToString(x, 2);
			if (s.Length == 11)
				output.Append(s.Substring(1));
			else
				output.Append(s.PadLeft(10, '0'));
		}
		return output.ToString();
	}
}
