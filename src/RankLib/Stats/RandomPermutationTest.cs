namespace RankLib.Stats;

public class RandomPermutationTest : SignificanceTest
{
	public static int NPermutation = 10000;
	private static readonly string[] Pad = ["", "0", "00", "000", "0000", "00000", "000000", "0000000", "00000000", "000000000"];

	/// <summary>
	/// Run the randomization test
	/// </summary>
	public override double Test(Dictionary<string, double> target, Dictionary<string, double> baseline)
	{
		var b = baseline.Values.ToArray(); // Baseline
		var t = target.Values.ToArray();   // Target

		var trueDiff = Math.Abs(BasicStats.Mean(b) - BasicStats.Mean(t));
		var pvalue = 0.0;
		var pb = new double[baseline.Count]; // Permutation of baseline
		var pt = new double[target.Count];   // Permutation of target

		for (var i = 0; i < NPermutation; i++)
		{
			var bits = RandomBitVector(b.Length).ToCharArray();
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
				pvalue += 1.0;
		}

		return pvalue / NPermutation;
	}

	/// <summary>
	/// Generate a random bit vector of a certain size
	/// </summary>
	private string RandomBitVector(int size)
	{
		var random = Random.Shared;
		var output = "";
		for (var i = 0; i < (size / 10) + 1; i++)
		{
			var x = (int)(Math.Pow(2, 10) * random.NextDouble());
			var s = Convert.ToString(x, 2);
			if (s.Length == 11)
				output += s.Substring(1);
			else
				output += Pad[10 - s.Length] + s;
		}
		return output;
	}
}
