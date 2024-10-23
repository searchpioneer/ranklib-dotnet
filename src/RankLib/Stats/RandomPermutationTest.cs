using System.Runtime.CompilerServices;
using System.Security.Cryptography;

namespace RankLib.Stats;

/// <summary>
/// A permutation-based statistical significance test that evaluates
/// the difference between two distributions by running random permutations
/// on the data and calculating a p-value based on how often the permuted
/// difference exceeds the observed difference.
/// </summary>
public class RandomPermutationTest : ISignificanceTest
{
	private static readonly RandomNumberGenerator Rng = RandomNumberGenerator.Create();

	public const int DefaultPermutationCount = 10_000;
	public int PermutationCount { get; }

	public RandomPermutationTest() : this(DefaultPermutationCount)
	{
	}

	public RandomPermutationTest(int permutationCount) => PermutationCount = permutationCount;

	/// <summary>
	/// Run the randomization test
	/// </summary>
	public double Test(Dictionary<string, double> target, Dictionary<string, double> baseline)
	{
		var b = new double[baseline.Count];//baseline
		var t = new double[target.Count];//target
		var c = 0;
		foreach (var key in baseline.Keys)
		{
			b[c] = baseline[key];
			t[c] = target[key];
			c++;
		}

		var trueDiff = Math.Abs(BasicStats.Mean(b) - BasicStats.Mean(t));
		var pValue = 0.0;
		var pb = new double[baseline.Count]; // Permutation of baseline
		var pt = new double[target.Count];   // Permutation of target

		for (var i = 0; i < PermutationCount; i++)
		{
			var bits = GenerateRandomBits(b.Length);
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
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static char[] GenerateRandomBits(int length)
	{
		var numBytes = (length + 7) / 8;
		Span<byte> bytes = stackalloc byte[numBytes];
		Rng.GetBytes(bytes);
		var chars = new char[length];
		var index = 0;
		for (var i = 0; i < numBytes && index < length; i++)
		{
			var currentByte = bytes[i];
			for (var j = 7; j >= 0 && index < length; j--)
				chars[index++] = ((currentByte >> j) & 1) == 1 ? '1' : '0';
		}

		return chars;
	}
}
