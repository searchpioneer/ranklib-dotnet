using RankLib.Utilities;

namespace RankLib.Learning;

/// <summary>
/// Samples rank lists from a pool of rank lists
/// </summary>
public static class Sampler
{
	/// <summary>
	/// Samples a subset of rank lists from a given pool based on the specified sampling rate.
	/// </summary>
	/// <param name="samplingPool">The pool of rank lists to sample from.</param>
	/// <param name="samplingRate">The sampling rate, indicating the proportion of the pool to sample. Must be between 0 and 1.</param>
	/// <param name="withReplacement">Indicates whether sampling should be done with replacement (<c>true</c>)
	/// or without replacement (<c>false</c>). When <c>true</c>, an itemcan be selected more than once when
	/// generating the samples.</param>
	/// <returns>
	/// A tuple containing two lists of rank lists:
	/// <list type="bullet">
	/// <item><description><c>samples</c>: The sampled subset of rank lists.</description></item>
	/// <item><description><c>remains</c>: The remaining rank lists in the pool that were not selected.</description></item>
	/// </list>
	/// </returns>
	/// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="samplingRate"/> is not between 0 and 1.</exception>
	public static (List<RankList> samples, List<RankList> remains) Sample(List<RankList> samplingPool, float samplingRate, bool withReplacement)
	{
		if (samplingRate is < 0 or > 1)
			throw new ArgumentOutOfRangeException(nameof(samplingRate), "sampling rate must be between 0 and 1");

		var samples = new List<RankList>();
		var remains = new List<RankList>();
		var size = (int)(samplingRate * samplingPool.Count);

		if (withReplacement)
		{
			var used = new int[samplingPool.Count];
			Array.Fill(used, 0);

			for (var i = 0; i < size; i++)
			{
				var selected = ThreadsafeSeedableRandom.Shared.Next(samplingPool.Count);
				samples.Add(samplingPool[selected]);
				used[selected] = 1;
			}

			for (var i = 0; i < samplingPool.Count; i++)
			{
				if (used[i] == 0)
					remains.Add(samplingPool[i]);
			}
		}
		else
		{
			var indices = Enumerable.Range(0, samplingPool.Count).ToList();
			for (var i = 0; i < size; i++)
			{
				var selected = ThreadsafeSeedableRandom.Shared.Next(indices.Count);
				samples.Add(samplingPool[indices[selected]]);
				indices.RemoveAt(selected);
			}

			for (var i = 0; i < indices.Count; i++)
				remains.Add(samplingPool[indices[i]]);
		}

		return (samples, remains);
	}
}
