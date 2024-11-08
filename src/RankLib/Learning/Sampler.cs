using RankLib.Utilities;

namespace RankLib.Learning;

using System;
using System.Collections.Generic;
using System.Linq;

public static class Sampler
{
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
