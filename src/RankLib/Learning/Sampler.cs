namespace RankLib.Learning;

using System;
using System.Collections.Generic;
using System.Linq;

public class Sampler
{
	protected List<RankList> samples = null; // Bag data
	protected List<RankList> remains = null; // Out-of-bag data

	public List<RankList> Sample(List<RankList> samplingPool, float samplingRate, bool withReplacement)
	{
		var r = new Random();
		samples = new List<RankList>();
		var size = (int)(samplingRate * samplingPool.Count);

		if (withReplacement)
		{
			var used = new int[samplingPool.Count];
			Array.Fill(used, 0);

			for (var i = 0; i < size; i++)
			{
				var selected = r.Next(samplingPool.Count);
				samples.Add(samplingPool[selected]);
				used[selected] = 1;
			}

			remains = new List<RankList>();
			for (var i = 0; i < samplingPool.Count; i++)
			{
				if (used[i] == 0)
				{
					remains.Add(samplingPool[i]);
				}
			}
		}
		else
		{
			var indices = Enumerable.Range(0, samplingPool.Count).ToList();
			for (var i = 0; i < size; i++)
			{
				var selected = r.Next(indices.Count);
				samples.Add(samplingPool[indices[selected]]);
				indices.RemoveAt(selected);
			}

			remains = new List<RankList>();
			for (var i = 0; i < indices.Count; i++)
			{
				remains.Add(samplingPool[indices[i]]);
			}
		}

		return samples;
	}

	public List<RankList> Samples => samples;

	public List<RankList> Remains => remains;
}
