using RankLib.Learning;

namespace RankLib.Features;

/// <summary>
/// Normalizes features
/// </summary>
public abstract class Normalizer
{
	public abstract void Normalize(RankList rankList);

	public void Normalize(List<RankList> samples)
	{
		foreach (var rankList in samples)
		{
			Normalize(rankList);
		}
	}

	public abstract void Normalize(RankList rankList, int[] fids);

	public void Normalize(List<RankList> samples, int[] fids)
	{
		foreach (var rankList in samples)
		{
			Normalize(rankList, fids);
		}
	}

	public int[] RemoveDuplicateFeatures(int[] fids)
	{
		var uniqueSet = new HashSet<int>(fids);
		return uniqueSet.ToArray();
	}

	public abstract string Name { get; }
}
