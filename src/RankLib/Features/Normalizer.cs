using RankLib.Learning;

namespace RankLib.Features;

/// <summary>
/// Normalizes features
/// </summary>
public abstract class Normalizer
{
	/// <summary>
	/// Normalizes features for the given rank list
	/// </summary>
	/// <param name="rankList">The rank list with features to normalize.</param>
	public abstract void Normalize(RankList rankList);

	/// <summary>
	/// Normalizes features for the given rank lists
	/// </summary>
	/// <param name="rankLists">The rank lists with features to normalize.</param>
	public void Normalize(List<RankList> rankLists)
	{
		foreach (var rankList in rankLists)
			Normalize(rankList);
	}

	/// <summary>
	/// Normalizes features for the given rank list
	/// </summary>
	/// <param name="rankList">The rank list with features to normalize.</param>
	/// <param name="fids">The feature IDs to normalize</param>
	public abstract void Normalize(RankList rankList, int[] fids);

	/// <summary>
	/// Normalizes features for the given rank lists
	/// </summary>
	/// <param name="rankLists">The rank lists with features to normalize.</param>
	/// <param name="fids">The feature IDs to normalize</param>
	public void Normalize(List<RankList> rankLists, int[] fids)
	{
		foreach (var rankList in rankLists)
			Normalize(rankList, fids);
	}

	protected static int[] RemoveDuplicateFeatures(int[] fids)
	{
		var uniqueSet = new HashSet<int>(fids);
		return uniqueSet.ToArray();
	}

	/// <summary>
	/// Gets the name of the normalizer
	/// </summary>
	public abstract string Name { get; }
}
