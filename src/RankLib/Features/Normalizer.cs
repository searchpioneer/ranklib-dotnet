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
	/// <param name="featureIds">The feature IDs to normalize</param>
	public abstract void Normalize(RankList rankList, int[] featureIds);

	/// <summary>
	/// Normalizes features for the given rank lists
	/// </summary>
	/// <param name="rankLists">The rank lists with features to normalize.</param>
	/// <param name="featureIds">The feature IDs to normalize</param>
	public void Normalize(List<RankList> rankLists, int[] featureIds)
	{
		foreach (var rankList in rankLists)
			Normalize(rankList, featureIds);
	}

	/// <summary>
	/// Removes duplicate features, returning a new array instance.
	/// </summary>
	/// <param name="featureIds">The features to deduplicate</param>
	/// <returns>A new instance of an array</returns>
	protected static int[] RemoveDuplicateFeatures(int[] featureIds)
	{
		var uniqueSet = new HashSet<int>(featureIds);
		return uniqueSet.ToArray();
	}

	/// <summary>
	/// Gets the name of the normalizer
	/// </summary>
	public abstract string Name { get; }
}
