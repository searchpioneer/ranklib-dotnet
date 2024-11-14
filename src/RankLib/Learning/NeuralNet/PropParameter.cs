namespace RankLib.Learning.NeuralNet;

/// <summary>
/// Propagation parameters used during learning.
/// </summary>
public class PropParameter
{
	/// <summary>
	/// Gets the index of the current data point in the rank list
	/// </summary>
	/// <remarks>
	/// Used by <see cref="RankNet"/>
	/// </remarks>
	public int Current { get; } = -1;

	/// <summary>
	/// Gets the pair map
	/// </summary>
	/// <remarks>
	/// Used by <see cref="RankNet"/>
	/// </remarks>
	public int[][] PairMap { get; } = [];

	/// <summary>
	/// Gets the pair weight
	/// </summary>
	/// <remarks>
	/// Used by <see cref="LambdaRank"/> and <see cref="RankNet"/>
	/// </remarks>
	public float[][]? PairWeight { get; }

	/// <summary>
	/// Gets the target values
	/// </summary>
	/// <remarks>
	/// Used by <see cref="LambdaRank"/> and <see cref="RankNet"/>
	/// </remarks>
	public float[][] TargetValue { get; } = [];

	/// <summary>
	/// Gets the relevance labels
	/// </summary>
	/// <remarks>
	/// Used by <see cref="ListNet"/>
	/// </remarks>
	public float[] Labels { get; } = [];

	// Constructor for RankNet
	public PropParameter(int current, int[][] pairMap)
	{
		Current = current;
		PairMap = pairMap;
	}

	// Constructor for LambdaRank
	public PropParameter(int current, int[][] pairMap, float[][] pairWeight, float[][] targetValue)
	{
		Current = current;
		PairMap = pairMap;
		PairWeight = pairWeight;
		TargetValue = targetValue;
	}

	// Constructor for ListNet
	public PropParameter(float[] labels) => Labels = labels;
}
