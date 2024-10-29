namespace RankLib.Learning.NeuralNet;

/// <summary>
/// Propagation parameters
/// </summary>
public class PropParameter
{
	// RankNet
	public int Current { get; } = -1; // index of current data point in the ranked list
	public int[][] PairMap { get; } = [];

	// LambdaRank: RankNet + the following
	public float[][]? PairWeight { get; }
	public float[][] TargetValue { get; } = [];

	// ListNet
	public float[] Labels { get; } = []; // relevance labels

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
