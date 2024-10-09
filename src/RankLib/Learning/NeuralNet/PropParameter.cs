namespace RankLib.Learning.NeuralNet;

public class PropParameter
{
	// RankNet
	public int Current { get; protected set; } = -1; // index of current data point in the ranked list
	public int[][] PairMap { get; protected set; } = [];

	// LambdaRank: RankNet + the following
	public float[][] PairWeight { get; protected set; } = [];
	public float[][] TargetValue { get; protected set; } = [];

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
