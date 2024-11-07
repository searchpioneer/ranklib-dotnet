#pragma warning disable CS8602 // Dereference of a possibly null reference.
#pragma warning disable CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider declaring as nullable.
namespace RankLib.Learning.Tree;

/// <summary>
/// A decision tree-based model used for ranking tasks, which partitions the feature space
/// into regions and fits a constant value (or prediction) in each region,
/// enabling ranking based on the predicted relevance scores for items.
/// </summary>
public class RegressionTree
{
	// -1 for unlimited number of nodes (the size of the tree will then be controlled *ONLY* by minLeafSupport)
	private readonly int _nodes = 10;
	private readonly int _minLeafSupport = 1;
	private Split? _root;
	private List<Split> _leaves;
	private double[] _trainingLabels = [];
	private int[] _index = [];
	private FeatureHistogram _hist;

	public RegressionTree(Split root)
	{
		_root = root;
		_leaves = root.Leaves();
	}

	public RegressionTree(int nLeaves, DataPoint[] trainingSamples, double[] labels, FeatureHistogram hist, int minLeafSupport)
	{
		_nodes = nLeaves;
		_trainingLabels = labels;
		_hist = hist;
		_minLeafSupport = minLeafSupport;
		_index = new int[trainingSamples.Length];

		for (var i = 0; i < trainingSamples.Length; i++)
			_index[i] = i;
	}

	/// <summary>
	/// Fits the tree from the specified training data.
	/// </summary>
	public async Task FitAsync()
	{
		var queue = new List<Split>();
		_root = new Split(_index, _hist, float.MaxValue, 0)
		{
			IsRoot = true
		};

		// Ensure inserts occur only after successful splits
		if (await _root.TrySplitAsync(_trainingLabels, _minLeafSupport).ConfigureAwait(false))
		{
			Insert(queue, _root.Left);
			Insert(queue, _root.Right);
		}

		var taken = 0;
		while ((_nodes == -1 || taken + queue.Count < _nodes) && queue.Count > 0)
		{
			var leaf = queue[0];
			queue.RemoveAt(0);

			if (leaf.GetSamples().Length < 2 * _minLeafSupport)
			{
				taken++;
				continue;
			}

			// unsplit-able (i.e. variance(s)==0; or after-split variance is higher than before)
			if (!await leaf.TrySplitAsync(_trainingLabels, _minLeafSupport).ConfigureAwait(false))
				taken++;
			else
			{
				Insert(queue, leaf.Left);
				Insert(queue, leaf.Right);
			}
		}
		_leaves = _root.Leaves();
	}

	/// <summary>
	/// Gets the tree output for the input sample
	/// </summary>
	public double Eval(DataPoint dataPoint) => _root.Eval(dataPoint);

	/// <summary>
	/// Gets all leaf nodes in the tree
	/// </summary>
	public List<Split> Leaves => _leaves;

	/// <summary>
	/// Clears the samples associated with each leaf (when they are no longer necessary) in order to save memory
	/// </summary>
	public void ClearSamples()
	{
		_trainingLabels = [];
		_index = [];
		_hist = null!;

		for (var i = 0; i < _leaves.Count; i++)
			_leaves[i].ClearSamples();
	}

	/**
     * Generate the string representation of the tree
     */
	public override string ToString() => _root != null ? _root.ToString() : string.Empty;

	public string ToString(string indent) => _root != null ? _root.ToString(indent) : string.Empty;

	public double Variance()
	{
		double variance = 0;
		for (var i = 0; i < _leaves.Count; i++)
			variance += _leaves[i].Deviance;

		return variance;
	}

	private static void Insert(List<Split> splits, Split split)
	{
		var i = 0;
		while (i < splits.Count)
		{
			if (splits[i].Deviance > split.Deviance)
				i++;
			else
				break;
		}
		splits.Insert(i, split);
	}
}
