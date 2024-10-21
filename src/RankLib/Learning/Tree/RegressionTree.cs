namespace RankLib.Learning.Tree;

public class RegressionTree
{
	// Parameters
	protected int nodes = 10; // -1 for unlimited number of nodes (the size of the tree will then be controlled *ONLY* by minLeafSupport)
	protected int minLeafSupport = 1;

	// Member variables and functions
	protected Split? root = null;
	protected List<Split> leaves = null;

	protected DataPoint[] trainingSamples = [];
	protected double[] trainingLabels = [];
	protected int[] features = [];
	protected float[][] thresholds = [];
	protected int[] index = [];
	protected FeatureHistogram hist = null;

	public RegressionTree(Split root)
	{
		this.root = root;
		leaves = root.Leaves();
	}

	public RegressionTree(int nLeaves, DataPoint[] trainingSamples, double[] labels, FeatureHistogram hist, int minLeafSupport)
	{
		nodes = nLeaves;
		this.trainingSamples = trainingSamples;
		trainingLabels = labels;
		this.hist = hist;
		this.minLeafSupport = minLeafSupport;
		index = new int[trainingSamples.Length];
		for (var i = 0; i < trainingSamples.Length; i++)
		{
			index[i] = i;
		}
	}

	/**
     * Fit the tree from the specified training data
     */
	public async Task Fit()
	{
		var queue = new LinkedList<Split>();
		root = new Split(index, hist, float.MaxValue, 0)
		{
			Root = true
		};

		// Ensure inserts occur only after successful splits
		if (await root.TrySplit(trainingLabels, minLeafSupport))
		{
			Insert(queue, root.GetLeft());
			Insert(queue, root.GetRight());
		}

		var taken = 0;
		while ((nodes == -1 || taken + queue.Count < nodes) && queue.Count > 0)
		{
			var leaf = queue.First!.Value;
			queue.RemoveFirst();

			if (leaf.GetSamples().Length < 2 * minLeafSupport)
			{
				taken++;
				continue;
			}

			if (!(await leaf.TrySplit(trainingLabels, minLeafSupport)))
			{
				taken++;
			}
			else
			{
				Insert(queue, leaf.GetLeft());
				Insert(queue, leaf.GetRight());
			}
		}
		leaves = root.Leaves();
	}

	/**
     * Get the tree output for the input sample
     * @param dp
     * @return
     */
	public double Eval(DataPoint dp) => root.Eval(dp);

	/**
     * Retrieve all leaf nodes in the tree
     * @return
     */
	public List<Split> Leaves => leaves;

	/**
     * Clear samples associated with each leaf (when they are no longer necessary) in order to save memory
     */
	public void ClearSamples()
	{
		trainingSamples = null;
		trainingLabels = null;
		features = null;
		thresholds = null;
		index = null;
		hist = null;
		for (var i = 0; i < leaves.Count; i++)
		{
			leaves[i].ClearSamples();
		}
	}

	/**
     * Generate the string representation of the tree
     */
	public override string ToString()
	{
		if (root != null)
		{
			return root.ToString();
		}
		return string.Empty;
	}

	public string ToString(string indent)
	{
		if (root != null)
		{
			return root.ToString(indent);
		}
		return string.Empty;
	}

	public double Variance()
	{
		double var = 0;
		for (var i = 0; i < leaves.Count; i++)
		{
			var += leaves[i].GetDeviance();
		}
		return var;
	}

	protected void Insert(LinkedList<Split> ls, Split s)
	{
		var current = ls.First;
		while (current != null)
		{
			if (current.Value.GetDeviance() > s.GetDeviance())
			{
				current = current.Next;
			}
			else
			{
				break;
			}
		}

		if (current == null)
			ls.AddFirst(s);
		else
			ls.AddBefore(current, s);
	}
}
