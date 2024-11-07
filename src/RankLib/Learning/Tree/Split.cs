using System.Text;
using RankLib.Utilities;

namespace RankLib.Learning.Tree;

/// <summary>
/// Split represents a decision point in a regression or decision tree model,
/// where the dataset is divided based on a feature and a threshold value,
/// determining how data is partitioned to improve the accuracy of ranking or prediction.
/// </summary>
public class Split
{
	private int _featureId = -1;
	private float _threshold;
	private double _output;
	private Split _left;
	private Split _right;

	// Intermediate variables (ONLY used during learning)
	//*DO NOT* attempt to access them once the training is done
	private readonly double _sumLabel;
	private readonly double _sqSumLabel;
	private double _deviance;
	private int[][]? _sortedSampleIDs;
	private int[] _samples = [];

	public Split() { }

	public Split(int featureId, float threshold, double deviance)
	{
		_featureId = featureId;
		_threshold = threshold;
		_deviance = deviance;
	}

	public Split(int[][] sortedSampleIDs, double deviance, double sumLabel, double sqSumLabel)
	{
		_sortedSampleIDs = sortedSampleIDs;
		_deviance = deviance;
		_sumLabel = sumLabel;
		_sqSumLabel = sqSumLabel;
		_output = sumLabel / sortedSampleIDs[0].Length;
	}

	public Split(int[] samples, FeatureHistogram histogram, double deviance, double sumLabel)
	{
		_samples = samples;
		Histogram = histogram;
		_deviance = deviance;
		_sumLabel = sumLabel;
		_output = sumLabel / samples.Length;
	}

	public void Set(int featureId, float threshold, double deviance)
	{
		_featureId = featureId;
		_threshold = threshold;
		_deviance = deviance;
	}

	public FeatureHistogram? Histogram { get; private set; }

	/// <summary>
	/// Whether this split is a root split.
	/// </summary>
	public bool IsRoot { get; set; }

	/// <summary>
	/// Gets or sets the left split.
	/// </summary>
	public Split Left
	{
		get => _left;
		set => _left = value;
	}

	/// <summary>
	/// Gets or sets the right split.
	/// </summary>
	public Split Right
	{
		get => _right;
		set => _right = value;
	}

	/// <summary>
	/// Gets or sets the output.
	/// </summary>
	public double Output
	{
		get => _output;
		set => _output = value;
	}

	/// <summary>
	/// Gets the deviance (Mean squared error "S").
	/// </summary>
	public double Deviance => _deviance;

	public List<Split> Leaves()
	{
		var list = new List<Split>();
		Leaves(list);
		return list;
	}

	private void Leaves(List<Split> leaves)
	{
		if (_featureId == -1)
			leaves.Add(this);
		else
		{
			_left.Leaves(leaves);
			_right.Leaves(leaves);
		}
	}

	public double Eval(DataPoint dataPoint)
	{
		var split = this;
		while (split._featureId != -1)
		{
			split = dataPoint.GetFeatureValue(split._featureId) <= split._threshold
				? split._left
				: split._right;
		}
		return split._output;
	}

	public override string ToString() => ToString("");

	public string ToString(string indent)
	{
		var builder = new StringBuilder();
		builder.Append(indent).Append("<split>\n");
		builder.Append(GetString(indent + "\t"));
		builder.Append(indent).Append("</split>\n");
		return builder.ToString();
	}

	private string GetString(string indent)
	{
		var builder = new StringBuilder();
		if (_featureId == -1)
			builder.Append(indent).Append("<output> ").Append(_output.ToRankLibString()).Append(" </output>\n");
		else
		{
			builder.Append(indent).Append("<feature> ").Append(_featureId).Append(" </feature>\n");
			builder.Append(indent).Append("<threshold> ").Append(_threshold.ToRankLibString()).Append(" </threshold>\n");
			builder.Append(indent).Append("<split pos=\"left\">\n");
			builder.Append(_left.GetString(indent + "\t"));
			builder.Append(indent).Append("</split>\n");
			builder.Append(indent).Append("<split pos=\"right\">\n");
			builder.Append(_right.GetString(indent + "\t"));
			builder.Append(indent).Append("</split>\n");
		}
		return builder.ToString();
	}

	// Internal functions (ONLY used during learning)
	//*DO NOT* attempt to call them once the training is done
	public async Task<bool> TrySplitAsync(double[] trainingLabels, int minLeafSupport)
	{
		if (Histogram is null)
			throw new InvalidOperationException("Histogram is null");

		return await Histogram.FindBestSplitAsync(this, trainingLabels, minLeafSupport).ConfigureAwait(false);
	}

	public int[] GetSamples() => _sortedSampleIDs != null ? _sortedSampleIDs[0] : _samples;

	public int[][]? GetSampleSortedIndex() => _sortedSampleIDs;

	public void ClearSamples()
	{
		_sortedSampleIDs = null;
		_samples = [];
		Histogram = null;
	}
}
