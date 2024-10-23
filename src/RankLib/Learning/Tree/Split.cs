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
	private double _avgLabel;
	private Split _left;
	private Split _right;

	// Intermediate variables (ONLY used during learning)
	//*DO NOT* attempt to access them once the training is done
	private readonly double _sumLabel;
	private readonly double _sqSumLabel;
	private double _deviance; // Mean squared error "S"
	private int[][]? _sortedSampleIDs;
	private int[] _samples = [];

	public FeatureHistogram? Histogram { get; private set; }

	public bool IsRoot { get; set; }

	public Split Left
	{
		get => _left;
		set => _left = value;
	}

	public Split Right
	{
		get => _right;
		set => _right = value;
	}

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
		_avgLabel = sumLabel / sortedSampleIDs[0].Length;
	}

	public Split(int[] samples, FeatureHistogram histogram, double deviance, double sumLabel)
	{
		_samples = samples;
		Histogram = histogram;
		_deviance = deviance;
		_sumLabel = sumLabel;
		_avgLabel = sumLabel / samples.Length;
	}

	public void Set(int featureId, float threshold, double deviance)
	{
		_featureId = featureId;
		_threshold = threshold;
		_deviance = deviance;
	}

	public void SetOutput(float output) => _avgLabel = output;

	public double GetDeviance() => _deviance;

	public double GetOutput() => _avgLabel;

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
		var n = this;
		while (n._featureId != -1)
		{
			n = dataPoint.GetFeatureValue(n._featureId) <= n._threshold
				? n._left
				: n._right;
		}
		return n._avgLabel;
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
			builder.Append(indent).Append("<output> ").Append(_avgLabel.ToRankLibString()).Append(" </output>\n");
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

	public double SumLabel => _sumLabel;

	public double SqSumLabel => _sqSumLabel;

	public void ClearSamples()
	{
		_sortedSampleIDs = null;
		_samples = [];
		Histogram = null;
	}
}
