using System.Globalization;
using System.Text;

namespace RankLib.Learning.Tree;

public class Split
{
	// Key attributes of a split (tree node)
	private int _featureId = -1;
	private float _threshold;
	private double _avgLabel;

	// Intermediate variables (ONLY used during learning)
	//*DO NOT* attempt to access them once the training is done
	private readonly double _sumLabel;
	private readonly double _sqSumLabel;
	private Split _left;
	private Split _right;
	private double _deviance; // Mean squared error "S"
	private int[][]? _sortedSampleIDs;
	private int[] _samples = [];
	public FeatureHistogram? Histogram { get; private set; }

	public bool IsRoot { get; set; }

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

	public void SetLeft(Split s) => _left = s;

	public void SetRight(Split s) => _right = s;

	public void SetOutput(float output) => _avgLabel = output;

	public Split GetLeft() => _left;

	public Split GetRight() => _right;

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
			builder.Append(indent).Append("<output> ").Append(GetString(_avgLabel)).Append(" </output>\n");
		else
		{
			builder.Append(indent).Append("<feature> ").Append(_featureId).Append(" </feature>\n");
			builder.Append(indent).Append("<threshold> ").Append(GetString(_threshold)).Append(" </threshold>\n");
			builder.Append(indent).Append("<split pos=\"left\">\n");
			builder.Append(_left.GetString(indent + "\t"));
			builder.Append(indent).Append("</split>\n");
			builder.Append(indent).Append("<split pos=\"right\">\n");
			builder.Append(_right.GetString(indent + "\t"));
			builder.Append(indent).Append("</split>\n");
		}
		return builder.ToString();
	}

	private static string GetString(double value) => double.IsInteger(value) ? value.ToString("F1") : value.ToString(CultureInfo.InvariantCulture);
	private static string GetString(float value) => float.IsInteger(value) ? value.ToString("F1") : value.ToString(CultureInfo.InvariantCulture);

	// Internal functions (ONLY used during learning)
	//*DO NOT* attempt to call them once the training is done
	public async Task<bool> TrySplitAsync(double[] trainingLabels, int minLeafSupport)
	{
		if (Histogram is null)
			throw new InvalidOperationException("Histogram is null");

		return await Histogram.FindBestSplitAsync(this, trainingLabels, minLeafSupport);
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
