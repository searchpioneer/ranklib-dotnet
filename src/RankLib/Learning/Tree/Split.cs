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
	private Split _left = null;
	private Split _right = null;
	private double _deviance; // Mean squared error "S"
	private int[][]? _sortedSampleIDs;
	private int[] samples = null;
	public FeatureHistogram? hist { get; private set; }

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

	public Split(int[] samples, FeatureHistogram hist, double deviance, double sumLabel)
	{
		this.samples = samples;
		this.hist = hist;
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
		{
			leaves.Add(this);
		}
		else
		{
			_left.Leaves(leaves);
			_right.Leaves(leaves);
		}
	}

	public double Eval(DataPoint dp)
	{
		var n = this;
		while (n._featureId != -1)
		{
			n = dp.GetFeatureValue(n._featureId) <= n._threshold ? n._left : n._right;
		}
		return n._avgLabel;
	}

	public override string ToString() => ToString("");

	public string ToString(string indent)
	{
		var buf = new StringBuilder();
		buf.Append(indent).Append("<split>\n");
		buf.Append(GetString(indent + "\t"));
		buf.Append(indent).Append("</split>\n");
		return buf.ToString();
	}

	private string GetString(string indent)
	{
		var buf = new StringBuilder();
		if (_featureId == -1)
		{
			buf.Append(indent).Append("<output>").Append(_avgLabel).Append("</output>\n");
		}
		else
		{
			buf.Append(indent).Append("<feature>").Append(_featureId).Append("</feature>\n");
			buf.Append(indent).Append("<threshold> ").Append(_threshold).Append("</threshold>\n");
			buf.Append(indent).Append("<split pos=\"left\">\n");
			buf.Append(_left.GetString(indent + "\t"));
			buf.Append(indent).Append("</split>\n");
			buf.Append(indent).Append("<split pos=\"right\">\n");
			buf.Append(_right.GetString(indent + "\t"));
			buf.Append(indent).Append("</split>\n");
		}
		return buf.ToString();
	}

	// Internal functions (ONLY used during learning)
	//*DO NOT* attempt to call them once the training is done
	public bool TrySplit(double[] trainingLabels, int minLeafSupport)
	{
		if (hist is null)
			throw new InvalidOperationException("Histogram is null");

		return hist.FindBestSplit(this, trainingLabels, minLeafSupport);
	}

	public int[] GetSamples() => _sortedSampleIDs != null ? _sortedSampleIDs[0] : samples;

	public int[][]? GetSampleSortedIndex() => _sortedSampleIDs;

	public double SumLabel => _sumLabel;

	public double SqSumLabel => _sqSumLabel;

	public void ClearSamples()
	{
		_sortedSampleIDs = null;
		samples = [];
		hist = null;
	}

	public bool Root { get; set; }
}
