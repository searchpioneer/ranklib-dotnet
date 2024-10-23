using RankLib.Utilities;

namespace RankLib.Learning.Tree;

public class FeatureHistogram
{
	public class Config
	{
		public int featureIdx = -1;
		public int thresholdIdx = -1;
		public double S = -1;
		public double errReduced = -1;
	}

	private readonly float _samplingRate;
	private readonly int _maxDegreesOfParallelism;
	private int[] _features = [];
	private float[][] _thresholds = [];
	private double[][] _sum = [];
	private double _sumResponse;
	private double _sqSumResponse;
	private int[][] _count = [];
	private int[][] _sampleToThresholdMap = [];
	private double[] _impacts = [];

	// Reuse of parent resources
	private bool _reuseParent;

	public FeatureHistogram(float samplingRate, int? maxDegreesOfParallelism = null)
	{
		_samplingRate = samplingRate;
		_maxDegreesOfParallelism = maxDegreesOfParallelism ?? Environment.ProcessorCount;
	}

	public async Task ConstructAsync(DataPoint[] samples, double[] labels, int[][] sampleSortedIdx, int[] features, float[][] thresholds, double[] impacts)
	{
		_features = features;
		_thresholds = thresholds;
		_impacts = impacts;
		_sumResponse = 0;
		_sqSumResponse = 0;
		_sum = new double[features.Length][];
		_count = new int[features.Length][];
		_sampleToThresholdMap = new int[features.Length][];

		if (_maxDegreesOfParallelism == 1)
			Construct(samples, labels, sampleSortedIdx, thresholds, 0, features.Length - 1);
		else
		{
			await ParallelExecutor.ExecuteAsync(
				new Worker(this, samples, labels, sampleSortedIdx, thresholds),
				features.Length,
				_maxDegreesOfParallelism);
		}
	}

	protected void Construct(DataPoint[] samples, double[] labels, int[][] sampleSortedIdx, float[][] thresholds, int start, int end)
	{
		for (var i = start; i <= end; i++)
		{
			var fid = _features[i];
			var idx = sampleSortedIdx[i];

			double sumLeft = 0;
			var threshold = thresholds[i];
			var sumLabel = new double[threshold.Length];
			var c = new int[threshold.Length];
			var stMap = new int[samples.Length];

			var last = -1;
			for (var t = 0; t < threshold.Length; t++)
			{
				var j = last + 1;
				for (; j < idx.Length; j++)
				{
					var k = idx[j];
					if (samples[k].GetFeatureValue(fid) > threshold[t])
						break;

					sumLeft += labels[k];
					if (i == 0)
					{
						_sumResponse += labels[k];
						_sqSumResponse += labels[k] * labels[k];
					}
					stMap[k] = t;
				}
				last = j - 1;
				sumLabel[t] = sumLeft;
				c[t] = last + 1;
			}
			_sampleToThresholdMap[i] = stMap;
			_sum[i] = sumLabel;
			_count[i] = c;
		}
	}

	protected internal async Task UpdateAsync(double[] labels)
	{
		_sumResponse = 0;
		_sqSumResponse = 0;

		if (_maxDegreesOfParallelism == 1)
			Update(labels, 0, _features.Length - 1);
		else
		{
			await ParallelExecutor.ExecuteAsync(
				new Worker(this, labels),
				_features.Length,
				_maxDegreesOfParallelism);
		}
	}

	protected void Update(double[] labels, int start, int end)
	{
		for (var f = start; f <= end; f++)
			Array.Fill(_sum[f], 0);

		// for each pseudo-response
		for (var k = 0; k < labels.Length; k++)
		{
			// for each feature
			for (var f = start; f <= end; f++)
			{
				// find the best threshold for a given pseudo-response
				var t = _sampleToThresholdMap[f][k];

				// build a histogram at feature f, threshold t for
				// add the pseudo response that fits in here
				//
				// later, this can let us pick the best split
				// by finding the point t in the histogram with
				// divides the pseudo-response space
				_sum[f][t] += labels[k];
				if (f == 0)
				{
					// accumulate each pseudo response
					// effectively: for each pseudo response k,
					// 		accumulate sumResponse, sqSumResponse
					_sumResponse += labels[k];
					_sqSumResponse += labels[k] * labels[k];
				}
				//count doesn't change, so no need to re-compute
			}
		}

		for (var f = start; f <= end; f++)
		{
			for (var t = 1; t < _thresholds[f].Length; t++)
				_sum[f][t] += _sum[f][t - 1];
		}
	}

	public async Task ConstructAsync(FeatureHistogram parent, int[] soi, double[] labels)
	{
		_features = parent._features;
		_thresholds = parent._thresholds;
		_impacts = parent._impacts;
		_sumResponse = 0;
		_sqSumResponse = 0;
		_sum = new double[_features.Length][];
		_count = new int[_features.Length][];
		_sampleToThresholdMap = parent._sampleToThresholdMap;

		if (_maxDegreesOfParallelism == 1)
			Construct(parent, soi, labels, 0, _features.Length - 1);
		else
		{
			await ParallelExecutor.ExecuteAsync(
				new Worker(this, parent, soi, labels),
				_features.Length,
				_maxDegreesOfParallelism);
		}
	}

	protected void Construct(FeatureHistogram parent, int[] soi, double[] labels, int start, int end)
	{
		for (var i = start; i <= end; i++)
		{
			var threshold = _thresholds[i];
			_sum[i] = new double[threshold.Length];
			_count[i] = new int[threshold.Length];
			Array.Fill(_sum[i], 0);
			Array.Fill(_count[i], 0);
		}

		for (var i = 0; i < soi.Length; i++)
		{
			var k = soi[i];
			for (var f = start; f <= end; f++)
			{
				var t = _sampleToThresholdMap[f][k];
				_sum[f][t] += labels[k];
				_count[f][t]++;
				if (f == 0)
				{
					_sumResponse += labels[k];
					_sqSumResponse += labels[k] * labels[k];
				}
			}
		}

		for (var f = start; f <= end; f++)
		{
			for (var t = 1; t < _thresholds[f].Length; t++)
			{
				_sum[f][t] += _sum[f][t - 1];
				_count[f][t] += _count[f][t - 1];
			}
		}
	}

	public async Task ConstructAsync(FeatureHistogram parent, FeatureHistogram leftSibling, bool reuseParent)
	{
		_reuseParent = reuseParent;
		_features = parent._features;
		_thresholds = parent._thresholds;
		_impacts = parent._impacts;
		_sumResponse = parent._sumResponse - leftSibling._sumResponse;
		_sqSumResponse = parent._sqSumResponse - leftSibling._sqSumResponse;

		if (reuseParent)
		{
			_sum = parent._sum;
			_count = parent._count;
		}
		else
		{
			_sum = new double[_features.Length][];
			_count = new int[_features.Length][];
		}
		_sampleToThresholdMap = parent._sampleToThresholdMap;

		if (_maxDegreesOfParallelism == 1)
			Construct(parent, leftSibling, 0, _features.Length - 1);
		else
		{
			await ParallelExecutor.ExecuteAsync(
				new Worker(this, parent, leftSibling),
				_features.Length,
				_maxDegreesOfParallelism);
		}
	}

	protected void Construct(FeatureHistogram parent, FeatureHistogram leftSibling, int start, int end)
	{
		for (var f = start; f <= end; f++)
		{
			var threshold = _thresholds[f];
			if (!_reuseParent)
			{
				_sum[f] = new double[threshold.Length];
				_count[f] = new int[threshold.Length];
			}

			for (var t = 0; t < threshold.Length; t++)
			{
				_sum[f][t] = parent._sum[f][t] - leftSibling._sum[f][t];
				_count[f][t] = parent._count[f][t] - leftSibling._count[f][t];
			}
		}
	}

	private Config FindBestSplit(int[] usedFeatures, int minLeafSupport, int start, int end)
	{
		var cfg = new Config();
		var totalCount = _count[start][_count[start].Length - 1];
		for (var f = start; f <= end; f++)
		{
			var i = usedFeatures[f];
			var threshold = _thresholds[i];

			for (var t = 0; t < threshold.Length; t++)
			{
				var countLeft = _count[i][t];
				var countRight = totalCount - countLeft;
				if (countLeft < minLeafSupport || countRight < minLeafSupport)
					continue;

				var sumLeft = _sum[i][t];
				var sumRight = _sumResponse - sumLeft;

				// See: http://www.dcc.fc.up.pt/~ltorgo/PhD/th3.pdf  pp69
				//
				// This S approximates the error. S is relative to this decision
				//
				// Error is really (sqSumResponse / count) - S / count
				// we add back in the error calculation

				var s = (sumLeft * sumLeft / countLeft) + (sumRight * sumRight / countRight);
				var errSt = (_sqSumResponse / totalCount) * (s / totalCount);
				if (cfg.S < s)
				{
					cfg.S = s;
					cfg.featureIdx = i;
					cfg.thresholdIdx = t;
					cfg.errReduced = errSt;
				}
			}
		}
		return cfg;
	}

	public async Task<bool> FindBestSplitAsync(Split sp, double[] labels, int minLeafSupport)
	{
		if (sp.GetDeviance() >= 0 && sp.GetDeviance() <= 0)
			return false; // No need to split

		int[] usedFeatures;
		if (_samplingRate < 1) //need to do sub sampling (feature sampling)
		{
			var size = (int)(_samplingRate * _features.Length);
			usedFeatures = new int[size];
			//put all features into a pool
			var featurePool = new List<int>();
			for (var i = 0; i < _features.Length; i++)
				featurePool.Add(i);

			//do sampling, without replacement
			var random = ThreadsafeSeedableRandom.Shared;
			for (var i = 0; i < size; i++)
			{
				var selected = random.Next(featurePool.Count);
				usedFeatures[i] = featurePool[selected];
				featurePool.RemoveAt(selected);
			}
		}
		else //no sub sampling, all features will be used
		{
			usedFeatures = new int[_features.Length];
			for (var i = 0; i < _features.Length; i++)
				usedFeatures[i] = i;
		}

		var best = new Config();
		if (_maxDegreesOfParallelism == 1)
			best = FindBestSplit(usedFeatures, minLeafSupport, 0, usedFeatures.Length - 1);
		else
		{
			var workers = await ParallelExecutor.ExecuteAsync(
				new Worker(this, usedFeatures, minLeafSupport),
				usedFeatures.Length,
				_maxDegreesOfParallelism);

			foreach (var worker in workers)
			{
				if (best.S < worker.Cfg.S)
					best = worker.Cfg;
			}
		}

		// ReSharper disable once CompareOfFloatsByEqualityOperator
		if (best.S == -1) // cannot be split, for some reason...
			return false;

		// bestFeaturesHist is the best features
		var bestFeaturesHist = _sum[best.featureIdx];
		var sampleCount = _count[best.featureIdx];

		var s = bestFeaturesHist[^1];
		var c = sampleCount[bestFeaturesHist.Length - 1];

		var sumLeft = bestFeaturesHist[best.thresholdIdx];
		var countLeft = sampleCount[best.thresholdIdx];

		var sumRight = s - sumLeft;
		var countRight = c - countLeft;

		var left = new int[countLeft];
		var right = new int[countRight];
		var l = 0;
		var r = 0;
		var idx = sp.GetSamples();
		for (var j = 0; j < idx.Length; j++)
		{
			var k = idx[j];
			if (_sampleToThresholdMap[best.featureIdx][k] <= best.thresholdIdx)// go to the left
				left[l++] = k;
			else // go to the right
				right[r++] = k;
		}

		// update impact with info on best
		_impacts[best.featureIdx] += best.errReduced;

		var lh = new FeatureHistogram(_samplingRate, _maxDegreesOfParallelism);
		await lh.ConstructAsync(sp.Histogram!, left, labels);
		var rh = new FeatureHistogram(_samplingRate, _maxDegreesOfParallelism);
		await rh.ConstructAsync(sp.Histogram!, lh, !sp.IsRoot);

		var var = _sqSumResponse - _sumResponse * _sumResponse / idx.Length;
		var varLeft = lh._sqSumResponse - lh._sumResponse * lh._sumResponse / left.Length;
		var varRight = rh._sqSumResponse - rh._sumResponse * rh._sumResponse / right.Length;

		sp.Set(_features[best.featureIdx], _thresholds[best.featureIdx][best.thresholdIdx], var);
		sp.SetLeft(new Split(left, lh, varLeft, sumLeft));
		sp.SetRight(new Split(right, rh, varRight, sumRight));

		sp.ClearSamples();

		return true;
	}

	// Worker class for multithreading tasks
	private class Worker : WorkerThread
	{
		private FeatureHistogram _fh;
		private int _type;
		private int[] _usedFeatures;
		private int _minLeafSup;
		internal Config Cfg;
		private double[] _labels;
		private FeatureHistogram _parent;
		private int[] _soi;
		private FeatureHistogram _leftSibling;
		private DataPoint[] _samples;
		private int[][] _sampleSortedIdx;
		private float[][] _thresholds;

		private Worker() { }

		public Worker(FeatureHistogram fh, int[] usedFeatures, int minLeafSup)
		{
			_type = 0;
			_fh = fh;
			_usedFeatures = usedFeatures;
			_minLeafSup = minLeafSup;
		}

		public Worker(FeatureHistogram fh, double[] labels)
		{
			_type = 1;
			_fh = fh;
			_labels = labels;
		}

		public Worker(FeatureHistogram fh, FeatureHistogram parent, int[] soi, double[] labels)
		{
			_type = 2;
			_fh = fh;
			_parent = parent;
			_soi = soi;
			_labels = labels;
		}

		public Worker(FeatureHistogram fh, FeatureHistogram parent, FeatureHistogram leftSibling)
		{
			_type = 3;
			_fh = fh;
			_parent = parent;
			_leftSibling = leftSibling;
		}

		public Worker(FeatureHistogram fh, DataPoint[] samples, double[] labels, int[][] sampleSortedIdx, float[][] thresholds)
		{
			_type = 4;
			_fh = fh;
			_samples = samples;
			_labels = labels;
			_sampleSortedIdx = sampleSortedIdx;
			_thresholds = thresholds;
		}

		public override Task RunAsync() =>
			Task.Run(() =>
			{
				switch (_type)
				{
					case 0:
						Cfg = _fh.FindBestSplit(_usedFeatures, _minLeafSup, start, end);
						break;
					case 1:
						_fh.Update(_labels, start, end);
						break;
					case 2:
						_fh.Construct(_parent, _soi, _labels, start, end);
						break;
					case 3:
						_fh.Construct(_parent, _leftSibling, start, end);
						break;
					case 4:
						_fh.Construct(_samples, _labels, _sampleSortedIdx, _thresholds, start, end);
						break;
				}
			});

		public override WorkerThread Clone()
		{
			var wk = new Worker
			{
				_fh = _fh,
				_type = _type,

				// find best split (type == 0)
				_usedFeatures = _usedFeatures,
				_minLeafSup = _minLeafSup,

				// update
				_labels = _labels,

				// construct
				_parent = _parent,
				_soi = _soi,

				// construct sibling
				_leftSibling = _leftSibling,

				// construct full
				_samples = _samples,
				_sampleSortedIdx = _sampleSortedIdx,
				_thresholds = _thresholds
			};

			return wk;
		}
	}
}
