using RankLib.Utilities;

namespace RankLib.Learning.Tree;

public sealed class FeatureHistogram
{
	private sealed class Config
	{
		public int FeatureIdx = -1;
		public int ThresholdIdx = -1;
		public double S = -1;
		public double ErrReduced = -1;
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
	private bool _reuseParent;

	/// <summary>
	/// Initializes a new instance of <see cref="FeatureHistogram"/>
	/// </summary>
	/// <param name="samplingRate">the sampling rate</param>
	/// <param name="maxDegreesOfParallelism">
	/// the maximum number of concurrent tasks allowed when splitting
	/// up workloads that can be run on multiple threads.
	/// If unspecified, uses the count of all available processors.
	/// </param>
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
			var partitions =
				Partitioner.PartitionEnumerable(_features.Length, _maxDegreesOfParallelism);
			await Parallel.ForEachAsync(
				partitions,
				new ParallelOptions { MaxDegreeOfParallelism = _maxDegreesOfParallelism },
				async (range, cancellationToken) =>
				{
					await Task.Run(
						() => Construct(samples, labels, sampleSortedIdx, thresholds, range.Start.Value,
							range.End.Value), cancellationToken).ConfigureAwait(false);
				}).ConfigureAwait(false);
		}
	}

		public async Task<bool> FindBestSplitAsync(Split sp, double[] labels, int minLeafSupport)
	{
		if (sp.Deviance == 0)
			return false; // No need to split

		int[] usedFeatures;
		if (_samplingRate < 1) //need to do sub sampling (feature sampling)
		{
			var size = (int)(_samplingRate * _features.Length);
			usedFeatures = new int[size];
			//put all features into a pool
			var featurePool = new List<int>(_features.Length);
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
			var tasks =
				Partitioner.PartitionEnumerable(usedFeatures.Length, _maxDegreesOfParallelism)
					.Select<Range, Task<Config>>(range => new Task<Config>(() => FindBestSplit(usedFeatures, minLeafSupport, range.Start.Value, range.End.Value)))
					.ToList();

			await Parallel.ForEachAsync(tasks, new ParallelOptions { MaxDegreeOfParallelism = _maxDegreesOfParallelism }, async (task, _) =>
			{
				task.Start();
				await task.ConfigureAwait(false);
			}).ConfigureAwait(false);

			foreach (var task in tasks)
			{
				if (best.S < task.Result.S)
					best = task.Result;
			}
		}

		// ReSharper disable once CompareOfFloatsByEqualityOperator
		if (best.S == -1) // cannot be split, for some reason...
			return false;

		// bestFeaturesHist is the best features
		var bestFeaturesHist = _sum[best.FeatureIdx];
		var sampleCount = _count[best.FeatureIdx];

		var s = bestFeaturesHist[^1];
		var c = sampleCount[bestFeaturesHist.Length - 1];

		var sumLeft = bestFeaturesHist[best.ThresholdIdx];
		var countLeft = sampleCount[best.ThresholdIdx];

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
			if (_sampleToThresholdMap[best.FeatureIdx][k] <= best.ThresholdIdx)// go to the left
				left[l++] = k;
			else // go to the right
				right[r++] = k;
		}

		// update impact with info on best
		_impacts[best.FeatureIdx] += best.ErrReduced;

		var lh = new FeatureHistogram(_samplingRate, _maxDegreesOfParallelism);
		await lh.ConstructAsync(sp.Histogram!, left, labels).ConfigureAwait(false);
		var rh = new FeatureHistogram(_samplingRate, _maxDegreesOfParallelism);
		await rh.ConstructAsync(sp.Histogram!, lh, !sp.IsRoot).ConfigureAwait(false);

		var var = _sqSumResponse - _sumResponse * _sumResponse / idx.Length;
		var varLeft = lh._sqSumResponse - lh._sumResponse * lh._sumResponse / left.Length;
		var varRight = rh._sqSumResponse - rh._sumResponse * rh._sumResponse / right.Length;

		sp.Set(_features[best.FeatureIdx], _thresholds[best.FeatureIdx][best.ThresholdIdx], var);
		sp.Left = new Split(left, lh, varLeft, sumLeft);
		sp.Right = new Split(right, rh, varRight, sumRight);
		sp.ClearSamples();

		return true;
	}

	private void Construct(DataPoint[] samples, double[] labels, int[][] sampleSortedIdx, float[][] thresholds, int start, int end)
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

	internal async Task UpdateAsync(double[] labels)
	{
		_sumResponse = 0;
		_sqSumResponse = 0;

		if (_maxDegreesOfParallelism == 1)
			Update(labels, 0, _features.Length - 1);
		else
		{
			var partitions =
				Partitioner.PartitionEnumerable(_features.Length, _maxDegreesOfParallelism);
			await Parallel.ForEachAsync(
				partitions,
				new ParallelOptions { MaxDegreeOfParallelism = _maxDegreesOfParallelism },
				async (range, cancellationToken) =>
				{
					await Task.Run(() => Update(labels, range.Start.Value, range.End.Value), cancellationToken)
						.ConfigureAwait(false);
				}).ConfigureAwait(false);
		}
	}

	private void Update(double[] labels, int start, int end)
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

	private async Task ConstructAsync(FeatureHistogram parent, int[] soi, double[] labels)
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
			Construct(soi, labels, 0, _features.Length - 1);
		else
		{
			var partitions =
				Partitioner.PartitionEnumerable(_features.Length, _maxDegreesOfParallelism);

			await Parallel.ForEachAsync(
				partitions,
				new ParallelOptions { MaxDegreeOfParallelism = _maxDegreesOfParallelism },
				async (range, cancellationToken) =>
				{
					await Task.Run(() => Construct(soi, labels, range.Start.Value, range.End.Value),
						cancellationToken).ConfigureAwait(false);
				}).ConfigureAwait(false);
		}
	}

	private void Construct(int[] soi, double[] labels, int start, int end)
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

	private async Task ConstructAsync(FeatureHistogram parent, FeatureHistogram leftSibling, bool reuseParent)
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
			var partitions =
				Partitioner.PartitionEnumerable(_features.Length, _maxDegreesOfParallelism);
			await Parallel.ForEachAsync(
				partitions,
				new ParallelOptions { MaxDegreeOfParallelism = _maxDegreesOfParallelism },
				async (range, cancellationToken) =>
				{
					await Task.Run(() => Construct(parent, leftSibling, range.Start.Value, range.End.Value),
						cancellationToken).ConfigureAwait(false);
				}).ConfigureAwait(false);
		}
	}

	private void Construct(FeatureHistogram parent, FeatureHistogram leftSibling, int start, int end)
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
					cfg.FeatureIdx = i;
					cfg.ThresholdIdx = t;
					cfg.ErrReduced = errSt;
				}
			}
		}
		return cfg;
	}
}
