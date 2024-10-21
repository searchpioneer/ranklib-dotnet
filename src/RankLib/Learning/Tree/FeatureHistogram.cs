using RankLib.Utilities;

namespace RankLib.Learning.Tree;

public static class ParallelExecutor
{
	public static async Task<TWorker[]> ExecuteAsync<TWorker>(TWorker worker, int nTasks, int maxDegreeOfParallelism = -1, CancellationToken cancellationToken = default)
		where TWorker : WorkerThread
	{
		if (maxDegreeOfParallelism <= 0)
		{
			maxDegreeOfParallelism = Environment.ProcessorCount;
		}

		var partition = Partition(nTasks, maxDegreeOfParallelism);
		var workers = new TWorker[partition.Length - 1];

		await Parallel.ForEachAsync(Enumerable.Range(0, partition.Length - 1),
			new ParallelOptions
			{
				MaxDegreeOfParallelism = maxDegreeOfParallelism,
				CancellationToken = cancellationToken
			},
			async (i, ct) =>
			{
				var w = (TWorker)worker.Clone();
				w.Set(partition[i], partition[i + 1] - 1);
				workers[i] = w;
				await w.RunAsync().ConfigureAwait(false);
			});

		return workers;
	}

	public static async Task ExecuteAsync(IEnumerable<RunnableTask> tasks, int maxDegreeOfParallelism = -1, CancellationToken cancellationToken = default)
	{
		if (maxDegreeOfParallelism <= 0)
			maxDegreeOfParallelism = Environment.ProcessorCount;

		await Parallel.ForEachAsync(tasks,
			new ParallelOptions
			{
				MaxDegreeOfParallelism = maxDegreeOfParallelism,
				CancellationToken = cancellationToken
			},
			async (task, ct) =>
			{
				await task.RunAsync().ConfigureAwait(false);
			});
	}

	public static IEnumerable<Range> PartitionEnumerable(int listSize, int nChunks)
	{
		nChunks = Math.Min(listSize, nChunks);
		var chunkSize = listSize / nChunks;
		var mod = listSize % nChunks;
		var current = 0;

		for (var i = 0; i < nChunks; i++)
		{
			var size = chunkSize + (i < mod ? 1 : 0);
			var end = current + size;
			yield return new Range(current, end - 1);
			current = end;
		}
	}

	public static int[] Partition(int listSize, int nChunks)
	{
		nChunks = Math.Min(listSize, nChunks);
		var chunkSize = listSize / nChunks;
		var mod = listSize % nChunks;
		var partition = new int[nChunks + 1];
		partition[0] = 0;
		for (var i = 1; i <= nChunks; i++)
		{
			partition[i] = partition[i - 1] + chunkSize + (i <= mod ? 1 : 0);
		}
		return partition;
	}
}

public class FeatureHistogram
{
	public class Config
	{
		public int featureIdx = -1;
		public int thresholdIdx = -1;
		public double S = -1;
		public double errReduced = -1;
	}

	// Parameter
	public static float samplingRate = 1;

	// Variables
	public float[] accumFeatureImpact = null;
	private int[] _features = null;
	private float[][] _thresholds = null;
	private double[][] _sum = null;
	private double _sumResponse = 0;
	private double _sqSumResponse = 0;
	private int[][] _count = null;
	private int[][] _sampleToThresholdMap = null;
	private double[] _impacts;

	// Reuse of parent resources
	private bool reuseParent = false;

	public async Task Construct(DataPoint[] samples, double[] labels, int[][] sampleSortedIdx, int[] features, float[][] thresholds, double[] impacts)
	{
		_features = features;
		_thresholds = thresholds;
		_impacts = impacts;
		_sumResponse = 0;
		_sqSumResponse = 0;
		_sum = new double[features.Length][];
		_count = new int[features.Length][];
		_sampleToThresholdMap = new int[features.Length][];

		var threadPool = MyThreadPool.Instance;
		if (threadPool.Size() == 1)
		{
			Construct(samples, labels, sampleSortedIdx, thresholds, 0, features.Length - 1);
		}
		else
		{
			await ParallelExecutor.ExecuteAsync(
				new Worker(this, samples, labels, sampleSortedIdx, thresholds),
				features.Length,
				threadPool.Size());
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

	protected internal async Task Update(double[] labels)
	{
		_sumResponse = 0;
		_sqSumResponse = 0;

		var threadPool = MyThreadPool.Instance;
		if (threadPool.Size() == 1)
		{
			Update(labels, 0, _features.Length - 1);
		}
		else
		{
			await ParallelExecutor.ExecuteAsync(
				new Worker(this, labels),
				_features.Length,
				threadPool.Size());
		}
	}

	protected void Update(double[] labels, int start, int end)
	{
		for (var f = start; f <= end; f++)
		{
			Array.Fill(_sum[f], 0);
		}
		for (var k = 0; k < labels.Length; k++)
		{
			for (var f = start; f <= end; f++)
			{
				var t = _sampleToThresholdMap[f][k];
				_sum[f][t] += labels[k];
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
			}
		}
	}

	public async Task Construct(FeatureHistogram parent, int[] soi, double[] labels)
	{
		_features = parent._features;
		_thresholds = parent._thresholds;
		_impacts = parent._impacts;
		_sumResponse = 0;
		_sqSumResponse = 0;
		_sum = new double[_features.Length][];
		_count = new int[_features.Length][];
		_sampleToThresholdMap = parent._sampleToThresholdMap;

		var threadPool = MyThreadPool.Instance;
		if (threadPool.Size() == 1)
		{
			Construct(parent, soi, labels, 0, _features.Length - 1);
		}
		else
		{
			await ParallelExecutor.ExecuteAsync(
				new Worker(this, parent, soi, labels),
				_features.Length,
				threadPool.Size());
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

		foreach (var k in soi)
		{
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

	public async Task Construct(FeatureHistogram parent, FeatureHistogram leftSibling, bool reuseParent)
	{
		this.reuseParent = reuseParent;
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

		var p = MyThreadPool.Instance;
		if (p.Size() == 1)
		{
			Construct(parent, leftSibling, 0, _features.Length - 1);
		}
		else
		{
			await ParallelExecutor.ExecuteAsync(
				new Worker(this, parent, leftSibling),
				_features.Length,
				p.Size());
		}
	}

	protected void Construct(FeatureHistogram parent, FeatureHistogram leftSibling, int start, int end)
	{
		for (var f = start; f <= end; f++)
		{
			var threshold = _thresholds[f];
			if (!reuseParent)
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

	public Config FindBestSplit(int[] usedFeatures, int minLeafSupport, int start, int end)
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
				{
					continue;
				}

				var sumLeft = _sum[i][t];
				var sumRight = _sumResponse - sumLeft;

				var S = sumLeft * sumLeft / countLeft + sumRight * sumRight / countRight;
				var errST = (_sqSumResponse / totalCount) * (S / totalCount);
				if (cfg.S < S)
				{
					cfg.S = S;
					cfg.featureIdx = i;
					cfg.thresholdIdx = t;
					cfg.errReduced = errST;
				}
			}
		}
		return cfg;
	}

	public async Task<bool> FindBestSplit(Split sp, double[] labels, int minLeafSupport)
	{
		if (sp.GetDeviance() >= 0.0 && sp.GetDeviance() <= 0.0)
		{
			return false; // No need to split
		}

		int[] usedFeatures;
		if (samplingRate < 1) // Subsampling (feature sampling)
		{
			var size = (int)(samplingRate * _features.Length);
			usedFeatures = new int[size];
			var featurePool = new List<int>();
			for (var i = 0; i < _features.Length; i++)
			{
				featurePool.Add(i);
			}

			var random = new Random();
			for (var i = 0; i < size; i++)
			{
				var selected = random.Next(featurePool.Count);
				usedFeatures[i] = featurePool[selected];
				featurePool.RemoveAt(selected);
			}
		}
		else // No subsampling, use all features
		{
			usedFeatures = Enumerable.Range(0, _features.Length).ToArray();
		}

		var best = new Config();
		var threadPool = MyThreadPool.Instance;
		if (threadPool.Size() == 1)
		{
			best = FindBestSplit(usedFeatures, minLeafSupport, 0, usedFeatures.Length - 1);
		}
		else
		{
			var workers = await ParallelExecutor.ExecuteAsync(new Worker(this, usedFeatures, minLeafSupport), usedFeatures.Length,
				threadPool.Size());
			foreach (var worker in workers)
			{
				if (best.S < worker.cfg.S)
					best = worker.cfg;
			}
		}

		if (best.S == -1)
		{
			return false;
		}

		// bestFeaturesHist is the best features
		var bestFeaturesHist = _sum[best.featureIdx];
		var sampleCount = _count[best.featureIdx];

		var s = bestFeaturesHist[bestFeaturesHist.Length - 1];
		var c = sampleCount[bestFeaturesHist.Length - 1];

		var sumLeft = bestFeaturesHist[best.thresholdIdx];
		var countLeft = sampleCount[best.thresholdIdx];

		var sumRight = s - sumLeft;
		var countRight = c - countLeft;

		var left = new int[countLeft];
		var right = new int[countRight];
		var l = 0;
		var r = 0;
		var k = 0;
		var idx = sp.GetSamples();
		foreach (var element in idx)
		{
			k = element;
			if (_sampleToThresholdMap[best.featureIdx][k] <= best.thresholdIdx)
			{
				left[l++] = k;
			}
			else
			{
				right[r++] = k;
			}
		}

		var lh = new FeatureHistogram();
		await lh.Construct(sp.Histogram, left, labels);
		var rh = new FeatureHistogram();
		await rh.Construct(sp.Histogram, lh, !sp.Root);

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
		private FeatureHistogram fh;
		private int type;
		private int[] usedFeatures;
		private int minLeafSup;
		internal Config cfg;
		private double[] labels;
		private FeatureHistogram parent;
		private int[] soi;
		private FeatureHistogram leftSibling;
		private DataPoint[] samples;
		private int[][] sampleSortedIdx;
		private float[][] thresholds;

		public Worker() { }

		public Worker(FeatureHistogram fh, int[] usedFeatures, int minLeafSup)
		{
			type = 0;
			this.fh = fh;
			this.usedFeatures = usedFeatures;
			this.minLeafSup = minLeafSup;
		}

		public Worker(FeatureHistogram fh, double[] labels)
		{
			type = 1;
			this.fh = fh;
			this.labels = labels;
		}

		public Worker(FeatureHistogram fh, FeatureHistogram parent, int[] soi, double[] labels)
		{
			type = 2;
			this.fh = fh;
			this.parent = parent;
			this.soi = soi;
			this.labels = labels;
		}

		public Worker(FeatureHistogram fh, FeatureHistogram parent, FeatureHistogram leftSibling)
		{
			type = 3;
			this.fh = fh;
			this.parent = parent;
			this.leftSibling = leftSibling;
		}

		public Worker(FeatureHistogram fh, DataPoint[] samples, double[] labels, int[][] sampleSortedIdx, float[][] thresholds)
		{
			type = 4;
			this.fh = fh;
			this.samples = samples;
			this.labels = labels;
			this.sampleSortedIdx = sampleSortedIdx;
			this.thresholds = thresholds;
		}

		public override void Run()
		{
			switch (type)
			{
				case 0:
					cfg = fh.FindBestSplit(usedFeatures, minLeafSup, start, end);
					break;
				case 1:
					fh.Update(labels, start, end);
					break;
				case 2:
					fh.Construct(parent, soi, labels, start, end);
					break;
				case 3:
					fh.Construct(parent, leftSibling, start, end);
					break;
				case 4:
					fh.Construct(samples, labels, sampleSortedIdx, thresholds, start, end);
					break;
			}
		}

		public override Task RunAsync() =>
			Task.Run(() =>
			{
				switch (type)
				{
					case 0:
						cfg = fh.FindBestSplit(usedFeatures, minLeafSup, start, end);
						break;
					case 1:
						fh.Update(labels, start, end);
						break;
					case 2:
						fh.Construct(parent, soi, labels, start, end);
						break;
					case 3:
						fh.Construct(parent, leftSibling, start, end);
						break;
					case 4:
						fh.Construct(samples, labels, sampleSortedIdx, thresholds, start, end);
						break;
				}
			});

		public override WorkerThread Clone()
		{
			var wk = new Worker
			{
				fh = fh,
				type = type,

				// find best split
				usedFeatures = usedFeatures,
				minLeafSup = minLeafSup,

				// update
				labels = labels,

				// construct
				parent = parent,
				soi = soi,

				// construct sibling
				leftSibling = leftSibling,

				// construct full
				samples = samples,
				sampleSortedIdx = sampleSortedIdx,
				thresholds = thresholds
			};

			return wk;
		}
	}
}
