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

	// Parameter
	public static float samplingRate = 1;

	// Variables
	public float[] accumFeatureImpact = null;
	public int[] features = null;
	public float[][] thresholds = null;
	public double[][] sum = null;
	public double sumResponse = 0;
	public double sqSumResponse = 0;
	public int[][] count = null;
	public int[][] sampleToThresholdMap = null;
	public double[] impacts;

	// Reuse of parent resources
	private bool reuseParent = false;

	public void Construct(DataPoint[] samples, double[] labels, int[][] sampleSortedIdx, int[] features, float[][] thresholds, double[] impacts)
	{
		this.features = features;
		this.thresholds = thresholds;
		this.impacts = impacts;

		sumResponse = 0;
		sqSumResponse = 0;

		sum = new double[features.Length][];
		count = new int[features.Length][];
		sampleToThresholdMap = new int[features.Length][];

		var threadPool = MyThreadPool.Instance;
		if (threadPool.Size() == 1)
		{
			Construct(samples, labels, sampleSortedIdx, thresholds, 0, features.Length - 1);
		}
		else
		{
			threadPool.Execute(new Worker(this, samples, labels, sampleSortedIdx, thresholds), features.Length);
		}
	}

	protected void Construct(DataPoint[] samples, double[] labels, int[][] sampleSortedIdx, float[][] thresholds, int start, int end)
	{
		for (var i = start; i <= end; i++)
		{
			var fid = features[i];
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
						sumResponse += labels[k];
						sqSumResponse += labels[k] * labels[k];
					}
					stMap[k] = t;
				}
				last = j - 1;
				sumLabel[t] = sumLeft;
				c[t] = last + 1;
			}
			sampleToThresholdMap[i] = stMap;
			sum[i] = sumLabel;
			count[i] = c;
		}
	}

	protected internal void Update(double[] labels)
	{
		sumResponse = 0;
		sqSumResponse = 0;

		var threadPool = MyThreadPool.Instance;
		if (threadPool.Size() == 1)
		{
			Update(labels, 0, features.Length - 1);
		}
		else
		{
			threadPool.Execute(new Worker(this, labels), features.Length);
		}
	}

	protected void Update(double[] labels, int start, int end)
	{
		for (var f = start; f <= end; f++)
		{
			Array.Fill(sum[f], 0);
		}
		for (var k = 0; k < labels.Length; k++)
		{
			for (var f = start; f <= end; f++)
			{
				var t = sampleToThresholdMap[f][k];
				sum[f][t] += labels[k];
				if (f == 0)
				{
					sumResponse += labels[k];
					sqSumResponse += labels[k] * labels[k];
				}
			}
		}
		for (var f = start; f <= end; f++)
		{
			for (var t = 1; t < thresholds[f].Length; t++)
			{
				sum[f][t] += sum[f][t - 1];
			}
		}
	}

	public void Construct(FeatureHistogram parent, int[] soi, double[] labels)
	{
		features = parent.features;
		thresholds = parent.thresholds;
		impacts = parent.impacts;
		sumResponse = 0;
		sqSumResponse = 0;
		sum = new double[features.Length][];
		count = new int[features.Length][];
		sampleToThresholdMap = parent.sampleToThresholdMap;

		var threadPool = MyThreadPool.Instance;
		if (threadPool.Size() == 1)
		{
			Construct(parent, soi, labels, 0, features.Length - 1);
		}
		else
		{
			threadPool.Execute(new Worker(this, parent, soi, labels), features.Length);
		}
	}

	protected void Construct(FeatureHistogram parent, int[] soi, double[] labels, int start, int end)
	{
		for (var i = start; i <= end; i++)
		{
			var threshold = thresholds[i];
			sum[i] = new double[threshold.Length];
			count[i] = new int[threshold.Length];
			Array.Fill(sum[i], 0);
			Array.Fill(count[i], 0);
		}

		foreach (var k in soi)
		{
			for (var f = start; f <= end; f++)
			{
				var t = sampleToThresholdMap[f][k];
				sum[f][t] += labels[k];
				count[f][t]++;
				if (f == 0)
				{
					sumResponse += labels[k];
					sqSumResponse += labels[k] * labels[k];
				}
			}
		}

		for (var f = start; f <= end; f++)
		{
			for (var t = 1; t < thresholds[f].Length; t++)
			{
				sum[f][t] += sum[f][t - 1];
				count[f][t] += count[f][t - 1];
			}
		}
	}

	public void Construct(FeatureHistogram parent, FeatureHistogram leftSibling, bool reuseParent)
	{
		this.reuseParent = reuseParent;
		features = parent.features;
		thresholds = parent.thresholds;
		impacts = parent.impacts;
		sumResponse = parent.sumResponse - leftSibling.sumResponse;
		sqSumResponse = parent.sqSumResponse - leftSibling.sqSumResponse;

		if (reuseParent)
		{
			sum = parent.sum;
			count = parent.count;
		}
		else
		{
			sum = new double[features.Length][];
			count = new int[features.Length][];
		}
		sampleToThresholdMap = parent.sampleToThresholdMap;

		var p = MyThreadPool.Instance;
		if (p.Size() == 1)
		{
			Construct(parent, leftSibling, 0, features.Length - 1);
		}
		else
		{
			p.Execute(new Worker(this, parent, leftSibling), features.Length);
		}
	}

	protected void Construct(FeatureHistogram parent, FeatureHistogram leftSibling, int start, int end)
	{
		for (var f = start; f <= end; f++)
		{
			var threshold = thresholds[f];
			if (!reuseParent)
			{
				sum[f] = new double[threshold.Length];
				count[f] = new int[threshold.Length];
			}
			for (var t = 0; t < threshold.Length; t++)
			{
				sum[f][t] = parent.sum[f][t] - leftSibling.sum[f][t];
				count[f][t] = parent.count[f][t] - leftSibling.count[f][t];
			}
		}
	}

	public Config FindBestSplit(int[] usedFeatures, int minLeafSupport, int start, int end)
	{
		var cfg = new Config();
		var totalCount = count[start][count[start].Length - 1];
		for (var f = start; f <= end; f++)
		{
			var i = usedFeatures[f];
			var threshold = thresholds[i];

			for (var t = 0; t < threshold.Length; t++)
			{
				var countLeft = count[i][t];
				var countRight = totalCount - countLeft;
				if (countLeft < minLeafSupport || countRight < minLeafSupport)
				{
					continue;
				}

				var sumLeft = sum[i][t];
				var sumRight = sumResponse - sumLeft;

				var S = sumLeft * sumLeft / countLeft + sumRight * sumRight / countRight;
				var errST = (sqSumResponse / totalCount) * (S / totalCount);
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

	public bool FindBestSplit(Split sp, double[] labels, int minLeafSupport)
	{
		if (sp.GetDeviance() >= 0.0 && sp.GetDeviance() <= 0.0)
		{
			return false; // No need to split
		}

		int[] usedFeatures;
		if (samplingRate < 1) // Subsampling (feature sampling)
		{
			var size = (int)(samplingRate * features.Length);
			usedFeatures = new int[size];
			var featurePool = new List<int>();
			for (var i = 0; i < features.Length; i++)
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
			usedFeatures = Enumerable.Range(0, features.Length).ToArray();
		}

		var best = new Config();
		var threadPool = MyThreadPool.Instance;
		if (threadPool.Size() == 1)
		{
			best = FindBestSplit(usedFeatures, minLeafSupport, 0, usedFeatures.Length - 1);
		}
		else
		{
			var workers = threadPool.Execute(new Worker(this, usedFeatures, minLeafSupport), usedFeatures.Length);
			foreach (var worker in workers)
			{
				var wk = (Worker)worker;
				if (best.S < wk.cfg.S)
				{
					best = wk.cfg;
				}
			}
		}

		if (best.S == -1)
		{
			return false;
		}

		// bestFeaturesHist is the best features
		var bestFeaturesHist = sum[best.featureIdx];
		var sampleCount = count[best.featureIdx];

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
			if (sampleToThresholdMap[best.featureIdx][k] <= best.thresholdIdx)
			{
				left[l++] = k;
			}
			else
			{
				right[r++] = k;
			}
		}

		var lh = new FeatureHistogram();
		lh.Construct(sp.hist, left, labels);
		var rh = new FeatureHistogram();
		rh.Construct(sp.hist, lh, !sp.Root);

		var var = sqSumResponse - sumResponse * sumResponse / idx.Length;
		var varLeft = lh.sqSumResponse - lh.sumResponse * lh.sumResponse / left.Length;
		var varRight = rh.sqSumResponse - rh.sumResponse * rh.sumResponse / right.Length;

		sp.Set(features[best.featureIdx], thresholds[best.featureIdx][best.thresholdIdx], var);
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
