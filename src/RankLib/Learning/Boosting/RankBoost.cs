using System.Globalization;
using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Learning.Tree;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning.Boosting;

/// <summary>
/// Parameters for <see cref="RankBoost"/> ranker.
/// </summary>
public class RankBoostParameters : IRankerParameters
{
	/// <summary>
	/// Default number of iterations (rounds).
	/// </summary>
	public const int DefaultIterationCount = 300;

	/// <summary>
	/// Default number of threshold candidates.
	/// </summary>
	public const int DefaultThreshold = 10;

	/// <summary>
	/// Number of iterations (rounds).
	/// </summary>
	public int IterationCount { get; set; } = DefaultIterationCount;

	/// <summary>
	/// Number of threshold candidates
	/// </summary>
	public int Threshold { get; set; } = DefaultThreshold;


	public override string ToString()
	{
		var builder = new StringBuilder();
		builder.AppendLine($"No. of rounds: {IterationCount}");
		builder.AppendLine($"No. of threshold candidates: {Threshold}");
		return builder.ToString();
	}
}

/// <summary>
/// RankBoost is an ensemble learning algorithm. It is an adaptation of the AdaBoost algorithm for the ranking domain.
/// RankBoost is particularly useful when the goal is to combine the outputs of weak rankers
/// (simpler models or features) to produce a stronger, more accurate ranking model.
/// </summary>
/// <remarks>
/// <a href="https://www.jmlr.org/papers/volume4/freund03a/freund03a.pdf">
/// Y. Freund, R. Iyer, R. Schapire, and Y. Singer. An efficient boosting algorithm for combining preferences.
/// The Journal of Machine Learning Research, 4: 933-969, 2003.
/// </a>
/// </remarks>
public class RankBoost : Ranker<RankBoostParameters>
{
	internal const string RankerName = "RankBoost";
	private readonly ILogger<RankBoost> _logger;

	private double[][][] _sweight = []; // Sample weight D(x_0, x_1) -- the weight of x_1 ranked above x_2
	private double[][] _potential = []; // pi(x)
	private readonly List<List<int[]>> _sortedSamples = [];
	private double[][] _thresholds = []; // Candidate values for weak rankers' threshold, selected from feature values
	private int[][] _tSortedIdx = []; // Sorted (descend) index for @thresholds

	private List<RankBoostWeakRanker> _wRankers = []; // Best weak rankers at each round
	private List<double> _rWeight = []; // Alpha (weak rankers' weight)

	// To store the best model on validation data (if specified)
	private readonly List<RankBoostWeakRanker> _bestModelRankers = [];
	private readonly List<double> _bestModelWeights = [];

	private double _rT;
	private double _zT = 1.0;
	private int _totalCorrectPairs; // Crucial pairs

	public override string Name => RankerName;

	public RankBoost(ILogger<RankBoost>? logger = null) : base() =>
		_logger = logger ?? NullLogger<RankBoost>.Instance;

	public RankBoost(List<RankList> samples, int[] features, MetricScorer scorer, ILogger<RankBoost>? logger = null)
		: base(samples, features, scorer) =>
		_logger = logger ?? NullLogger<RankBoost>.Instance;

	private int[] Reorder(RankList rankList, int fid)
	{
		var score = new double[rankList.Count];
		for (var i = 0; i < rankList.Count; i++)
			score[i] = rankList[i].GetFeatureValue(fid);

		return MergeSorter.Sort(score, false);
	}

	private void UpdatePotential()
	{
		for (var i = 0; i < Samples.Count; i++)
		{
			var rankList = Samples[i];
			for (var j = 0; j < rankList.Count; j++)
			{
				var p = 0.0;
				for (var k = j + 1; k < rankList.Count; k++)
					p += _sweight[i][j][k];

				for (var k = 0; k < j; k++)
					p -= _sweight[i][k][j];

				_potential[i][j] = p;
			}
		}
	}

	private RankBoostWeakRanker? LearnWeakRanker()
	{
		var bestFid = -1;
		double maxR = -10;
		var bestThreshold = -1.0;

		for (var i = 0; i < Features.Length; i++)
		{
			var sSortedIndex = _sortedSamples[i]; // Samples sorted (descending) by the current feature
			var idx = _tSortedIdx[i]; // Candidate thresholds for the current features
			var last = new int[Samples.Count]; // The last "touched" (and taken) position in each sample rank list

			for (var j = 0; j < Samples.Count; j++)
				last[j] = -1;

			var r = 0.0;
			foreach (var element in idx)
			{
				var t = _thresholds[i][element];
				for (var k = 0; k < Samples.Count; k++)
				{
					var rl = Samples[k];
					var sk = sSortedIndex[k];

					for (var l = last[k] + 1; l < rl.Count; l++)
					{
						var k1 = sk[l];
						var p = rl[k1];
						if (p.GetFeatureValue(Features[i]) > t) // Take it
						{
							r += _potential[k][sk[l]];
							last[k] = l;
						}
						else
							break;
					}
				}

				// Finish computing r
				if (r > maxR)
				{
					maxR = r;
					bestThreshold = t;
					bestFid = Features[i];
				}
			}
		}

		if (bestFid == -1)
			return null;

		_rT = _zT * maxR; // Save it so we won't have to re-compute when we need it
		return new RankBoostWeakRanker(bestFid, bestThreshold);
	}

	private void UpdateSampleWeights(double alphaT)
	{
		// Normalize sample weights after updating them
		_zT = 0.0; // Normalization factor

		for (var i = 0; i < Samples.Count; i++)
		{
			var rl = Samples[i];
			var dT = new double[rl.Count][];

			for (var j = 0; j < rl.Count - 1; j++)
			{
				dT[j] = new double[rl.Count];

				for (var k = j + 1; k < rl.Count; k++)
				{
					dT[j][k] = _sweight[i][j][k] * Math.Exp(alphaT * (_wRankers.Last().Score(rl[k]) - _wRankers.Last().Score(rl[j])));
					_zT += dT[j][k]; // Sum the new weight for normalization
				}
			}
			_sweight[i] = dT;
		}

		// Normalize the weights to make sure it's a valid distribution
		for (var i = 0; i < Samples.Count; i++)
		{
			var rl = Samples[i];
			for (var j = 0; j < rl.Count - 1; j++)
			{
				for (var k = j + 1; k < rl.Count; k++)
					_sweight[i][j][k] /= _zT; // Normalize by Z_t
			}
		}
	}

	public override Task InitAsync()
	{
		_logger.LogInformation("Initializing...");

		_wRankers = [];
		_rWeight = [];
		_totalCorrectPairs = 0;
		for (var i = 0; i < Samples.Count; i++)
		{
			Samples[i] = Samples[i].GetCorrectRanking(); // Ensure training samples are correctly ranked
			var rl = Samples[i];
			for (var j = 0; j < rl.Count - 1; j++)
			{
				for (var k = rl.Count - 1; k >= j + 1 && rl[j].Label > rl[k].Label; k--)
					_totalCorrectPairs++;
			}
		}

		_sweight = new double[Samples.Count][][];
		for (var i = 0; i < Samples.Count; i++)
		{
			var rl = Samples[i];
			_sweight[i] = new double[rl.Count][];
			for (var j = 0; j < rl.Count - 1; j++)
			{
				_sweight[i][j] = new double[rl.Count];
				for (var k = j + 1; k < rl.Count; k++)
					_sweight[i][j][k] = rl[j].Label > rl[k].Label ? 1.0 / _totalCorrectPairs : 0.0;
			}
		}

		_potential = new double[Samples.Count][];
		for (var i = 0; i < Samples.Count; i++)
			_potential[i] = new double[Samples[i].Count];

		if (Parameters.Threshold <= 0)
		{
			var count = 0;
			for (var i = 0; i < Samples.Count; i++)
				count += Samples[i].Count;

			_thresholds = new double[Features.Length][];
			for (var i = 0; i < Features.Length; i++)
				_thresholds[i] = new double[count];

			var c = 0;
			for (var i = 0; i < Samples.Count; i++)
			{
				var rl = Samples[i];
				for (var j = 0; j < rl.Count; j++)
				{
					for (var k = 0; k < Features.Length; k++)
						_thresholds[k][c] = rl[j].GetFeatureValue(Features[k]);

					c++;
				}
			}
		}
		else
		{
			var fmax = new double[Features.Length];
			var fmin = new double[Features.Length];
			for (var i = 0; i < Features.Length; i++)
			{
				fmax[i] = -1E6;
				fmin[i] = 1E6;
			}

			for (var i = 0; i < Samples.Count; i++)
			{
				var rl = Samples[i];
				for (var j = 0; j < rl.Count; j++)
				{
					for (var k = 0; k < Features.Length; k++)
					{
						double f = rl[j].GetFeatureValue(Features[k]);
						if (f > fmax[k])
							fmax[k] = f;
						if (f < fmin[k])
							fmin[k] = f;
					}
				}
			}

			_thresholds = new double[Features.Length][];
			for (var i = 0; i < Features.Length; i++)
			{
				var step = (Math.Abs(fmax[i] - fmin[i])) / Parameters.Threshold;
				_thresholds[i] = new double[Parameters.Threshold + 1];
				_thresholds[i][0] = fmax[i];
				for (var j = 1; j < Parameters.Threshold; j++)
					_thresholds[i][j] = _thresholds[i][j - 1] - step;

				_thresholds[i][Parameters.Threshold] = fmin[i] - 1.0E8;
			}
		}

		_tSortedIdx = new int[Features.Length][];
		for (var i = 0; i < Features.Length; i++)
			_tSortedIdx[i] = MergeSorter.Sort(_thresholds[i], false);

		for (var i = 0; i < Features.Length; i++)
		{
			var idx = new List<int[]>();
			for (var j = 0; j < Samples.Count; j++)
				idx.Add(Reorder(Samples[j], Features[i]));

			_sortedSamples.Add(idx);
		}

		return Task.CompletedTask;
	}

	public override Task LearnAsync()
	{
		_logger.LogInformation("Training starts...");
		_logger.PrintLog([7, 8, 9, 9, 9, 9], ["#iter",
			"Sel. F.",
			"Threshold",
			"Error",
			Scorer.Name + "-T",
			Scorer.Name + "-V"
		]);

		var bufferedLogger = new BufferedLogger(_logger, new StringBuilder());

		for (var t = 1; t <= Parameters.IterationCount; t++)
		{
			UpdatePotential();
			var wr = LearnWeakRanker();
			if (wr == null)
				break;

			var alphaT = 0.5 * SimpleMath.Ln((_zT + _rT) / (_zT - _rT));

			_wRankers.Add(wr);
			_rWeight.Add(alphaT);

			UpdateSampleWeights(alphaT);

			bufferedLogger.PrintLog([7, 8, 9, 9], [
				t.ToString(),
				wr.Fid.ToString(),
				SimpleMath.Round(wr.Threshold, 4).ToString(CultureInfo.InvariantCulture),
				SimpleMath.Round(_rT, 4).ToString(CultureInfo.InvariantCulture)
			]);

			if (t % 1 == 0)
			{
				bufferedLogger.PrintLog([9], [SimpleMath.Round(Scorer.Score(Rank(Samples)), 4).ToString(CultureInfo.InvariantCulture)]);
				if (ValidationSamples != null)
				{
					var score = Scorer.Score(Rank(ValidationSamples));
					if (score > ValidationDataScore)
					{
						ValidationDataScore = score;
						_bestModelRankers.Clear();
						_bestModelRankers.AddRange(_wRankers);
						_bestModelWeights.Clear();
						_bestModelWeights.AddRange(_rWeight);
					}
					bufferedLogger.PrintLog([9], [SimpleMath.Round(score, 4).ToString(CultureInfo.InvariantCulture)]);
				}
			}
			bufferedLogger.FlushLog();
		}

		if (ValidationSamples != null && _bestModelRankers.Count > 0)
		{
			_wRankers.Clear();
			_rWeight.Clear();
			_wRankers.AddRange(_bestModelRankers);
			_rWeight.AddRange(_bestModelWeights);
		}

		TrainingDataScore = SimpleMath.Round(Scorer.Score(Rank(Samples)), 4);
		_logger.LogInformation("Finished successfully.");
		_logger.LogInformation("{ScorerName} on training data: {Score}", Scorer.Name, TrainingDataScore);

		if (ValidationSamples != null)
		{
			ValidationDataScore = Scorer.Score(Rank(ValidationSamples));
			_logger.LogInformation("{ScorerName} on validation data: {Score}", Scorer.Name, SimpleMath.Round(ValidationDataScore, 4));
		}

		return Task.CompletedTask;
	}

	public override double Eval(DataPoint dataPoint)
	{
		var score = 0.0;
		for (var j = 0; j < _wRankers.Count; j++)
			score += _rWeight[j] * _wRankers[j].Score(dataPoint);

		return score;
	}

	public override string GetModel()
	{
		var output = new StringBuilder();
		output.Append($"## {Name}\n");
		output.Append($"## Iteration = {Parameters.IterationCount}\n");
		output.Append($"## No. of threshold candidates = {Parameters.Threshold}\n");

		for (var i = 0; i < _wRankers.Count; i++)
			output.Append($"{_wRankers[i]}:{_rWeight[i]}{(i == _rWeight.Count - 1 ? "" : " ")}");

		return output.ToString();
	}

	public override void LoadFromString(string model)
	{
		try
		{
			using var inReader = new StringReader(model);
			string? content;
			while ((content = inReader.ReadLine()) != null)
			{
				content = content.Trim();
				if (content.Length == 0 || content.StartsWith("##"))
					continue;

				break;
			}

			if (content == null)
				throw RankLibException.Create("Model name is not found.");

			_rWeight = new List<double>();
			_wRankers = new List<RankBoostWeakRanker>();

			var idx = content.LastIndexOf('#');
			if (idx != -1)
			{
				content = content.Substring(0, idx).Trim();
			}

			var fs = content.Split(" ");
			foreach (var item in fs)
			{
				if (string.IsNullOrWhiteSpace(item))
					continue;

				var strs = item.Split(":");
				var fid = int.Parse(strs[0]);
				var threshold = double.Parse(strs[1]);
				var weight = double.Parse(strs[2]);
				_rWeight.Add(weight);
				_wRankers.Add(new RankBoostWeakRanker(fid, threshold));
			}

			Features = new int[_rWeight.Count];
			for (var i = 0; i < _rWeight.Count; i++)
				Features[i] = _wRankers[i].Fid;
		}
		catch (Exception ex)
		{
			throw RankLibException.Create("Error loading model", ex);
		}
	}
}
