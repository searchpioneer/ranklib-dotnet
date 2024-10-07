using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Learning.Tree;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning.Boosting;

public class RankBoost : Ranker
{
	private readonly ILogger<RankBoost> _logger;

	public static int NIteration = 300; // Number of rounds
	public static int NThreshold = 10;

	protected double[][][] _sweight = null; // Sample weight D(x_0, x_1) -- the weight of x_1 ranked above x_2
	protected double[][] _potential = null; // pi(x)
	protected List<List<int[]>> _sortedSamples = new List<List<int[]>>();
	protected double[][] _thresholds = null; // Candidate values for weak rankers' threshold, selected from feature values
	protected int[][] _tSortedIdx = null; // Sorted (descend) index for @thresholds

	protected List<RBWeakRanker> _wRankers = null; // Best weak rankers at each round
	protected List<double> _rWeight = null; // Alpha (weak rankers' weight)

	// To store the best model on validation data (if specified)
	protected List<RBWeakRanker> _bestModelRankers = new List<RBWeakRanker>();
	protected List<double> _bestModelWeights = new List<double>();

	private double _R_t = 0.0;
	private double _Z_t = 1.0;
	private int _totalCorrectPairs = 0; // Crucial pairs

	public RankBoost(ILogger<RankBoost>? logger = null) : base(logger) =>
		_logger = logger ?? NullLogger<RankBoost>.Instance;

	public RankBoost(List<RankList> samples, int[] features, MetricScorer scorer, ILogger<RankBoost>? logger = null)
		: base(samples, features, scorer, logger) =>
		_logger = logger ?? NullLogger<RankBoost>.Instance;

	private int[] Reorder(RankList rl, int fid)
	{
		var score = new double[rl.Count];
		for (var i = 0; i < rl.Count; i++)
		{
			score[i] = rl[i].GetFeatureValue(fid);
		}
		return MergeSorter.Sort(score, false);
	}

	private void UpdatePotential()
	{
		for (var i = 0; i < Samples.Count; i++)
		{
			var rl = Samples[i];
			for (var j = 0; j < rl.Count; j++)
			{
				var p = 0.0;
				for (var k = j + 1; k < rl.Count; k++)
				{
					p += _sweight[i][j][k];
				}
				for (var k = 0; k < j; k++)
				{
					p -= _sweight[i][k][j];
				}
				_potential[i][j] = p;
			}
		}
	}

	private RBWeakRanker LearnWeakRanker()
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
			{
				last[j] = -1;
			}

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
						{
							break;
						}
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

		_R_t = _Z_t * maxR; // Save it so we won't have to re-compute when we need it
		return new RBWeakRanker(bestFid, bestThreshold);
	}

	private void UpdateSampleWeights(double alpha_t)
	{
		// Normalize sample weights after updating them
		_Z_t = 0.0; // Normalization factor

		for (var i = 0; i < Samples.Count; i++)
		{
			var rl = Samples[i];
			var D_t = new double[rl.Count][];

			for (var j = 0; j < rl.Count - 1; j++)
			{
				D_t[j] = new double[rl.Count];

				for (var k = j + 1; k < rl.Count; k++)
				{
					D_t[j][k] = _sweight[i][j][k] * Math.Exp(alpha_t * (_wRankers.Last().Score(rl[k]) - _wRankers.Last().Score(rl[j])));
					_Z_t += D_t[j][k]; // Sum the new weight for normalization
				}
			}
			_sweight[i] = D_t;
		}

		// Normalize the weights to make sure it's a valid distribution
		for (var i = 0; i < Samples.Count; i++)
		{
			var rl = Samples[i];
			for (var j = 0; j < rl.Count - 1; j++)
			{
				for (var k = j + 1; k < rl.Count; k++)
				{
					_sweight[i][j][k] /= _Z_t; // Normalize by Z_t
				}
			}
		}
	}

	public override void Init()
	{
		_logger.LogInformation("Initializing...");

		_wRankers = new List<RBWeakRanker>();
		_rWeight = new List<double>();

		_totalCorrectPairs = 0;
		for (var i = 0; i < Samples.Count; i++)
		{
			Samples[i] = Samples[i].GetCorrectRanking(); // Ensure training samples are correctly ranked
			var rl = Samples[i];
			for (var j = 0; j < rl.Count - 1; j++)
			{
				for (var k = rl.Count - 1; k >= j + 1 && rl[j].Label > rl[k].Label; k--)
				{
					_totalCorrectPairs++;
				}
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
				{
					_sweight[i][j][k] = rl[j].Label > rl[k].Label ? 1.0 / _totalCorrectPairs : 0.0;
				}
			}
		}

		_potential = new double[Samples.Count][];
		for (var i = 0; i < Samples.Count; i++)
		{
			_potential[i] = new double[Samples[i].Count];
		}

		if (NThreshold <= 0)
		{
			var count = 0;
			for (var i = 0; i < Samples.Count; i++)
			{
				count += Samples[i].Count;
			}

			_thresholds = new double[Features.Length][];
			for (var i = 0; i < Features.Length; i++)
			{
				_thresholds[i] = new double[count];
			}

			var c = 0;
			for (var i = 0; i < Samples.Count; i++)
			{
				var rl = Samples[i];
				for (var j = 0; j < rl.Count; j++)
				{
					for (var k = 0; k < Features.Length; k++)
					{
						_thresholds[k][c] = rl[j].GetFeatureValue(Features[k]);
					}
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
				var step = (Math.Abs(fmax[i] - fmin[i])) / NThreshold;
				_thresholds[i] = new double[NThreshold + 1];
				_thresholds[i][0] = fmax[i];
				for (var j = 1; j < NThreshold; j++)
				{
					_thresholds[i][j] = _thresholds[i][j - 1] - step;
				}
				_thresholds[i][NThreshold] = fmin[i] - 1.0E8;
			}
		}

		_tSortedIdx = new int[Features.Length][];
		for (var i = 0; i < Features.Length; i++)
		{
			_tSortedIdx[i] = MergeSorter.Sort(_thresholds[i], false);
		}

		for (var i = 0; i < Features.Length; i++)
		{
			var idx = new List<int[]>();
			for (var j = 0; j < Samples.Count; j++)
			{
				idx.Add(Reorder(Samples[j], Features[i]));
			}
			_sortedSamples.Add(idx);
		}
	}

	public override void Learn()
	{
		_logger.LogInformation("Training starts...");
		PrintLogLn(new[] { 7, 8, 9, 9, 9, 9 }, new[] { "#iter", "Sel. F.", "Threshold", "Error", Scorer.Name + "-T", Scorer.Name + "-V" });

		for (var t = 1; t <= NIteration; t++)
		{
			UpdatePotential();
			var wr = LearnWeakRanker();
			if (wr == null)
				break;

			var alpha_t = 0.5 * SimpleMath.Ln((_Z_t + _R_t) / (_Z_t - _R_t));

			_wRankers.Add(wr);
			_rWeight.Add(alpha_t);

			UpdateSampleWeights(alpha_t);

			PrintLog(new[] { 7, 8, 9, 9 }, new[] { t.ToString(), wr.GetFid().ToString(), SimpleMath.Round(wr.GetThreshold(), 4).ToString(), SimpleMath.Round(_R_t, 4).ToString() });

			if (t % 1 == 0)
			{
				PrintLog(new[] { 9 }, new[] { SimpleMath.Round(Scorer.Score(Rank(Samples)), 4).ToString() });
				if (ValidationSamples != null)
				{
					var score = Scorer.Score(Rank(ValidationSamples));
					if (score > BestScoreOnValidationData)
					{
						BestScoreOnValidationData = score;
						_bestModelRankers.Clear();
						_bestModelRankers.AddRange(_wRankers);
						_bestModelWeights.Clear();
						_bestModelWeights.AddRange(_rWeight);
					}
					PrintLog(new[] { 9 }, new[] { SimpleMath.Round(score, 4).ToString() });
				}
			}
			FlushLog();
		}

		if (ValidationSamples != null && _bestModelRankers.Count > 0)
		{
			_wRankers.Clear();
			_rWeight.Clear();
			_wRankers.AddRange(_bestModelRankers);
			_rWeight.AddRange(_bestModelWeights);
		}

		ScoreOnTrainingData = SimpleMath.Round(Scorer.Score(Rank(Samples)), 4);
		_logger.LogInformation("Finished successfully.");
		_logger.LogInformation("{ScorerName} on training data: {Score}", Scorer.Name, ScoreOnTrainingData);

		if (ValidationSamples != null)
		{
			BestScoreOnValidationData = Scorer.Score(Rank(ValidationSamples));
			_logger.LogInformation("{ScorerName} on validation data: {Score}", Scorer.Name, SimpleMath.Round(BestScoreOnValidationData, 4));
		}
	}

	public override double Eval(DataPoint p)
	{
		var score = 0.0;
		for (var j = 0; j < _wRankers.Count; j++)
		{
			score += _rWeight[j] * _wRankers[j].Score(p);
		}
		return score;
	}

	public override Ranker CreateNew() => new RankBoost();

	public override string ToString()
	{
		var output = new StringBuilder();
		for (var i = 0; i < _wRankers.Count; i++)
		{
			output.Append($"{_wRankers[i]}:{_rWeight[i]}{(i == _rWeight.Count - 1 ? "" : " ")}");
		}
		return output.ToString();
	}

	public override string Model()
	{
		var output = new StringBuilder();
		output.Append($"## {Name}\n");
		output.Append($"## Iteration = {NIteration}\n");
		output.Append($"## No. of threshold candidates = {NThreshold}\n");
		output.Append(ToString());
		return output.ToString();
	}

	public override void LoadFromString(string fullText)
	{
		try
		{
			using var inReader = new StringReader(fullText);
			string? content;
			while ((content = inReader.ReadLine()) != null)
			{
				content = content.Trim();
				if (content.Length == 0 || content.StartsWith("##"))
				{
					continue;
				}
				break;
			}

			if (content == null)
				throw RankLibError.Create("Model name is not found.");

			_rWeight = new List<double>();
			_wRankers = new List<RBWeakRanker>();

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
				_wRankers.Add(new RBWeakRanker(fid, threshold));
			}

			Features = new int[_rWeight.Count];
			for (var i = 0; i < _rWeight.Count; i++)
			{
				Features[i] = _wRankers[i].GetFid();
			}
		}
		catch (Exception ex)
		{
			throw RankLibError.Create("Error in RankBoost::load(): ", ex);
		}
	}

	public override void PrintParameters()
	{
		_logger.LogInformation("No. of rounds: {Rounds}", NIteration);
		_logger.LogInformation("No. of threshold candidates: {Candidates}", NThreshold);
	}

	public override string Name => "RankBoost";
}
