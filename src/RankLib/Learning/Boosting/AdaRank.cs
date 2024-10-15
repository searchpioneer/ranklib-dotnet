using System.Globalization;
using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning.Boosting;

public class AdaRankParameters : IRankerParameters
{
	public int NIteration { get; set; } = 500;
	public double Tolerance { get; set; } = 0.002;
	public bool TrainWithEnqueue { get; set; } = true;

	/// <summary>
	/// Max number of times a feature can be selected consecutively before being removed
	/// </summary>
	public int MaxSelCount { get; set; } = 5;

	public void Log(ILogger logger)
	{
		logger.LogInformation("No. of rounds: {NIteration}", NIteration);
		logger.LogInformation("Train with 'enqueue': {TrainWithEnqueue}", TrainWithEnqueue ? "Yes" : "No");
		logger.LogInformation("Tolerance: {Tolerance}", Tolerance);
		logger.LogInformation("Max Sel. Count: {MaxSelCount}", MaxSelCount);
	}
}

public class AdaRank : Ranker<AdaRankParameters>
{
	internal const string RankerName = "AdaRank";

	private readonly ILogger<AdaRank> _logger;
	private readonly Dictionary<int, int> _usedFeatures = new();
	private double[] _sweight = []; // Sample weight
	private List<AdaRankWeakRanker> _rankers = []; // Alpha
	private List<double> _rweight = []; // Weak rankers' weight
	private List<AdaRankWeakRanker> _bestModelRankers = [];
	private List<double> _bestModelWeights = [];

	// For the implementation of tricks
	private int _lastFeature = -1;
	private int _lastFeatureConsecutiveCount;
	private bool _performanceChanged;
	private List<int> _featureQueue = [];
	private double[] _backupSampleWeight = [];
	private double _backupTrainScore;
	private double _lastTrainedScore = -1.0;

	public override string Name => RankerName;

	public AdaRank(ILogger<AdaRank>? logger = null) => _logger = logger ?? NullLogger<AdaRank>.Instance;

	public AdaRank(List<RankList> samples, int[] features, MetricScorer scorer, ILogger<AdaRank>? logger = null) : base(
		samples, features, scorer, logger) =>
		_logger = logger ?? NullLogger<AdaRank>.Instance;

	private void UpdateBestModelOnValidation()
	{
		_bestModelRankers.Clear();
		_bestModelRankers.AddRange(_rankers);
		_bestModelWeights.Clear();
		_bestModelWeights.AddRange(_rweight);
	}

	private AdaRankWeakRanker? LearnWeakRanker()
	{
		var bestScore = -1.0;
		AdaRankWeakRanker? bestWeakRanker = null;

		foreach (var i in Features)
		{
			if (_featureQueue.Contains(i) || _usedFeatures.ContainsKey(i))
				continue;

			var wr = new AdaRankWeakRanker(i);
			var s = 0.0;
			for (var j = 0; j < Samples.Count; j++)
			{
				var t = Scorer.Score(wr.Rank(Samples[j])) * _sweight[j];
				s += t;
			}

			if (bestScore < s)
			{
				bestScore = s;
				bestWeakRanker = wr;
			}
		}

		return bestWeakRanker;
	}

	private int Learn(int startIteration, bool withEnqueue)
	{
		var t = startIteration;

		for (; t <= Parameters.NIteration; t++)
		{
			PrintLog([7], [t.ToString()]);

			var bestWeakRanker = LearnWeakRanker();
			if (bestWeakRanker == null)
				break;

			if (withEnqueue)
			{
				if (bestWeakRanker.Fid == _lastFeature)
				{
					_featureQueue.Add(_lastFeature);
					_rankers.RemoveAt(_rankers.Count - 1);
					_rweight.RemoveAt(_rweight.Count - 1);
					Array.Copy(_backupSampleWeight, _sweight, _sweight.Length);
					BestScoreOnValidationData = 0.0;
					_lastTrainedScore = _backupTrainScore;
					PrintLogLn([8, 9, 9, 9], [bestWeakRanker.Fid.ToString(), "", "", "ROLLBACK"]);
					continue;
				}

				_lastFeature = bestWeakRanker.Fid;
				Array.Copy(_sweight, _backupSampleWeight, _sweight.Length);
				_backupTrainScore = _lastTrainedScore;
			}

			var num = 0.0;
			var denom = 0.0;
			for (var i = 0; i < Samples.Count; i++)
			{
				var tmp = Scorer.Score(bestWeakRanker.Rank(Samples[i]));
				num += _sweight[i] * (1.0 + tmp);
				denom += _sweight[i] * (1.0 - tmp);
			}

			_rankers.Add(bestWeakRanker);
			var alphaT = 0.5 * SimpleMath.Ln(num / denom);
			_rweight.Add(alphaT);

			var trainedScore = 0.0;
			var total = 0.0;

			foreach (var sample in Samples)
			{
				var tmp = Scorer.Score(Rank(sample));
				total += Math.Exp(-alphaT * tmp);
				trainedScore += tmp;
			}

			trainedScore /= Samples.Count;
			var delta = trainedScore + Parameters.Tolerance - _lastTrainedScore;
			var status = delta > 0 ? "OK" : "DAMN";

			if (!withEnqueue)
			{
				if (trainedScore != _lastTrainedScore)
				{
					_performanceChanged = true;
					_lastFeatureConsecutiveCount = 0;
					_usedFeatures.Clear();
				}
				else
				{
					_performanceChanged = false;
					if (_lastFeature == bestWeakRanker.Fid)
					{
						_lastFeatureConsecutiveCount++;
						if (_lastFeatureConsecutiveCount == Parameters.MaxSelCount)
						{
							status = "F. REM.";
							_lastFeatureConsecutiveCount = 0;
							_usedFeatures[_lastFeature] = 1;
						}
					}
					else
					{
						_lastFeatureConsecutiveCount = 0;
						_usedFeatures.Clear();
					}
				}

				_lastFeature = bestWeakRanker.Fid;
			}

			PrintLog([8, 9], [bestWeakRanker.Fid.ToString(), SimpleMath.Round(trainedScore, 4).ToString(CultureInfo.InvariantCulture)]);
			if (t % 1 == 0 && ValidationSamples != null)
			{
				var scoreOnValidation = Scorer.Score(Rank(ValidationSamples));
				if (scoreOnValidation > BestScoreOnValidationData)
				{
					BestScoreOnValidationData = scoreOnValidation;
					UpdateBestModelOnValidation();
				}

				PrintLog([9, 9], [SimpleMath.Round(scoreOnValidation, 4).ToString(CultureInfo.InvariantCulture), status
				]);
			}
			else
			{
				PrintLog([9, 9], ["", status]);
			}

			FlushLog();

			if (delta <= 0)
			{
				_rankers.RemoveAt(_rankers.Count - 1);
				_rweight.RemoveAt(_rweight.Count - 1);
				break;
			}

			_lastTrainedScore = trainedScore;

			for (var i = 0; i < _sweight.Length; i++)
			{
				_sweight[i] *= Math.Exp(-alphaT * Scorer.Score(Rank(Samples[i]))) / total;
			}
		}

		return t;
	}

	public override void Init()
	{
		_logger.LogInformation("Initializing...");
		_usedFeatures.Clear();

		_sweight = new double[Samples.Count];
		for (var i = 0; i < _sweight.Length; i++)
		{
			_sweight[i] = 1.0f / Samples.Count;
		}

		_backupSampleWeight = new double[_sweight.Length];
		Array.Copy(_sweight, _backupSampleWeight, _sweight.Length);

		_lastTrainedScore = -1.0;
		_rankers = new List<AdaRankWeakRanker>();
		_rweight = new List<double>();
		_featureQueue = new List<int>();
		BestScoreOnValidationData = 0.0;
		_bestModelRankers = new List<AdaRankWeakRanker>();
		_bestModelWeights = new List<double>();
	}

	public override void Learn()
	{
		_logger.LogInformation("Training starts...");
		PrintLogLn([7, 8, 9, 9, 9], ["#iter", "Sel. F.", Scorer.Name + "-T", Scorer.Name + "-V", "Status"]);

		if (Parameters.TrainWithEnqueue)
		{
			var t = Learn(1, true);
			for (var i = _featureQueue.Count - 1; i >= 0; i--)
			{
				_featureQueue.RemoveAt(i);
				t = Learn(t, false);
			}
		}
		else
		{
			Learn(1, false);
		}

		if (ValidationSamples != null && _bestModelRankers.Count > 0)
		{
			_rankers.Clear();
			_rweight.Clear();
			_rankers.AddRange(_bestModelRankers);
			_rweight.AddRange(_bestModelWeights);
		}

		ScoreOnTrainingData = SimpleMath.Round(Scorer.Score(Rank(Samples)), 4);
		_logger.LogInformation("Finished successfully.");
		_logger.LogInformation($"{Scorer.Name} on training data: {ScoreOnTrainingData}");

		if (ValidationSamples != null)
		{
			BestScoreOnValidationData = Scorer.Score(Rank(ValidationSamples));
			_logger.LogInformation($"{Scorer.Name} on validation data: {SimpleMath.Round(BestScoreOnValidationData, 4)}");
		}
	}


	public override double Eval(DataPoint dataPoint)
	{
		var score = 0.0;
		for (var j = 0; j < _rankers.Count; j++)
		{
			score += _rweight[j] * dataPoint.GetFeatureValue(_rankers[j].Fid);
		}
		return score;
	}

	public override string ToString()
	{
		var output = new StringBuilder();
		for (var i = 0; i < _rankers.Count; i++)
			output.Append(_rankers[i].Fid + ":" + _rweight[i] + (i == _rankers.Count - 1 ? "" : " "));
		return output.ToString();
	}

	public override string Model
	{
		get
		{
			var output = new StringBuilder();
			output.Append($"## {Name}\n");
			output.Append($"## Iteration = {Parameters.NIteration}\n");
			output.Append($"## Train with enqueue: {(Parameters.TrainWithEnqueue ? "Yes" : "No")}\n");
			output.Append($"## Tolerance = {Parameters.Tolerance}\n");
			output.Append($"## Max consecutive selection count = {Parameters.MaxSelCount}\n");
			output.Append(ToString());
			return output.ToString();
		}
	}

	public override void LoadFromString(string fullText)
	{
		try
		{
			using var reader = new StringReader(fullText);
			KeyValuePairs? kvp = null;
			while (reader.ReadLine() is { } content)
			{
				content = content.Trim();
				if (content.Length == 0 || content.StartsWith("##"))
				{
					continue;
				}
				kvp = new KeyValuePairs(content);
				break;
			}

			if (kvp == null)
			{
				throw new InvalidOperationException("Error in AdaRank::LoadFromString: Unable to load model");
			}

			_rweight = new List<double>();
			_rankers = new List<AdaRankWeakRanker>();
			Features = new int[kvp.Count];

			for (var i = 0; i < kvp.Count; i++)
			{
				var kv = kvp[i];
				Features[i] = int.Parse(kv.Key);
				_rankers.Add(new AdaRankWeakRanker(Features[i]));
				_rweight.Add(double.Parse(kv.Key));
			}
		}
		catch (Exception ex)
		{
			throw new InvalidOperationException("Error loading AdaRank from string", ex);
		}
	}
}
