using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Utilities;
using KeyValuePair = RankLib.Utilities.KeyValuePair;

namespace RankLib.Learning.Boosting;

public class AdaRank : Ranker
{
	// TODO: logging
	private static readonly ILogger<AdaRank> _logger = NullLogger<AdaRank>.Instance;

	// Parameters
	public static int NIteration = 500;
	public static double Tolerance = 0.002;
	public static bool TrainWithEnqueue = true;
	public static int MaxSelCount = 5; // Max number of times a feature can be selected consecutively before being removed

	protected Dictionary<int, int> _usedFeatures = new();
	protected double[]? _sweight = null; // Sample weight
	protected List<WeakRanker>? _rankers = null; // Alpha
	protected List<double>? _rweight = null; // Weak rankers' weight

	protected List<WeakRanker>? _bestModelRankers = null;
	protected List<double>? _bestModelWeights = null;

	// For the implementation of tricks
	private int _lastFeature = -1;
	private int _lastFeatureConsecutiveCount = 0;
	private bool _performanceChanged = false;
	private List<int> _featureQueue = null;
	protected double[] _backupSampleWeight = null;
	protected double _backupTrainScore = 0.0;
	protected double _lastTrainedScore = -1.0;

	public AdaRank() { }

	public AdaRank(List<RankList> samples, int[] features, MetricScorer scorer) : base(samples, features, scorer) { }

	private void UpdateBestModelOnValidation()
	{
		_bestModelRankers.Clear();
		_bestModelRankers.AddRange(_rankers);
		_bestModelWeights.Clear();
		_bestModelWeights.AddRange(_rweight);
	}

	private WeakRanker? LearnWeakRanker()
	{
		var bestScore = -1.0;
		WeakRanker? bestWR = null;

		foreach (var i in _features)
		{
			if (_featureQueue.Contains(i) || _usedFeatures.ContainsKey(i))
				continue;

			var wr = new WeakRanker(i);
			var s = 0.0;
			for (var j = 0; j < _samples.Count; j++)
			{
				var t = _scorer.Score(wr.Rank(_samples[j])) * _sweight[j];
				s += t;
			}

			if (bestScore < s)
			{
				bestScore = s;
				bestWR = wr;
			}
		}

		return bestWR;
	}

	private int Learn(int startIteration, bool withEnqueue)
	{
		var t = startIteration;

		for (; t <= NIteration; t++)
		{
			PrintLog(new[] { 7 }, new[] { t.ToString() });

			var bestWR = LearnWeakRanker();
			if (bestWR == null)
				break;

			if (withEnqueue)
			{
				if (bestWR.GetFID() == _lastFeature)
				{
					_featureQueue.Add(_lastFeature);
					_rankers.RemoveAt(_rankers.Count - 1);
					_rweight.RemoveAt(_rweight.Count - 1);
					Array.Copy(_backupSampleWeight, _sweight, _sweight.Length);
					_bestScoreOnValidationData = 0.0;
					_lastTrainedScore = _backupTrainScore;
					PrintLogLn(new[] { 8, 9, 9, 9 }, new[] { bestWR.GetFID().ToString(), "", "", "ROLLBACK" });
					continue;
				}
				else
				{
					_lastFeature = bestWR.GetFID();
					Array.Copy(_sweight, _backupSampleWeight, _sweight.Length);
					_backupTrainScore = _lastTrainedScore;
				}
			}

			var num = 0.0;
			var denom = 0.0;
			for (var i = 0; i < _samples.Count; i++)
			{
				var tmp = _scorer.Score(bestWR.Rank(_samples[i]));
				num += _sweight[i] * (1.0 + tmp);
				denom += _sweight[i] * (1.0 - tmp);
			}

			_rankers.Add(bestWR);
			var alpha_t = 0.5 * SimpleMath.Ln(num / denom);
			_rweight.Add(alpha_t);

			var trainedScore = 0.0;
			var total = 0.0;

			foreach (var sample in _samples)
			{
				var tmp = _scorer.Score(Rank(sample));
				total += Math.Exp(-alpha_t * tmp);
				trainedScore += tmp;
			}

			trainedScore /= _samples.Count;
			var delta = trainedScore + Tolerance - _lastTrainedScore;
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
					if (_lastFeature == bestWR.GetFID())
					{
						_lastFeatureConsecutiveCount++;
						if (_lastFeatureConsecutiveCount == MaxSelCount)
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

				_lastFeature = bestWR.GetFID();
			}

			PrintLog(new[] { 8, 9 }, new[] { bestWR.GetFID().ToString(), SimpleMath.Round(trainedScore, 4).ToString() });
			if (t % 1 == 0 && _validationSamples != null)
			{
				var scoreOnValidation = _scorer.Score(Rank(_validationSamples));
				if (scoreOnValidation > _bestScoreOnValidationData)
				{
					_bestScoreOnValidationData = scoreOnValidation;
					UpdateBestModelOnValidation();
				}

				PrintLog(new[] { 9, 9 }, new[] { SimpleMath.Round(scoreOnValidation, 4).ToString(), status });
			}
			else
			{
				PrintLog(new[] { 9, 9 }, new[] { "", status });
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
				_sweight[i] *= Math.Exp(-alpha_t * _scorer.Score(Rank(_samples[i]))) / total;
			}
		}

		return t;
	}

	public override void Init()
	{
		_logger.LogInformation("Initializing...");
		_usedFeatures.Clear();

		_sweight = new double[_samples.Count];
		for (var i = 0; i < _sweight.Length; i++)
		{
			_sweight[i] = 1.0f / _samples.Count;
		}

		_backupSampleWeight = new double[_sweight.Length];
		Array.Copy(_sweight, _backupSampleWeight, _sweight.Length);
		_lastTrainedScore = -1.0;

		_rankers = new List<WeakRanker>();
		_rweight = new List<double>();

		_featureQueue = new List<int>();

		_bestScoreOnValidationData = 0.0;
		_bestModelRankers = new List<WeakRanker>();
		_bestModelWeights = new List<double>();
	}

	public override void Learn()
	{
		_logger.LogInformation("Training starts...");
		PrintLogLn(new[] { 7, 8, 9, 9, 9 }, new[] { "#iter", "Sel. F.", _scorer.Name() + "-T", _scorer.Name() + "-V", "Status" });

		if (TrainWithEnqueue)
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

		if (_validationSamples != null && _bestModelRankers.Count > 0)
		{
			_rankers.Clear();
			_rweight.Clear();
			_rankers.AddRange(_bestModelRankers);
			_rweight.AddRange(_bestModelWeights);
		}

		_scoreOnTrainingData = SimpleMath.Round(_scorer.Score(Rank(_samples)), 4);
		_logger.LogInformation($"Finished successfully.");
		_logger.LogInformation($"{_scorer.Name()} on training data: {_scoreOnTrainingData}");

		if (_validationSamples != null)
		{
			_bestScoreOnValidationData = _scorer.Score(Rank(_validationSamples));
			_logger.LogInformation($"{_scorer.Name()} on validation data: {SimpleMath.Round(_bestScoreOnValidationData, 4)}");
		}
	}


	public override double Eval(DataPoint p)
	{
		var score = 0.0;
		for (var j = 0; j < _rankers.Count; j++)
		{
			score += _rweight[j] * p.GetFeatureValue(_rankers[j].GetFID());
		}
		return score;
	}

	public override Ranker CreateNew() => new AdaRank();

	public override string ToString()
	{
		var output = new StringBuilder();
		for (var i = 0; i < _rankers.Count; i++)
		{
			output.Append(_rankers[i].GetFID() + ":" + _rweight[i] + (i == _rankers.Count - 1 ? "" : " "));
		}
		return output.ToString();
	}

	public override string Model()
	{
		var output = new StringBuilder();
		output.Append("## " + Name() + "\n");
		output.Append("## Iteration = " + NIteration + "\n");
		output.Append("## Train with enqueue: " + (TrainWithEnqueue ? "Yes" : "No") + "\n");
		output.Append("## Tolerance = " + Tolerance + "\n");
		output.Append("## Max consecutive selection count = " + MaxSelCount + "\n");
		output.Append(ToString());
		return output.ToString();
	}

	public override void LoadFromString(string fullText)
	{
		try
		{
			using (var reader = new StringReader(fullText))
			{
				string content = null;
				KeyValuePair kvp = null;

				while ((content = reader.ReadLine()) != null)
				{
					content = content.Trim();
					if (content.Length == 0 || content.StartsWith("##"))
					{
						continue;
					}
					kvp = new KeyValuePair(content);
					break;
				}

				if (kvp == null)
				{
					throw new InvalidOperationException("Error in AdaRank::LoadFromString: Unable to load model");
				}

				var keys = kvp.Keys();
				var values = kvp.Values();
				_rweight = new List<double>();
				_rankers = new List<WeakRanker>();
				_features = new int[keys.Count];

				for (var i = 0; i < keys.Count; i++)
				{
					_features[i] = int.Parse(keys[i]);
					_rankers.Add(new WeakRanker(_features[i]));
					_rweight.Add(double.Parse(values[i]));
				}
			}
		}
		catch (Exception ex)
		{
			throw new InvalidOperationException("Error in AdaRank::LoadFromString: ", ex);
		}
	}

	public override void PrintParameters()
	{
		_logger.LogInformation($"No. of rounds: {NIteration}");
		_logger.LogInformation($"Train with 'enqueue': {(TrainWithEnqueue ? "Yes" : "No")}");
		_logger.LogInformation($"Tolerance: {Tolerance}");
		_logger.LogInformation($"Max Sel. Count: {MaxSelCount}");
	}

	public override string Name() => "AdaRank";
}
