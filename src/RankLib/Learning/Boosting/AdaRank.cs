using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning.Boosting;

public class AdaRank : Ranker
{
	private readonly ILogger<AdaRank> _logger;

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

	private WeakRanker? LearnWeakRanker()
	{
		var bestScore = -1.0;
		WeakRanker? bestWeakRanker = null;

		foreach (var i in Features)
		{
			if (_featureQueue.Contains(i) || _usedFeatures.ContainsKey(i))
				continue;

			var wr = new WeakRanker(i);
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

		for (; t <= NIteration; t++)
		{
			PrintLog([7], [t.ToString()]);

			var bestWeakRanker = LearnWeakRanker();
			if (bestWeakRanker == null)
				break;

			if (withEnqueue)
			{
				if (bestWeakRanker.GetFID() == _lastFeature)
				{
					_featureQueue.Add(_lastFeature);
					_rankers.RemoveAt(_rankers.Count - 1);
					_rweight.RemoveAt(_rweight.Count - 1);
					Array.Copy(_backupSampleWeight, _sweight, _sweight.Length);
					BestScoreOnValidationData = 0.0;
					_lastTrainedScore = _backupTrainScore;
					PrintLogLn([8, 9, 9, 9], [bestWeakRanker.GetFID().ToString(), "", "", "ROLLBACK"]);
					continue;
				}
				else
				{
					_lastFeature = bestWeakRanker.GetFID();
					Array.Copy(_sweight, _backupSampleWeight, _sweight.Length);
					_backupTrainScore = _lastTrainedScore;
				}
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
					if (_lastFeature == bestWeakRanker.GetFID())
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

				_lastFeature = bestWeakRanker.GetFID();
			}

			PrintLog(new[] { 8, 9 }, new[] { bestWeakRanker.GetFID().ToString(), SimpleMath.Round(trainedScore, 4).ToString() });
			if (t % 1 == 0 && ValidationSamples != null)
			{
				var scoreOnValidation = Scorer.Score(Rank(ValidationSamples));
				if (scoreOnValidation > BestScoreOnValidationData)
				{
					BestScoreOnValidationData = scoreOnValidation;
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

		_rankers = new List<WeakRanker>();
		_rweight = new List<double>();

		_featureQueue = new List<int>();

		BestScoreOnValidationData = 0.0;
		_bestModelRankers = new List<WeakRanker>();
		_bestModelWeights = new List<double>();
	}

	public override void Learn()
	{
		_logger.LogInformation("Training starts...");
		PrintLogLn(new[] { 7, 8, 9, 9, 9 }, new[] { "#iter", "Sel. F.", Scorer.Name + "-T", Scorer.Name + "-V", "Status" });

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

		if (ValidationSamples != null && _bestModelRankers.Count > 0)
		{
			_rankers.Clear();
			_rweight.Clear();
			_rankers.AddRange(_bestModelRankers);
			_rweight.AddRange(_bestModelWeights);
		}

		ScoreOnTrainingData = SimpleMath.Round(Scorer.Score(Rank(Samples)), 4);
		_logger.LogInformation($"Finished successfully.");
		_logger.LogInformation($"{Scorer.Name} on training data: {ScoreOnTrainingData}");

		if (ValidationSamples != null)
		{
			BestScoreOnValidationData = Scorer.Score(Rank(ValidationSamples));
			_logger.LogInformation($"{Scorer.Name} on validation data: {SimpleMath.Round(BestScoreOnValidationData, 4)}");
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

	public override string Model
	{
		get
		{
			var output = new StringBuilder();
			output.Append("## " + Name + "\n");
			output.Append("## Iteration = " + NIteration + "\n");
			output.Append("## Train with enqueue: " + (TrainWithEnqueue ? "Yes" : "No") + "\n");
			output.Append("## Tolerance = " + Tolerance + "\n");
			output.Append("## Max consecutive selection count = " + MaxSelCount + "\n");
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

			var keys = kvp.Keys;
			var values = kvp.Values;
			_rweight = new List<double>();
			_rankers = new List<WeakRanker>();
			Features = new int[keys.Count];

			for (var i = 0; i < keys.Count; i++)
			{
				Features[i] = int.Parse(keys[i]);
				_rankers.Add(new WeakRanker(Features[i]));
				_rweight.Add(double.Parse(values[i]));
			}
		}
		catch (Exception ex)
		{
			throw new InvalidOperationException("Error in AdaRank::LoadFromString: ", ex);
		}
	}

	public override void PrintParameters()
	{
		_logger.LogInformation("No. of rounds: {NIteration}", NIteration);
		_logger.LogInformation("Train with 'enqueue': {TrainWithEnqueue}", TrainWithEnqueue ? "Yes" : "No");
		_logger.LogInformation("Tolerance: {Tolerance}", Tolerance);
		_logger.LogInformation("Max Sel. Count: {MaxSelCount}", MaxSelCount);
	}

	public override string Name => "AdaRank";
}
