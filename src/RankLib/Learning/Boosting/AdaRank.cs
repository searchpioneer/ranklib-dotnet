using System.Globalization;
using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning.Boosting;

/// <summary>
/// Parameters for <see cref="AdaRank"/>
/// </summary>
public class AdaRankParameters : IRankerParameters
{
	/// <summary>
	/// Default number of iterations (rounds).
	/// </summary>
	public const int DefaultIterationCount = 500;

	/// <summary>
	/// Default tolerance
	/// </summary>
	public const double DefaultTolerance = 0.002;

	/// <summary>
	/// Default train with enqueue
	/// </summary>
	public const bool DefaultTrainWithEnqueue = true;

	/// <summary>
	/// Default Max number of times a feature can be selected consecutively before being removed
	/// </summary>
	public const int DefaultMaximumSelectedCount = 5;

	/// <summary>
	/// Number of iterations (rounds).
	/// </summary>
	public int IterationCount { get; set; } = DefaultIterationCount;

	/// <summary>
	/// Tolerance
	/// </summary>
	public double Tolerance { get; set; } = DefaultTolerance;

	/// <summary>
	/// Whether to train with enqueue
	/// </summary>
	public bool TrainWithEnqueue { get; set; } = DefaultTrainWithEnqueue;

	/// <summary>
	/// Max number of times a feature can be selected consecutively before being removed
	/// </summary>
	public int MaximumSelectedCount { get; set; } = DefaultMaximumSelectedCount;

	/// <inheritdoc />
	public override string ToString()
	{
		var builder = new StringBuilder();
		builder.AppendLine($"No. of rounds: {IterationCount}");
		builder.AppendLine($"Train with 'enqueue': {(TrainWithEnqueue ? "Yes" : "No")}");
		builder.AppendLine($"Tolerance: {Tolerance}");
		builder.AppendLine($"Max Sel. Count: {MaximumSelectedCount}");
		return builder.ToString();
	}
}

/// <summary>
/// AdaRank is a boosting algorithm designed for ranking tasks,
/// optimizing ranking-specific metrics like NDCG and MAP by iteratively
/// training weak rankers and adapting to misranked instances in each iteration.
/// </summary>
/// <remarks>
/// <a href="https://dl.acm.org/doi/10.1145/1277741.1277809">
/// J. Xu and H. Li. AdaRank: a boosting algorithm for information retrieval. In Proc. of SIGIR, pages 391-398, 2007.
/// </a>
/// </remarks>
public class AdaRank : Ranker<AdaRankParameters>
{
	internal const string RankerName = "AdaRank";

	private readonly ILogger<AdaRank> _logger;
	private readonly Dictionary<int, int> _usedFeatures = new();
	private double[] _sampleWeights = []; // Sample weight
	private List<AdaRankWeakRanker> _rankers = []; // Alpha
	private List<double> _rankerWeights = []; // Weak rankers' weight
	private List<AdaRankWeakRanker> _bestModelRankers = [];
	private List<double> _bestModelWeights = [];

	// For the implementation of tricks
	private int _lastFeature = -1;
	private int _lastFeatureConsecutiveCount;
#pragma warning disable CS0414 // Field is assigned but its value is never used
	private bool _performanceChanged;
#pragma warning restore CS0414 // Field is assigned but its value is never used
	private List<int> _featureQueue = [];
	private double[] _backupSampleWeight = [];
	private double _backupTrainScore;
	private double _lastTrainedScore = -1.0;

	public override string Name => RankerName;

	public AdaRank(ILogger<AdaRank>? logger = null) => _logger = logger ?? NullLogger<AdaRank>.Instance;

	public AdaRank(List<RankList> samples, int[] features, MetricScorer scorer, ILogger<AdaRank>? logger = null) : base(
		samples, features, scorer) =>
		_logger = logger ?? NullLogger<AdaRank>.Instance;

	private void UpdateBestModelOnValidation()
	{
		_bestModelRankers.Clear();
		_bestModelRankers.AddRange(_rankers);
		_bestModelWeights.Clear();
		_bestModelWeights.AddRange(_rankerWeights);
	}

	private AdaRankWeakRanker? LearnWeakRanker()
	{
		var bestScore = -1.0;
		AdaRankWeakRanker? bestWeakRanker = null;

		for (var i = 0; i < Features.Length; i++)
		{
			var fid = Features[i];
			if (_featureQueue.Contains(fid) || _usedFeatures.ContainsKey(fid))
				continue;

			var weakRanker = new AdaRankWeakRanker(fid);
			var s = 0.0;
			for (var j = 0; j < Samples.Count; j++)
			{
				var t = Scorer.Score(weakRanker.Rank(Samples[j])) * _sampleWeights[j];
				s += t;
			}

			if (bestScore < s)
			{
				bestScore = s;
				bestWeakRanker = weakRanker;
			}
		}

		return bestWeakRanker;
	}

	private int Learn(int startIteration, bool withEnqueue)
	{
		var t = startIteration;
		var bufferedLogger = new BufferedLogger(_logger, new StringBuilder());
		for (; t <= Parameters.IterationCount; t++)
		{
			bufferedLogger.PrintLog([7], [t.ToString()]);

			var bestWeakRanker = LearnWeakRanker();
			if (bestWeakRanker == null)
				break;

			if (withEnqueue)
			{
				if (bestWeakRanker.Fid == _lastFeature)
				{
					_featureQueue.Add(_lastFeature);
					_rankers.RemoveAt(_rankers.Count - 1);
					_rankerWeights.RemoveAt(_rankerWeights.Count - 1);
					Array.Copy(_backupSampleWeight, _sampleWeights, _sampleWeights.Length);
					ValidationDataScore = 0.0;
					_lastTrainedScore = _backupTrainScore;
					bufferedLogger.PrintLogLn([8, 9, 9, 9], [bestWeakRanker.Fid.ToString(), "", "", "ROLLBACK"]);
					continue;
				}

				_lastFeature = bestWeakRanker.Fid;
				Array.Copy(_sampleWeights, _backupSampleWeight, _sampleWeights.Length);
				_backupTrainScore = _lastTrainedScore;
			}

			var num = 0.0;
			var denom = 0.0;
			for (var i = 0; i < Samples.Count; i++)
			{
				var tmp = Scorer.Score(bestWeakRanker.Rank(Samples[i]));
				num += _sampleWeights[i] * (1.0 + tmp);
				denom += _sampleWeights[i] * (1.0 - tmp);
			}

			_rankers.Add(bestWeakRanker);
			var alphaT = 0.5 * SimpleMath.Ln(num / denom);
			_rankerWeights.Add(alphaT);

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
				// ReSharper disable once CompareOfFloatsByEqualityOperator
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
						if (_lastFeatureConsecutiveCount == Parameters.MaximumSelectedCount)
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

			bufferedLogger.PrintLog([8, 9], [bestWeakRanker.Fid.ToString(), SimpleMath.Round(trainedScore, 4).ToString(CultureInfo.InvariantCulture)]);
			if (t % 1 == 0 && ValidationSamples != null)
			{
				var scoreOnValidation = Scorer.Score(Rank(ValidationSamples));
				if (scoreOnValidation > ValidationDataScore)
				{
					ValidationDataScore = scoreOnValidation;
					UpdateBestModelOnValidation();
				}

				bufferedLogger.PrintLog([9, 9],
					[SimpleMath.Round(scoreOnValidation, 4).ToString(CultureInfo.InvariantCulture), status]);
			}
			else
				bufferedLogger.PrintLog([9, 9], ["", status]);

			bufferedLogger.FlushLog();

			if (delta <= 0)
			{
				_rankers.RemoveAt(_rankers.Count - 1);
				_rankerWeights.RemoveAt(_rankerWeights.Count - 1);
				break;
			}

			_lastTrainedScore = trainedScore;

			for (var i = 0; i < _sampleWeights.Length; i++)
				_sampleWeights[i] *= Math.Exp(-alphaT * Scorer.Score(Rank(Samples[i]))) / total;
		}

		return t;
	}

	public override Task InitAsync()
	{
		_logger.LogInformation("Initializing...");
		_usedFeatures.Clear();

		_sampleWeights = new double[Samples.Count];
		for (var i = 0; i < _sampleWeights.Length; i++)
			_sampleWeights[i] = 1.0f / Samples.Count;

		_backupSampleWeight = new double[_sampleWeights.Length];
		Array.Copy(_sampleWeights, _backupSampleWeight, _sampleWeights.Length);

		_lastTrainedScore = -1.0;
		_rankers = [];
		_rankerWeights = [];
		_featureQueue = [];
		ValidationDataScore = 0.0;
		_bestModelRankers = [];
		_bestModelWeights = [];

		return Task.CompletedTask;
	}

	public override Task LearnAsync()
	{
		_logger.LogInformation("Training starts...");
		_logger.PrintLog([7, 8, 9, 9, 9], ["#iter", "Sel. F.", Scorer.Name + "-T", Scorer.Name + "-V", "Status"]);

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
			Learn(1, false);

		if (ValidationSamples != null && _bestModelRankers.Count > 0)
		{
			_rankers.Clear();
			_rankerWeights.Clear();
			_rankers.AddRange(_bestModelRankers);
			_rankerWeights.AddRange(_bestModelWeights);
		}

		TrainingDataScore = SimpleMath.Round(Scorer.Score(Rank(Samples)), 4);
		_logger.LogInformation("Finished successfully.");
		_logger.LogInformation($"{Scorer.Name} on training data: {TrainingDataScore}");

		if (ValidationSamples != null)
		{
			ValidationDataScore = Scorer.Score(Rank(ValidationSamples));
			_logger.LogInformation($"{Scorer.Name} on validation data: {SimpleMath.Round(ValidationDataScore, 4)}");
		}

		return Task.CompletedTask;
	}


	public override double Eval(DataPoint dataPoint)
	{
		var score = 0.0;
		for (var j = 0; j < _rankers.Count; j++)
			score += _rankerWeights[j] * dataPoint.GetFeatureValue(_rankers[j].Fid);

		return score;
	}

	public override string GetModel()
	{
		var output = new StringBuilder();
		output.Append($"## {Name}\n");
		output.Append($"## Iteration = {Parameters.IterationCount}\n");
		output.Append($"## Train with enqueue: {(Parameters.TrainWithEnqueue ? "Yes" : "No")}\n");
		output.Append($"## Tolerance = {Parameters.Tolerance}\n");
		output.Append($"## Max consecutive selection count = {Parameters.MaximumSelectedCount}\n");

		for (var i = 0; i < _rankers.Count; i++)
			output.Append(_rankers[i].Fid + ":" + _rankerWeights[i] + (i == _rankers.Count - 1 ? "" : " "));

		return output.ToString();
	}

	public override void LoadFromString(string model)
	{
		try
		{
			using var reader = new StringReader(model);
			KeyValuePairs? kvp = null;
			while (reader.ReadLine() is { } content)
			{
				content = content.Trim();
				if (content.Length == 0 || content.StartsWith("##"))
					continue;

				kvp = new KeyValuePairs(content);
				break;
			}

			if (kvp == null)
				throw new InvalidOperationException("Error in AdaRank::LoadFromString: Unable to load model");

			_rankerWeights = new List<double>();
			_rankers = new List<AdaRankWeakRanker>();
			Features = new int[kvp.Count];

			for (var i = 0; i < kvp.Count; i++)
			{
				var kv = kvp[i];
				Features[i] = int.Parse(kv.Key);
				_rankers.Add(new AdaRankWeakRanker(Features[i]));
				_rankerWeights.Add(double.Parse(kv.Key));
			}
		}
		catch (Exception ex)
		{
			throw RankLibException.Create("Error loading model", ex);
		}
	}
}
