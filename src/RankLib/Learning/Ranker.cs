using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning;

public abstract class Ranker<TRankerParameters> : Ranker, IRanker<TRankerParameters>
	where TRankerParameters : IRankerParameters, new()
{
	protected Ranker(ILogger<Ranker<TRankerParameters>>? logger = null) : base(logger)
	{
	}

	protected Ranker(List<RankList> samples, int[] features, MetricScorer scorer, ILogger<Ranker>? logger = null)
	: base(samples, features, scorer, logger)
	{
	}

	public TRankerParameters Parameters { get; set; } = new();

	IRankerParameters IRanker.Parameters
	{
		get => Parameters;
		set => Parameters = (TRankerParameters)value;
	}
}

public abstract class Ranker : IRanker
{
	private readonly ILogger<Ranker> _logger;
	private readonly StringBuilder _logBuffer = new();
	private MetricScorer? _scorer;

	public List<RankList> Samples { get; set; } = []; // training samples

	public List<RankList>? ValidationSamples { get; set; }

	public int[] Features { get; set; } = [];

	/// <summary>
	/// Gets or sets the scorer
	/// </summary>
	/// <remarks>
	/// If no scorer is assigned, a new instance of <see cref="APScorer"/> is instantiated on first get
	/// </remarks>
	public MetricScorer Scorer
	{
		get => _scorer ?? new APScorer();
		set => _scorer = value;
	}

	IRankerParameters IRanker.Parameters { get; set; } = default!;

	protected double ScoreOnTrainingData = 0.0;
	protected double BestScoreOnValidationData = 0.0;

	protected Ranker(ILogger<Ranker>? logger = null) => _logger = logger ?? NullLogger<Ranker>.Instance;

	protected Ranker(List<RankList> samples, int[] features, MetricScorer scorer, ILogger<Ranker>? logger = null)
	{
		Samples = samples;
		Features = features;
		_scorer = scorer;
		_logger = logger ?? NullLogger<Ranker>.Instance;
	}

	// Utility functions

	public double GetScoreOnTrainingData() => ScoreOnTrainingData;

	public double GetScoreOnValidationData() => BestScoreOnValidationData;

	public virtual RankList Rank(RankList rankList)
	{
		var scores = new double[rankList.Count];
		for (var i = 0; i < rankList.Count; i++)
		{
			scores[i] = Eval(rankList[i]);
		}

		var idx = MergeSorter.Sort(scores, false);
		return new RankList(rankList, idx);
	}

	public List<RankList> Rank(List<RankList> rankLists)
	{
		var rankedRankLists = new List<RankList>(rankLists.Count);
		for (var i = 0; i < rankLists.Count; i++)
		{
			rankedRankLists.Add(Rank(rankLists[i]));
		}
		return rankedRankLists;
	}

	public void Save(string modelFile)
	{
		try
		{
			var directory = Path.GetDirectoryName(Path.GetFullPath(modelFile));
			Directory.CreateDirectory(directory!);
		}
		catch (Exception e)
		{
			throw RankLibException.Create($"Error creating directory for model file '{modelFile}'", e);
		}

		FileUtils.Write(modelFile, Encoding.ASCII, Model);
		_logger.LogInformation("Model saved to: {ModelFile}", modelFile);
	}

	protected void PrintLog(int[] len, string[] messages)
	{
		if (_logger.IsEnabled(LogLevel.Information))
		{
			for (var i = 0; i < messages.Length; i++)
			{
				var msg = messages[i];
				if (msg.Length > len[i])
					_logBuffer.Append(msg.AsSpan(0, len[i]));
				else
					_logBuffer.Append(msg.PadRight(len[i], ' '));
				_logBuffer.Append(" | ");
			}
		}
	}

	protected void PrintLogLn(int[] len, string[] messages)
	{
		if (_logger.IsEnabled(LogLevel.Information))
		{
			PrintLog(len, messages);
			FlushLog();
		}
	}

	protected void FlushLog()
	{
		if (_logger.IsEnabled(LogLevel.Information))
		{
			if (_logBuffer.Length > 0)
			{
				_logger.LogInformation("{Message}", _logBuffer.ToString());
				_logBuffer.Clear();
			}
		}
	}

	public abstract void Init();
	public abstract void Learn();
	public abstract double Eval(DataPoint dataPoint);
	public abstract override string ToString();
	public abstract string Model { get; }
	public abstract void LoadFromString(string fullText);
	public abstract string Name { get; }
}
