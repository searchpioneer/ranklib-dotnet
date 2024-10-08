using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning;

public abstract class Ranker
{
	private readonly ILogger<Ranker> _logger;
	private readonly StringBuilder _logBuffer = new();

	protected List<RankList> Samples = new(); // training samples
	public int[]? Features { get; set; }

	/// <summary>
	/// Gets or sets the scorer
	/// </summary>
	public MetricScorer? Scorer { get; set; }

	protected double ScoreOnTrainingData = 0.0;
	protected double BestScoreOnValidationData = 0.0;
	protected List<RankList>? ValidationSamples;

	protected Ranker(ILogger<Ranker>? logger = null) => _logger = logger ?? NullLogger<Ranker>.Instance;

	protected Ranker(List<RankList> samples, int[] features, MetricScorer scorer, ILogger<Ranker>? logger = null)
	{
		Samples = samples;
		Features = features;
		Scorer = scorer;
		_logger = logger ?? NullLogger<Ranker>.Instance;
	}

	// Utility functions
	public void SetTrainingSet(List<RankList> samples) => Samples = samples;

	public void SetValidationSet(List<RankList> validationSamples) => ValidationSamples = validationSamples;

	public double GetScoreOnTrainingData() => ScoreOnTrainingData;

	public double GetScoreOnValidationData() => BestScoreOnValidationData;

	public virtual RankList Rank(RankList rl)
	{
		var scores = new double[rl.Count];
		for (var i = 0; i < rl.Count; i++)
		{
			scores[i] = Eval(rl[i]);
		}

		var idx = MergeSorter.Sort(scores, false);
		return new RankList(rl, idx);
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

	// Create the model file directory to write models into if not already there
	public void Save(string modelFile)
	{
		try
		{
			var directory = Path.GetDirectoryName(Path.GetFullPath(modelFile));
			Directory.CreateDirectory(directory!);
		}
		catch (Exception e)
		{
			throw RankLibException.Create($"Error creating kcv model file '{modelFile}'", e);
		}

		FileUtils.Write(modelFile, Encoding.ASCII, Model);
	}

	protected void PrintLog(int[] len, string[] msgs)
	{
		if (_logger.IsEnabled(LogLevel.Information))
		{
			for (var i = 0; i < msgs.Length; i++)
			{
				var msg = msgs[i];
				if (msg.Length > len[i])
				{
					_logBuffer.Append(msg.AsSpan(0, len[i]));
				}
				else
				{
					_logBuffer.Append(msg);
					for (var j = len[i] - msg.Length; j > 0; j--)
					{
						_logBuffer.Append(' ');
					}
				}
				_logBuffer.Append(" | ");
			}
		}
	}

	protected void PrintLogLn(int[] len, string[] msgs)
	{
		if (_logger.IsEnabled(LogLevel.Information))
		{
			PrintLog(len, msgs);
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
	public abstract double Eval(DataPoint p);
	public abstract Ranker CreateNew();
	public abstract override string ToString();
	public abstract string Model { get; }
	public abstract void LoadFromString(string fullText);

	/// <summary>
	/// Gets the name of this ranker
	/// </summary>
	public abstract string Name { get; }
	public abstract void PrintParameters();
}
