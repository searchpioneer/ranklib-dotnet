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

	public List<RankList> Samples { get; set; } = []; // training samples

	public List<RankList>? ValidationSamples { get; set; }

	public int[] Features { get; set; } = [];

	/// <summary>
	/// Gets or sets the scorer
	/// </summary>
	public MetricScorer? Scorer { get; set; }

	protected double ScoreOnTrainingData = 0.0;
	protected double BestScoreOnValidationData = 0.0;

	protected Ranker(ILogger<Ranker>? logger = null) => _logger = logger ?? NullLogger<Ranker>.Instance;

	protected Ranker(List<RankList> samples, int[] features, MetricScorer scorer, ILogger<Ranker>? logger = null)
	{
		Samples = samples;
		Features = features;
		Scorer = scorer;
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
	public abstract override string ToString();
	public abstract string Model { get; }
	public abstract void LoadFromString(string fullText);
	public abstract string Name { get; }
	public abstract void PrintParameters();
}
