using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning;

public abstract class Ranker
{
	private readonly ILogger<Ranker> _logger;

	protected List<RankList> Samples = new(); // training samples
	public int[]? Features { get; set; }
	public MetricScorer? Scorer { get; set; }

	protected double ScoreOnTrainingData = 0.0;
	protected double BestScoreOnValidationData = 0.0;

	protected List<RankList>? ValidationSamples = null;
	protected StringBuilder LogBuf = new(1000);

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

	public void SetValidationSet(List<RankList> samples) => ValidationSamples = samples;

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

	public List<RankList> Rank(List<RankList> l)
	{
		var ll = new List<RankList>(l.Count);
		for (var i = 0; i < l.Count; i++)
		{
			ll.Add(Rank(l[i]));
		}
		return ll;
	}

	// Create the model file directory to write models into if not already there
	public void Save(string modelFile)
	{
		// Determine if the directory to write to exists. If not, create it.
		var parentPath = Path.GetDirectoryName(Path.GetFullPath(modelFile));

		if (!Directory.Exists(parentPath))
		{
			try
			{
				Directory.CreateDirectory(parentPath);
			}
			catch (Exception e)
			{
				throw new InvalidOperationException($"Error creating kcv model file directory {modelFile}", e);
			}
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
					LogBuf.Append(msg.Substring(0, len[i]));
				}
				else
				{
					LogBuf.Append(msg);
					for (var j = len[i] - msg.Length; j > 0; j--)
					{
						LogBuf.Append(' ');
					}
				}
				LogBuf.Append(" | ");
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
			if (LogBuf.Length > 0)
			{
				_logger.LogInformation(LogBuf.ToString());
				LogBuf.Clear();
			}
		}
	}

	protected void Copy(double[] source, double[] target)
	{
		for (var j = 0; j < source.Length; j++)
		{
			target[j] = source[j];
		}
	}

	// Abstract methods that need to be implemented by derived classes
	public abstract void Init();
	public abstract void Learn();
	public abstract double Eval(DataPoint p);
	public abstract Ranker CreateNew();
	public abstract override string ToString();
	public abstract string Model { get; }
	public abstract void LoadFromString(string fullText);
	public abstract string Name { get; }
	public abstract void PrintParameters();
}
