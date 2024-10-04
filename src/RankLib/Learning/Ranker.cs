using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning;

public abstract class Ranker
{
	// TODO: logging
	private static readonly ILogger<Ranker> _logger = NullLogger<Ranker>.Instance;

	protected List<RankList> _samples = new List<RankList>(); // training samples
	protected int[] _features = null;
	protected MetricScorer _scorer = null;
	protected double _scoreOnTrainingData = 0.0;
	protected double _bestScoreOnValidationData = 0.0;

	protected List<RankList> _validationSamples = null;
	protected StringBuilder _logBuf = new StringBuilder(1000);

	protected Ranker() { }

	protected Ranker(List<RankList> samples, int[] features, MetricScorer scorer)
	{
		_samples = samples;
		_features = features;
		_scorer = scorer;
	}

	// Utility functions
	public void SetTrainingSet(List<RankList> samples) => _samples = samples;

	public void SetFeatures(int[] features) => _features = features;

	public void SetValidationSet(List<RankList> samples) => _validationSamples = samples;

	public void SetMetricScorer(MetricScorer scorer) => _scorer = scorer;

	public double GetScoreOnTrainingData() => _scoreOnTrainingData;

	public double GetScoreOnValidationData() => _bestScoreOnValidationData;

	public int[] GetFeatures() => _features;

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

		FileUtils.Write(modelFile, "ASCII", Model());
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
					_logBuf.Append(msg.Substring(0, len[i]));
				}
				else
				{
					_logBuf.Append(msg);
					for (var j = len[i] - msg.Length; j > 0; j--)
					{
						_logBuf.Append(' ');
					}
				}
				_logBuf.Append(" | ");
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
			if (_logBuf.Length > 0)
			{
				_logger.LogInformation(_logBuf.ToString());
				_logBuf.Clear();
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
	public abstract string Model();
	public abstract void LoadFromString(string fullText);
	public abstract string Name();
	public abstract void PrintParameters();
}
