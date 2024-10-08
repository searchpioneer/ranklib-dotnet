using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Stats;

namespace RankLib.Eval;

public partial class Analyzer
{
	[GeneratedRegex(@"\s+")]
	private static partial Regex WhitespaceRegex();

	private static readonly RandomPermutationTest RandomizedTest = new();
	private static readonly double[] ImprovementRatioThreshold = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1000];
	private const int IndexOfZero = 4;

	private readonly ILogger<Analyzer> _logger;

	public Analyzer(ILogger<Analyzer>? logger = null) => _logger = logger ?? NullLogger<Analyzer>.Instance;

	public class Result
	{
		public int Status = 0;
		public int Win = 0;
		public int Loss = 0;
		public int[] CountByImprovementRange;
	}

	private int LocateSegment(double value)
	{
		if (value > 0)
		{
			for (var i = IndexOfZero; i < ImprovementRatioThreshold.Length; i++)
			{
				if (value <= ImprovementRatioThreshold[i])
				{
					return i;
				}
			}
		}
		else if (value < 0)
		{
			for (var i = 0; i <= IndexOfZero; i++)
			{
				if (value < ImprovementRatioThreshold[i])
				{
					return i;
				}
			}
		}
		return -1;
	}

	public Dictionary<string, double> Read(string filename)
	{
		var performance = new Dictionary<string, double>();
		using (var reader = new StreamReader(filename))
		{
			while (reader.ReadLine() is { } line)
			{
				line = WhitespaceRegex().Replace(line.Trim(), "\t");
				var parts = line.Split('\t');
				performance[parts[1]] = double.Parse(parts[2]);
			}
		}
		_logger.LogInformation($"Reading {filename}... {performance.Count} ranked lists");
		return performance;
	}

	public void Compare(string directory, string baseFile)
	{
		directory = Path.GetFullPath(directory);
		var targetFiles = Directory.GetFiles(directory).ToList();

		targetFiles.RemoveAll(file => Path.GetFileName(file) == baseFile);
		targetFiles = targetFiles.Select(file => Path.Combine(directory, Path.GetFileName(file))).ToList();

		Compare(targetFiles, Path.Combine(directory, baseFile));
	}

	public void Compare(List<string> targetFiles, string baseFile)
	{
		var basePerformance = Read(baseFile);
		var targetPerformances = targetFiles.Select(Read).ToList();

		var results = Compare(basePerformance, targetPerformances);

		_logger.LogInformation("Overall comparison");
		_logger.LogInformation("System\tPerformance\tImprovement\tWin\tLoss\tp-value");

		_logger.LogInformation($"{Path.GetFileName(baseFile)} [baseline]\t{basePerformance["all"]:F4}");

		for (var i = 0; i < results.Length; i++)
		{
			if (results[i].Status == 0)
			{
				var delta = targetPerformances[i]["all"] - basePerformance["all"];
				var dp = delta * 100 / basePerformance["all"];
				_logger.LogInformation($"{Path.GetFileName(targetFiles[i])}\t{targetPerformances[i]["all"]:F4}\t" +
									  $"{(delta > 0 ? "+" : "")}{delta:F4} ({(delta > 0 ? "+" : "")}{dp:F2}%)" +
									  $"\t{results[i].Win}\t{results[i].Loss}\t{RandomizedTest.Test(targetPerformances[i], basePerformance)}");
			}
			else
			{
				_logger.LogInformation($"WARNING: [{targetFiles[i]}] skipped: NOT comparable to the baseline due to different ranked list IDs.");
			}
		}

		_logger.LogInformation("Detailed break down");
		var header = "";
		var tmp = new string[ImprovementRatioThreshold.Length];
		for (var i = 0; i < ImprovementRatioThreshold.Length; i++)
		{
			var t = $"{(int)(ImprovementRatioThreshold[i] * 100)}%";
			tmp[i] = t;
		}

		header += $"[ < {tmp[0]} )\t";
		for (var i = 0; i < ImprovementRatioThreshold.Length - 2; i++)
		{
			header += i >= IndexOfZero ? $"( {tmp[i]} , {tmp[i + 1]} ]\t" : $"[ {tmp[i]} , {tmp[i + 1]} )\t";
		}
		header += $"( > {tmp[ImprovementRatioThreshold.Length - 2]} ]";
		_logger.LogInformation("\t{Header}", header);

		for (var i = 0; i < targetFiles.Count; i++)
		{
			var resultDetails = targetFiles[i];
			foreach (var count in results[i].CountByImprovementRange)
			{
				resultDetails += "\t" + count;
			}
			_logger.LogInformation("{Result}", resultDetails);
		}
	}

	public Result[] Compare(Dictionary<string, double> basePerformance, List<Dictionary<string, double>> targets)
	{
		var results = new Result[targets.Count];
		for (var i = 0; i < targets.Count; i++)
		{
			results[i] = Compare(basePerformance, targets[i]);
		}
		return results;
	}

	public Result Compare(Dictionary<string, double> basePerformance, Dictionary<string, double> target)
	{
		var result = new Result
		{
			CountByImprovementRange = new int[ImprovementRatioThreshold.Length]
		};

		if (basePerformance.Count != target.Count)
		{
			result.Status = -1;
			return result;
		}

		foreach (var key in basePerformance.Keys)
		{
			if (!target.ContainsKey(key))
			{
				result.Status = -2;
				return result;
			}

			if (key == "all")
				continue;

			var baseValue = basePerformance[key];
			var targetValue = target[key];

			if (targetValue > baseValue)
			{
				result.Win++;
			}
			else if (targetValue < baseValue)
			{
				result.Loss++;
			}

			var change = targetValue - baseValue;
			if (change != 0)
			{
				result.CountByImprovementRange[LocateSegment(change)]++;
			}
		}

		return result;
	}
}
