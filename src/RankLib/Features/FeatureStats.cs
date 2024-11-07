using MathNet.Numerics.Statistics;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Learning;
using RankLib.Learning.Boosting;
using RankLib.Learning.NeuralNet;
using RankLib.Learning.Tree;

namespace RankLib.Features;

public class FeatureStats
{
	private readonly ILogger<FeatureStats> _logger;

	private readonly string _modelFileName;
	private readonly FileInfo _file;
	private static readonly string[] ModelsThatUseAllFeatures = [CoordinateAscent.RankerName, LambdaRank.RankerName, LinearRegression.RankerName, ListNet.RankerName, RankNet.RankerName];
	private static readonly string[] FeatureWeightModels = [AdaRank.RankerName, RankBoost.RankerName];
	private static readonly string[] TreeModels = [LambdaMART.RankerName, MART.RankerName, RandomForests.RankerName];

	public FeatureStats(string modelFileName, ILogger<FeatureStats>? logger = null)
	{
		_logger = logger ?? NullLogger<FeatureStats>.Instance;
		_file = new FileInfo(modelFileName);
		_modelFileName = _file.FullName;
	}

	private SortedDictionary<int, int> GetFeatureWeightFeatureFrequencies(StreamReader sr)
	{
		var featureFrequencies = new SortedDictionary<int, int>();

		try
		{
			while (sr.ReadLine() is { } line)
			{
				var lineSpan = line.AsSpan().Trim();
				if (lineSpan.IsEmpty || lineSpan.IsWhiteSpace() || lineSpan.Contains("##", StringComparison.Ordinal))
					continue;

				var ranges = new Span<Range>();
				lineSpan.Split(ranges, ' ');

				foreach (var range in ranges)
				{
					var feature = lineSpan[range];
					var colonIndex = feature.IndexOf(':');

					if (colonIndex != -1)
					{
						throw new ArgumentException("Invalid feature line: " + lineSpan.ToString());
					}

					var featureId = int.Parse(feature.Slice(0, colonIndex));
					if (!featureFrequencies.TryAdd(featureId, 1))
						featureFrequencies[featureId]++;
				}
			}
		}
		catch (Exception ex)
		{
			throw new Exception($"Exception: {ex.Message}", ex);
		}

		return featureFrequencies;
	}

	private SortedDictionary<int, int> GetTreeFeatureFrequencies(StreamReader reader)
	{
		var featureFrequencies = new SortedDictionary<int, int>();

		try
		{
			while (reader.ReadLine() is { } line)
			{
				var lineSpan = line.AsSpan().Trim();
				if (lineSpan.IsEmpty || lineSpan.IsWhiteSpace() || lineSpan.Contains("##", StringComparison.Ordinal))
					continue;

				if (lineSpan.Contains("<feature>", StringComparison.InvariantCultureIgnoreCase))
				{
					var quote1 = lineSpan.IndexOf('>');
					var quote2 = lineSpan.Slice(quote1).IndexOf('<');

					var featureIdStr = lineSpan.Slice(quote1 + 1, quote2 - 1);
					var featureId = int.Parse(featureIdStr);

					if (!featureFrequencies.TryAdd(featureId, 1))
						featureFrequencies[featureId]++;
				}
			}
		}
		catch (Exception ex)
		{
			throw new Exception($"Exception: {ex.Message}", ex);
		}

		return featureFrequencies;
	}

	public void WriteFeatureStats()
	{
		SortedDictionary<int, int>? featureFrequencies = null;
		string? modelName;

		try
		{
			using var reader = new StreamReader(_file.FullName);

			// Read model name from the file
			var modelLine = reader.ReadLine().AsSpan().Trim();
			var ranges = new Span<Range>();
			var len = modelLine.Split(ranges, ' ');
			modelName = len switch
			{
				2 => modelLine[ranges[1]].Trim().ToString(),
				3 => modelLine.Slice(ranges[1].Start.Value, ranges[2].End.Value - ranges[1].Start.Value)
					.Trim()
					.ToString(),
				_ => null
			};

			if (string.IsNullOrEmpty(modelName))
				throw new Exception($"Expected to find model name on first line, but found {modelLine}");

			// Handle models that use all features
			if (ModelsThatUseAllFeatures.Contains(modelName))
			{
				_logger.LogInformation("{ModelName} uses all features. Can't do selected model statistics for this algorithm.", modelName);
				return;
			}

			// Feature:Weight models
			if (FeatureWeightModels.Contains(modelName))
				featureFrequencies = GetFeatureWeightFeatureFrequencies(reader);
			// Tree models
			else if (TreeModels.Contains(modelName))
				featureFrequencies = GetTreeFeatureFrequencies(reader);
		}
		catch (IOException exception)
		{
			throw new Exception($"IOException on file {_modelFileName}: {exception.Message}", exception);
		}

		if (featureFrequencies is null)
			throw new Exception("No feature frequencies defined.");

		// Calculate feature statistics
		var featuresUsed = featureFrequencies.Count;

		_logger.LogInformation("Model File: {ModelFileName}", _modelFileName);
		_logger.LogInformation("Algorithm: {ModelName}", modelName);
		_logger.LogInformation("Feature frequencies:");

		var data = new List<double>(featuresUsed);

		foreach (var entry in featureFrequencies)
		{
			var featureId = entry.Key;
			var freq = entry.Value;
			_logger.LogInformation("\tFeature[{FeatureId}] : {Freq}", featureId, freq);
			data.Add(freq);
		}

		var stats = new DescriptiveStatistics(data);

		// Print out summary statistics
		_logger.LogInformation("Total Features Used: {FeaturesUsed}", featuresUsed);
		_logger.LogInformation($"Min frequency    : {stats.Minimum:0.00}");
		_logger.LogInformation($"Max frequency    : {stats.Maximum:0.00}");
		//logger.LogInformation($"Median frequency : {stats.Median:0.00}");
		_logger.LogInformation($"Avg frequency    : {stats.Mean:0.00}");
		_logger.LogInformation($"Variance         : {stats.Variance:0.00}");
		_logger.LogInformation($"STD              : {stats.StandardDeviation:0.00}");
	}
}
