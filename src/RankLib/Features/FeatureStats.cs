using MathNet.Numerics.Statistics;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Learning;
using RankLib.Learning.Boosting;
using RankLib.Learning.NeuralNet;
using RankLib.Learning.Tree;
using RankLib.Utilities;

namespace RankLib.Features;

public class FeatureStats
{
	private readonly ILogger<FeatureStats> _logger;
	private readonly string _modelFileName;
	private static readonly string[] ModelsThatUseAllFeatures = [CoordinateAscent.RankerName, LambdaRank.RankerName, LinearRegression.RankerName, ListNet.RankerName, RankNet.RankerName];
	private static readonly string[] FeatureWeightModels = [AdaRank.RankerName, RankBoost.RankerName];
	private static readonly string[] TreeModels = [LambdaMART.RankerName, MART.RankerName, RandomForests.RankerName];

	/// <summary>
	/// Instantiates a new instance of <see cref="FeatureStats"/>
	/// </summary>
	/// <param name="modelFile">The path of the model file to load the ranker from.</param>
	/// <param name="logger">Logger to log events</param>
	public FeatureStats(string modelFile, ILogger<FeatureStats>? logger = null)
	{
		_logger = logger ?? NullLogger<FeatureStats>.Instance;
		var file = new FileInfo(modelFile);
		_modelFileName = file.FullName;
	}

	private static SortedDictionary<int, int> GetFeatureWeightFeatureFrequencies(StreamReader reader)
	{
		var featureFrequencies = new SortedDictionary<int, int>();

		try
		{
			while (reader.ReadLine() is { } line)
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

	private static SortedDictionary<int, int> GetTreeFeatureFrequencies(StreamReader reader)
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

	/// <summary>
	/// Writes the feature statistics to the logger specified in the constructor.
	/// </summary>
	/// <exception cref="RankLibException">Exception reading model file</exception>
	public void WriteFeatureStats()
	{
		SortedDictionary<int, int>? featureFrequencies;
		string? modelName;

		try
		{
			using var reader = new StreamReader(_modelFileName);

			// Read model name from the file
			var modelLine = reader.ReadLine().AsSpan();
			modelName = modelLine.TrimStart('#').Trim().ToString();

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
			// Anything else
			else
				throw RankLibException.Create($"Expected to find model name on first line, but found {modelLine}");
		}
		catch (IOException exception)
		{
			throw RankLibException.Create($"Error reading file {_modelFileName}: {exception.Message}", exception);
		}

		// Calculate feature statistics
		var featuresUsed = featureFrequencies.Count;

		_logger.LogInformation("Model File: {ModelFileName}", _modelFileName);
		_logger.LogInformation("Algorithm: {ModelName}", modelName);
		_logger.LogInformation("Feature frequencies:");

		var data = new List<double>(featuresUsed);
		foreach (var (featureId, freq) in featureFrequencies)
		{
			_logger.LogInformation("\tFeature[{FeatureId}] : {Freq}", featureId, freq);
			data.Add(freq);
		}

		var stats = new DescriptiveStatistics(data);
		_logger.LogInformation("Total Features Used: {FeaturesUsed}", featuresUsed);
		_logger.LogInformation($"Min frequency    : {stats.Minimum:0.00}");
		_logger.LogInformation($"Max frequency    : {stats.Maximum:0.00}");
		_logger.LogInformation($"Median frequency : {data.Median():0.00}");
		_logger.LogInformation($"Avg frequency    : {stats.Mean:0.00}");
		_logger.LogInformation($"Variance         : {stats.Variance:0.00}");
		_logger.LogInformation($"STD              : {stats.StandardDeviation:0.00}");
	}
}
