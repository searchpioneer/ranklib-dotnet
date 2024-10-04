using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using System.Text.RegularExpressions;
using MathNet.Numerics.Statistics;

namespace RankLib.Features;

public partial class FeatureStats
{
    [GeneratedRegex(@"<feature>(\d+)</feature>")]
    private static partial Regex FeatureIdRegex();
    
    // TODO: logging
    private static readonly ILogger logger = NullLogger.Instance;
    
    private string _modelName;
    private readonly string _modelFileName;
    private readonly FileInfo _file;

    public FeatureStats(string modelFileName)
    {
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
                line = line.Trim().ToLower();

                if (string.IsNullOrWhiteSpace(line) || line.Contains("##"))
                    continue;

                var featureLines = line.Split(" ");
                foreach (var featureLine in featureLines)
                {
                    var featureId = int.Parse(featureLine.Split(":")[0]);

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

    private SortedDictionary<int, int> GetTreeFeatureFrequencies(StreamReader sr)
    {
        var featureFrequencies = new SortedDictionary<int, int>();

        try
        {
            while (sr.ReadLine() is { } line)
            {
                line = line.Trim().ToLower();

                if (string.IsNullOrWhiteSpace(line) || line.Contains("##"))
                    continue;

                if (line.Contains("<feature>"))
                {
                    var featureIdStr = FeatureIdRegex().Match(line).Groups[1].Value;
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
        SortedDictionary<int, int> featureFrequencies = null;

        try
        {
            using var sr = new StreamReader(_file.FullName);
            
            // Read model name from the file
            var modelLine = sr.ReadLine()?.Trim();
            var nameParts = modelLine?.Split(" ");
            var len = nameParts?.Length ?? 0;

            if (len == 2)
            {
                _modelName = nameParts![1].Trim();
            }
            else if (len == 3)
            {
                _modelName = $"{nameParts![1].Trim()} {nameParts[2].Trim()}";
            }

            if (string.IsNullOrEmpty(_modelName))
            {
                throw new Exception("No model name defined. Quitting.");
            }

            // Handle models that use all features
            if (new[] { "Coordinate Ascent", "LambdaRank", "Linear Regression", "ListNet", "RankNet" }.Contains(_modelName))
            {
                logger.LogInformation("{ModelName} uses all features. Can't do selected model statistics for this algorithm.", _modelName);
                return;
            }

            // Feature:Weight models
            if (new[] { "AdaRank", "RankBoost" }.Contains(_modelName))
            {
                featureFrequencies = GetFeatureWeightFeatureFrequencies(sr);
            }
            // Tree models
            else if (new[] { "LambdaMART", "MART", "Random Forests" }.Contains(_modelName))
            {
                featureFrequencies = GetTreeFeatureFrequencies(sr);
            }
        }
        catch (IOException ioe)
        {
            throw new Exception($"IOException on file {_modelFileName}: {ioe.Message}", ioe);
        }

        // Calculate feature statistics
        var featuresUsed = featureFrequencies?.Count ?? 0;

        logger.LogInformation("Model File: {ModelFileName}", _modelFileName);
        logger.LogInformation("Algorithm: {ModelName}", _modelName);
        logger.LogInformation("Feature frequencies:");

        var data = new List<double>(featuresUsed);
        
        foreach (var entry in featureFrequencies)
        {
            var featureId = entry.Key;
            var freq = entry.Value;
            logger.LogInformation("\tFeature[{FeatureId}] : {Freq}", featureId, freq);
            data.Add(freq);
        }

        var stats = new DescriptiveStatistics(data);

        // Print out summary statistics
        logger.LogInformation($"Total Features Used: {featuresUsed}");
        logger.LogInformation($"Min frequency    : {stats.Minimum:0.00}");
        logger.LogInformation($"Max frequency    : {stats.Maximum:0.00}");
        //logger.LogInformation($"Median frequency : {stats.Median:0.00}");
        logger.LogInformation($"Avg frequency    : {stats.Mean:0.00}");
        logger.LogInformation($"Variance         : {stats.Variance:0.00}");
        logger.LogInformation($"STD              : {stats.StandardDeviation:0.00}");
    }


}
