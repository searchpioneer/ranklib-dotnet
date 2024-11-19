using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Learning;
using RankLib.Utilities;

namespace RankLib.Features;

/// <summary>
/// Manages rank lists and features, including reading from and writing to file,
/// and preparing rank lists for learning and cross validation.
/// </summary>
public class FeatureManager
{
	private readonly ILogger<FeatureManager> _logger;

	/// <summary>
	/// Instantiates a new instance of <see cref="FeatureManager"/>
	/// </summary>
	/// <param name="logger">Logger to log messages</param>
	public FeatureManager(ILogger<FeatureManager>? logger = null) =>
		_logger = logger ?? NullLogger<FeatureManager>.Instance;

	/// <summary>
	/// Read a list of rank lists from the specified <paramref name="inputFile"/>
	/// </summary>
	/// <param name="inputFile">The file containing rank lists</param>
	/// <param name="mustHaveRelevantDocument">Whether to ignore rank lists that do not have
	/// any relevant documents. The default is <c>false</c>
	/// </param>
	/// <param name="useSparseRepresentation">
	/// Whether data points use a sparse representation.
	/// The default is <c>false</c>, resulting in data points with a dense representation</param>
	/// <returns>A new instance of a list of <see cref="RankList"/></returns>
	public List<RankList> ReadInput(string inputFile, bool mustHaveRelevantDocument = false, bool useSparseRepresentation = false)
	{
		var rankLists = new List<RankList>();
		var countEntries = 0;

		try
		{
			using var reader = SmartReader.OpenText(inputFile);
			var lastId = string.Empty;
			var hasRelevantDocument = false;
			var dataPoints = new List<DataPoint>();

			while (reader.ReadLine() is { } content)
			{
				var contentSpan = content.AsSpan().Trim();
				if (contentSpan.IsEmpty || contentSpan[0] == '#')
					continue;

				if (countEntries % 10000 == 0)
					_logger.LogInformation("Reading feature file [{InputFile}]: {CountEntries}...", inputFile, countEntries);

				DataPoint dataPoint = useSparseRepresentation
					? new SparseDataPoint(contentSpan)
					: new DenseDataPoint(contentSpan);

				if (!string.IsNullOrEmpty(lastId) && !lastId.Equals(dataPoint.Id, StringComparison.OrdinalIgnoreCase))
				{
					if (!mustHaveRelevantDocument || hasRelevantDocument)
						rankLists.Add(new RankList(dataPoints));

					dataPoints = new List<DataPoint>();
					hasRelevantDocument = false;
				}

				if (dataPoint.Label > 0)
					hasRelevantDocument = true;

				lastId = dataPoint.Id;
				dataPoints.Add(dataPoint);
				countEntries++;
			}

			if (dataPoints.Count > 0 && (!mustHaveRelevantDocument || hasRelevantDocument))
				rankLists.Add(new RankList(dataPoints));

			_logger.LogInformation(
				"Reading feature file [{InputFile}] completed. (Read {SamplesCount} ranked lists, {CountEntries} entries)",
				inputFile,
				rankLists.Count,
				countEntries);
		}
		catch (Exception ex)
		{
			throw RankLibException.Create("Error reading ranklists from file", ex);
		}

		return rankLists;
	}

	/// <summary>
	/// Read a list of rank lists from the specified <paramref name="inputFiles"/> and merge them together
	/// into a single list of rank lists.
	/// </summary>
	/// <param name="inputFiles">The files containing rank lists</param>
	/// <param name="mustHaveRelevantDocument">Whether to ignore rank lists that do not have
	/// any relevant documents. The default is <c>false</c>
	/// </param>
	/// <param name="useSparseRepresentation">
	/// Whether data points use a sparse representation.
	/// The default is <c>false</c>, resulting in data points with a dense representation</param>
	/// <returns>A new instance of a list of <see cref="RankList"/></returns>
	public List<RankList> ReadInput(List<string> inputFiles, bool mustHaveRelevantDocument = false, bool useSparseRepresentation = false)
	{
		var rankLists = new List<RankList>();
		foreach (var inputFile in inputFiles)
		{
			var rankList = ReadInput(inputFile, mustHaveRelevantDocument, useSparseRepresentation);
			rankLists.AddRange(rankList);
		}
		return rankLists;
	}

	/// <summary>
	/// Read features from the specified <paramref name="featureDefinitionFile"/>. There must be one
	/// feature per line.
	/// </summary>
	/// <param name="featureDefinitionFile">The file containing the feature definitions</param>
	/// <returns>A new instance of an array of features</returns>
	/// <exception cref="ArgumentException">The file is not a valid feature definition file</exception>
	/// <exception cref="RankLibException">There was an error reading the feature definition file</exception>
	public int[] ReadFeature(string featureDefinitionFile)
	{
		var featureIds = new List<int>();
		try
		{
			using var reader = SmartReader.OpenText(featureDefinitionFile);
			while (reader.ReadLine() is { } content)
			{
				var contentSpan = content.AsSpan().Trim();
				if (contentSpan.IsEmpty || contentSpan[0] == '#')
					continue;

				var firstTab = contentSpan.IndexOf('\t');
				if (firstTab == -1)
					throw new ArgumentException("feature definition file is not valid", nameof(featureDefinitionFile));

				var featureId = contentSpan.Slice(0, firstTab).Trim();
				featureIds.Add(int.Parse(featureId));
			}
		}
		catch (IOException ex)
		{
			throw RankLibException.Create("Error reading features", ex);
		}

		return featureIds.ToArray();
	}

	public int[] GetFeatureFromSampleVector(List<RankList> samples)
	{
		if (samples.Count == 0)
			throw RankLibException.Create("samples is empty");

		var maxFeatureCount = samples.Max(rl => rl.FeatureCount);
		var features = new int[maxFeatureCount];

		for (var i = 1; i <= maxFeatureCount; i++)
			features[i - 1] = i;

		return features;
	}

	public void PrepareCrossValidation(
		List<RankList> samples,
		int foldCount,
		List<List<RankList>> trainingData,
		List<List<RankList>> testData) =>
		PrepareCrossValidation(samples, foldCount, -1, trainingData, [], testData);

	public void PrepareCrossValidation(
		List<RankList> samples,
		int foldCount,
		float trainValidationSplit,
		List<List<RankList>> trainingData,
		List<List<RankList>> validationData,
		List<List<RankList>> testData)
	{
		var trainSamplesIdx = new List<List<int>>();
		var size = samples.Count / foldCount;
		var start = 0;
		var total = 0;

		for (var f = 0; f < foldCount; f++)
		{
			var foldIndexes = new List<int>();
			for (var i = 0; i < size && start + i < samples.Count; i++)
				foldIndexes.Add(start + i);

			trainSamplesIdx.Add(foldIndexes);
			total += foldIndexes.Count;
			start += size;
		}

		while (total < samples.Count)
			trainSamplesIdx[^1].Add(total++);

		for (var idx = 0; idx < trainSamplesIdx.Count; idx++)
		{
			var indexes = trainSamplesIdx[idx];
			_logger.LogInformation("Creating data for fold {TrainSamplesIdx}/{FoldCount}...", idx + 1, foldCount);
			var train = new List<RankList>();
			var test = new List<RankList>();
			var validation = new List<RankList>();

			foreach (var index in Enumerable.Range(0, samples.Count))
			{
				if (indexes.Contains(index))
					test.Add(new RankList(samples[index]));
				else
					train.Add(new RankList(samples[index]));
			}

			if (trainValidationSplit > 0)
			{
				var validationSize = (int)(train.Count * (1.0f - trainValidationSplit));
				for (var i = 0; i < validationSize; i++)
				{
					validation.Add(train.Last());
					train.RemoveAt(train.Count - 1);
				}
			}

			trainingData.Add(train);
			testData.Add(test);

			if (trainValidationSplit > 0)
				validationData.Add(validation);
		}

		_logger.LogInformation("Creating data for {FoldCount} folds completed.", foldCount);
		PrintQueriesForSplit("Train", trainingData);
		PrintQueriesForSplit("Validate", validationData);
		PrintQueriesForSplit("Test", testData);
	}

	private void PrintQueriesForSplit(string name, List<List<RankList>>? split)
	{
		if (split == null)
		{
			_logger.LogInformation("No {Name} split.", name);
			return;
		}

		foreach (var rankLists in split)
		{
			_logger.LogInformation("{Name} [{RankListIndex}] = ", name, split.IndexOf(rankLists));
			foreach (var rankList in rankLists)
				_logger.LogInformation(" \"{RankListId}\"", rankList.Id);
		}
	}

	public void PrepareSplit(List<RankList> samples, double percentTrain, List<RankList> trainingData, List<RankList> testData)
	{
		if (percentTrain is < 0 or > 1)
			throw new ArgumentException("percent train must be between 0 and 1.", nameof(percentTrain));

		var size = (int)(samples.Count * percentTrain);

		for (var i = 0; i < size; i++)
			trainingData.Add(new RankList(samples[i]));

		for (var i = size; i < samples.Count; i++)
			testData.Add(new RankList(samples[i]));
	}

	public void Save(List<RankList> samples, string outputFile)
	{
		try
		{
			using var writer = new StreamWriter(outputFile);
			foreach (var sample in samples)
				Save(sample, writer);
		}
		catch (Exception ex)
		{
			throw RankLibException.Create("Error saving rank lists to file", ex);
		}
	}

	private static void Save(RankList rankList, StreamWriter writer)
	{
		for (var i = 0; i < rankList.Count; i++)
		{
			var dataPoint = rankList[i];
			writer.WriteLine(dataPoint.ToString());
		}
	}
}
