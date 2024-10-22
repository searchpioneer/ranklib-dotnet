using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Learning;
using RankLib.Utilities;

namespace RankLib.Features;

public class FeatureManager
{
	private readonly ILogger<FeatureManager> _logger;

	public FeatureManager(ILogger<FeatureManager>? logger = null) =>
		_logger = logger ?? NullLogger<FeatureManager>.Instance;

	public List<RankList> ReadInput(string inputFile) => ReadInput(inputFile, false, false);

	public List<RankList> ReadInput(string inputFile, bool mustHaveRelDoc, bool useSparseRepresentation)
	{
		var samples = new List<RankList>();
		var countEntries = 0;

		try
		{
			using var reader = FileUtils.SmartReader(inputFile);
			var lastId = string.Empty;
			var hasRel = false;
			var dataPoints = new List<DataPoint>();

			while (reader.ReadLine() is { } content)
			{
				var contentSpan = content.AsSpan().Trim();

				if (contentSpan.IsEmpty || contentSpan[0] == '#')
					continue;

				if (countEntries % 10000 == 0)
					_logger.LogInformation("Reading feature file [{InputFile}]: {CountEntries}...", inputFile, countEntries);

				DataPoint dataPoint = useSparseRepresentation
					? new SparseDataPoint(contentSpan.ToString())
					: new DenseDataPoint(contentSpan.ToString());

				if (!string.IsNullOrEmpty(lastId) && !lastId.Equals(dataPoint.Id, StringComparison.OrdinalIgnoreCase))
				{
					if (!mustHaveRelDoc || hasRel)
						samples.Add(new RankList(dataPoints));

					dataPoints = new List<DataPoint>();
					hasRel = false;
				}

				if (dataPoint.Label > 0)
					hasRel = true;

				lastId = dataPoint.Id;
				dataPoints.Add(dataPoint);
				countEntries++;
			}

			if (dataPoints.Count != 0 && (!mustHaveRelDoc || hasRel))
				samples.Add(new RankList(dataPoints));

			_logger.LogInformation(
				"Reading feature file [{InputFile}] completed. (Read {SamplesCount} ranked lists, {CountEntries} entries)",
				inputFile,
				samples.Count,
				countEntries);
		}
		catch (Exception ex)
		{
			throw RankLibException.Create("Error reading samples from file", ex);
		}

		return samples;
	}

	public List<RankList> ReadInput(List<string> inputFiles)
	{
		var rankLists = new List<RankList>();
		foreach (var inputFile in inputFiles)
		{
			var rankList = ReadInput(inputFile, false, false);
			rankLists.AddRange(rankList);
		}
		return rankLists;
	}

	public int[] ReadFeature(string featureDefFile)
	{
		var fids = new List<int>();
		try
		{
			using var reader = FileUtils.SmartReader(featureDefFile);
			while (reader.ReadLine() is { } content)
			{
				var contentSpan = content.AsSpan().Trim();
				if (contentSpan.IsEmpty || contentSpan[0] == '#')
				{
					continue;
				}

				var firstTab = contentSpan.IndexOf('\t');
				if (firstTab == -1)
				{
					throw new ArgumentException("featureDefFile is not a valid feature file.", nameof(featureDefFile));
				}

				var fid = contentSpan.Slice(0, firstTab).Trim();
				fids.Add(int.Parse(fid));
			}
		}
		catch (IOException ex)
		{
			throw RankLibException.Create("Error in FeatureManager::readFeature(): ", ex);
		}

		return fids.ToArray();
	}

	public int[] GetFeatureFromSampleVector(List<RankList> samples)
	{
		if (samples.Count == 0)
			throw RankLibException.Create("samples is empty");

		var maxFeatureCount = samples.Max(rl => rl.FeatureCount);
		var features = new int[maxFeatureCount];

		for(var i=1; i <= maxFeatureCount; i++)
			features[i-1] = i;

		return features;
	}

	public void PrepareCV(List<RankList> samples, int nFold, List<List<RankList>> trainingData, List<List<RankList>> testData) =>
		PrepareCV(samples, nFold, -1, trainingData, [], testData);

	public void PrepareCV(List<RankList> samples, int nFold, float tvs, List<List<RankList>> trainingData, List<List<RankList>> validationData, List<List<RankList>> testData)
	{
		var trainSamplesIdx = new List<List<int>>();
		var size = samples.Count / nFold;
		var start = 0;
		var total = 0;

		for (var f = 0; f < nFold; f++)
		{
			var foldIndexes = new List<int>();
			for (var i = 0; i < size && start + i < samples.Count; i++)
			{
				foldIndexes.Add(start + i);
			}
			trainSamplesIdx.Add(foldIndexes);
			total += foldIndexes.Count;
			start += size;
		}

		while (total < samples.Count)
		{
			trainSamplesIdx[^1].Add(total++);
		}

		for (var idx = 0; idx < trainSamplesIdx.Count; idx++)
		{
			var indexes = trainSamplesIdx[idx];
			_logger.LogInformation("Creating data for fold {TrainSamplesIdx}/{NFold}...", idx + 1, nFold);
			var train = new List<RankList>();
			var test = new List<RankList>();
			var validation = new List<RankList>();

			foreach (var index in Enumerable.Range(0, samples.Count))
			{
				if (indexes.Contains(index))
				{
					test.Add(new RankList(samples[index]));
				}
				else
				{
					train.Add(new RankList(samples[index]));
				}
			}

			if (tvs > 0)
			{
				var validationSize = (int)(train.Count * (1.0f - tvs));
				for (var i = 0; i < validationSize; i++)
				{
					validation.Add(train.Last());
					train.RemoveAt(train.Count - 1);
				}
			}

			trainingData.Add(train);
			testData.Add(test);

			if (tvs > 0)
			{
				validationData.Add(validation);
			}
		}

		_logger.LogInformation("Creating data for {NFold} folds completed.", nFold);
		PrintQueriesForSplit("Train", trainingData);
		PrintQueriesForSplit("Validate", validationData);
		PrintQueriesForSplit("Test", testData);
	}

	public void PrintQueriesForSplit(string name, List<List<RankList>>? split)
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
			{
				_logger.LogInformation(" \"{RankListId}\"", rankList.Id);
			}
		}
	}

	public void PrepareSplit(List<RankList> samples, double percentTrain, List<RankList> trainingData, List<RankList> testData)
	{
		if (percentTrain is < 0 or > 1)
		{
			throw new ArgumentException("percentTrain must be between 0 and 1.", nameof(percentTrain));
		}

		var size = (int)(samples.Count * percentTrain);

		for (var i = 0; i < size; i++)
		{
			trainingData.Add(new RankList(samples[i]));
		}

		for (var i = size; i < samples.Count; i++)
		{
			testData.Add(new RankList(samples[i]));
		}
	}

	public void Save(List<RankList> samples, string outputFile)
	{
		try
		{
			using var writer = new StreamWriter(outputFile);
			foreach (var sample in samples)
			{
				Save(sample, writer);
			}
		}
		catch (Exception ex)
		{
			throw RankLibException.Create("Error in FeatureManager::save(): ", ex);
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
