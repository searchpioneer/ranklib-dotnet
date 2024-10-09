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
		var samples = new List<RankList>(1000);
		var countEntries = 0;

		try
		{
			using var reader = FileUtils.SmartReader(inputFile);
			var lastId = "";
			var hasRel = false;
			var rl = new List<DataPoint>(10000);

			while (reader.ReadLine() is { } content)
			{
				content = content.Trim();
				if (content.Length == 0 || content[0] == '#')
				{
					continue;
				}

				if (countEntries % 10000 == 0)
				{
					_logger.LogInformation("Reading feature file [{InputFile}]: {CountEntries}...", inputFile, countEntries);
				}

				DataPoint qp = useSparseRepresentation
					? new SparseDataPoint(content)
					: new DenseDataPoint(content);

				if (!string.IsNullOrEmpty(lastId) && !lastId.Equals(qp.Id, StringComparison.OrdinalIgnoreCase))
				{
					if (!mustHaveRelDoc || hasRel)
					{
						samples.Add(new RankList(rl));
					}
					rl = new List<DataPoint>();
					hasRel = false;
				}

				if (qp.Label > 0)
				{
					hasRel = true;
				}

				lastId = qp.Id;
				rl.Add(qp);
				countEntries++;
			}

			if (rl.Any() && (!mustHaveRelDoc || hasRel))
			{
				samples.Add(new RankList(rl));
			}

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
		var samples = new List<RankList>();
		foreach (var inputFile in inputFiles)
		{
			var s = ReadInput(inputFile, false, false);
			samples.AddRange(s);
		}
		return samples;
	}

	public int[] ReadFeature(string featureDefFile)
	{
		int[] features;
		var fids = new List<string>();

		try
		{
			using (var inStream = FileUtils.SmartReader(featureDefFile))
			{
				while (inStream.ReadLine() is { } content)
				{
					content = content.Trim();
					if (content.Length == 0 || content[0] == '#')
					{
						continue;
					}
					fids.Add(content.Split('\t')[0].Trim());
				}
			}

			features = fids.Select(int.Parse).ToArray();
		}
		catch (IOException ex)
		{
			throw RankLibException.Create("Error in FeatureManager::readFeature(): ", ex);
		}

		return features;
	}

	public int[] GetFeatureFromSampleVector(List<RankList> samples)
	{
		if (!samples.Any())
		{
			throw RankLibException.Create("Error in FeatureManager::getFeatureFromSampleVector(): There are no training samples.");
		}

		var maxFeatureCount = samples.Max(rl => rl.FeatureCount);
		var features = Enumerable.Range(1, maxFeatureCount).ToArray();

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
			trainSamplesIdx.Last().Add(total++);
		}

		foreach (var indexes in trainSamplesIdx)
		{
			_logger.LogInformation($"Creating data for fold {trainSamplesIdx.IndexOf(indexes) + 1}/{nFold}...");
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

		_logger.LogInformation($"Creating data for {nFold} folds completed.");
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
			using var outStream = new StreamWriter(outputFile);
			foreach (var sample in samples)
			{
				Save(sample, outStream);
			}
		}
		catch (Exception ex)
		{
			throw RankLibException.Create("Error in FeatureManager::save(): ", ex);
		}
	}

	private static void Save(RankList rankList, StreamWriter outStream)
	{
		for (var i = 0; i < rankList.Count; i++)
		{
			var dataPoint = rankList[i];
			outStream.WriteLine(dataPoint.ToString());
		}
	}
}
