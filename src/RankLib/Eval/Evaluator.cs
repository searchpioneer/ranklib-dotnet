using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Features;
using RankLib.Learning;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Eval;

/// <summary>
/// Evaluates rankers and rank lists.
/// Rankers can be trained and optionally saved to a model file, or loaded from a model file, and evaluated.
/// </summary>
public class Evaluator
{
	private readonly ILogger<Evaluator> _logger;
	private readonly RankerFactory _rankerFactory;
	private readonly MetricScorer _trainScorer;
	private readonly MetricScorer _testScorer;
	private readonly RankerTrainer _trainer;
	private readonly bool _mustHaveRelevantDocument;
	private readonly bool _useSparseRepresentation;
	private readonly FeatureManager _featureManager;
	private readonly bool _normalize;
	private readonly Normalizer _normalizer;

	/// <summary>
	/// Initializes a new instance of <see cref="Evaluator"/>
	/// </summary>
	/// <param name="rankerFactory">The factory for creating rankers</param>
	/// <param name="featureManager">The feature manager</param>
	/// <param name="trainScorer">The retrieval metric score used for training</param>
	/// <param name="testScorer">The retrieval metric score used for testing</param>
	/// <param name="trainer">The ranker trainer</param>
	/// <param name="normalizer">The normalizer</param>
	/// <param name="mustHaveRelevantDocument"></param>
	/// <param name="useSparseRepresentation"></param>
	/// <param name="logger"></param>
	public Evaluator(
		RankerFactory rankerFactory,
		FeatureManager featureManager,
		MetricScorer trainScorer,
		MetricScorer testScorer,
		RankerTrainer trainer,
		Normalizer? normalizer = null,
		bool mustHaveRelevantDocument = false,
		bool useSparseRepresentation = false,
		ILogger<Evaluator>? logger = null
	)
	{
		_rankerFactory = rankerFactory;
		_featureManager = featureManager;
		_trainScorer = trainScorer;
		_testScorer = testScorer;
		_trainer = trainer;
		_mustHaveRelevantDocument = mustHaveRelevantDocument;
		_useSparseRepresentation = useSparseRepresentation;
		_logger = logger ?? NullLogger<Evaluator>.Instance;
		_normalize = normalizer != null;
		_normalizer = normalizer ?? SumNormalizer.Instance;
	}

	public Evaluator(
		RankerFactory rankerFactory,
		FeatureManager featureManager,
		MetricScorer scorer,
		RankerTrainer trainer,
		Normalizer? normalizer = null,
		bool mustHaveRelevantDocument = false,
		bool useSparseRepresentation = false,
		ILogger<Evaluator>? logger = null
	)
	{
		_rankerFactory = rankerFactory;
		_featureManager = featureManager;
		_trainScorer = scorer;
		_testScorer = scorer;
		_trainer = trainer;
		_mustHaveRelevantDocument = mustHaveRelevantDocument;
		_useSparseRepresentation = useSparseRepresentation;
		_logger = logger ?? NullLogger<Evaluator>.Instance;
		_normalize = normalizer != null;
		_normalizer = normalizer ?? SumNormalizer.Instance;
	}

	private List<RankList> ReadInput(string inputFile) =>
		_featureManager.ReadInput(inputFile, _mustHaveRelevantDocument, _useSparseRepresentation);

	private void Normalize(List<RankList> samples, int[] featureIds)
	{
		foreach (var sample in samples)
			_normalizer.Normalize(sample, featureIds);
	}

	private void NormalizeAll(List<List<RankList>> samples, int[] featureIds)
	{
		foreach (var sample in samples)
			Normalize(sample, featureIds);
	}

	private int[]? ReadFeature(string? featureDefFile) =>
		string.IsNullOrEmpty(featureDefFile) ? null : _featureManager.ReadFeature(featureDefFile);

	public double Evaluate(IRanker? ranker, List<RankList> rankLists)
	{
		var rankedList = ranker != null ? ranker.Rank(rankLists) : rankLists;
		return _testScorer.Score(rankedList);
	}

	public async Task EvaluateAsync(
		Type rankerType,
		string trainFile,
		string? validationFile,
		string? testFile,
		string? featureDefinitionFile,
		string? modelFile = null,
		IRankerParameters? parameters = default)
	{
		if (!typeof(IRanker).IsAssignableFrom(rankerType))
			throw new ArgumentException($"Ranker type {rankerType} is not a ranker");

		var train = ReadInput(trainFile);
		var validation = !string.IsNullOrEmpty(validationFile) ? ReadInput(validationFile) : null;
		var test = !string.IsNullOrEmpty(testFile) ? ReadInput(testFile) : null;
		var features = ReadFeature(featureDefinitionFile) ?? _featureManager.GetFeatureFromSampleVector(train);

		if (_normalize)
		{
			Normalize(train, features);
			if (validation != null)
				Normalize(validation, features);
			if (test != null)
				Normalize(test, features);
		}

		var ranker = await _trainer.TrainAsync(rankerType, train, validation, features, _trainScorer, parameters)
			.ConfigureAwait(false);

		if (test != null)
		{
			var rankScore = Evaluate(ranker, test);
			_logger.LogInformation($"{_testScorer.Name} on test data: {Math.Round(rankScore, 4)}");
		}

		if (!string.IsNullOrEmpty(modelFile))
		{
			await ranker.SaveAsync(modelFile);
			_logger.LogInformation("Model saved to: {ModelFile}", modelFile);
		}
	}

	public Task EvaluateAsync<TRanker, TRankerParameters>(
		string trainFile,
		string? validationFile,
		string? testFile,
		string? featureDefinitionFile,
		string? modelFile = null,
		TRankerParameters? parameters = default)
		where TRanker : IRanker<TRankerParameters>
		where TRankerParameters : IRankerParameters =>
		EvaluateAsync(typeof(TRanker), trainFile, validationFile, testFile, featureDefinitionFile, modelFile, parameters);

	public async Task EvaluateAsync(
		Type rankerType,
		string sampleFile,
		string? validationFile,
		string? featureDefinitionFile,
		double percentTrain,
		string? modelFile = null,
		IRankerParameters? parameters = default)
	{
		var train = new List<RankList>();
		var test = new List<RankList>();
		var features = PrepareSplit(sampleFile, featureDefinitionFile, percentTrain, _normalize, train, test);
		var validation = !string.IsNullOrEmpty(validationFile) ? ReadInput(validationFile) : null;

		if (_normalize && validation != null)
			Normalize(validation, features);

		var ranker = await _trainer.TrainAsync(rankerType, train, validation, features, _trainScorer, parameters)
			.ConfigureAwait(false);

		var rankScore = Evaluate(ranker, test);
		_logger.LogInformation($"{_testScorer.Name} on test data: {Math.Round(rankScore, 4)}");

		if (!string.IsNullOrEmpty(modelFile))
		{
			await ranker.SaveAsync(modelFile);
			_logger.LogInformation("Model saved to: {ModelFile}", modelFile);
		}
	}

	public Task EvaluateAsync<TRanker, TRankerParameters>(
		string sampleFile,
		string? validationFile,
		string featureDefinitionFile,
		double percentTrain,
		string? modelFile = null,
		TRankerParameters? parameters = default)
		where TRanker : IRanker<TRankerParameters>
		where TRankerParameters : IRankerParameters =>
		EvaluateAsync(typeof(TRanker), sampleFile, validationFile, featureDefinitionFile, percentTrain, modelFile, parameters);

	public async Task EvaluateAsync(
		Type rankerType,
		string trainFile,
		double percentTrain,
		string? testFile,
		string? featureDefinitionFile,
		string? modelFile = null,
		IRankerParameters? parameters = default)
	{
		var train = new List<RankList>();
		var validation = new List<RankList>();
		var features = PrepareSplit(trainFile, featureDefinitionFile, percentTrain, _normalize, train, validation);
		var test = !string.IsNullOrEmpty(testFile) ? ReadInput(testFile) : null;

		if (_normalize && test != null)
			Normalize(test, features);

		var ranker = await _trainer.TrainAsync(rankerType, train, validation, features, _trainScorer, parameters)
			.ConfigureAwait(false);

		if (test != null)
		{
			var rankScore = Evaluate(ranker, test);
			_logger.LogInformation($"{_testScorer.Name} on test data: {Math.Round(rankScore, 4)}");
		}

		if (!string.IsNullOrEmpty(modelFile))
		{
			await ranker.SaveAsync(modelFile);
			_logger.LogInformation("Model saved to: {ModelFile}", modelFile);
		}
	}

	public Task EvaluateAsync<TRanker, TRankerParameters>(
		string trainFile,
		double percentTrain,
		string? testFile,
		string featureDefinitionFile,
		string? modelFile = null,
		TRankerParameters? parameters = default)
		where TRanker : IRanker<TRankerParameters>
		where TRankerParameters : IRankerParameters =>
		EvaluateAsync(typeof(TRanker), trainFile, percentTrain, testFile, featureDefinitionFile, modelFile, parameters);

	public Task EvaluateAsync<TRanker, TRankerParameters>(
		string sampleFile,
		string featureDefinitionFile,
		int foldCount,
		string modelDir,
		string modelFile,
		TRankerParameters? parameters = default)
		where TRanker : IRanker<TRankerParameters>
		where TRankerParameters : IRankerParameters =>
		EvaluateAsync(typeof(TRanker), sampleFile, featureDefinitionFile, foldCount, -1, modelDir, modelFile, parameters);

	public async Task EvaluateAsync(
		Type rankerType,
		string sampleFile,
		string? featureDefinitionFile,
		int foldCount,
		float tvs,
		string modelDir,
		string modelFile,
		IRankerParameters? parameters = default)
	{
		var trainingData = new List<List<RankList>>();
		var validationData = new List<List<RankList>>();
		var testData = new List<List<RankList>>();
		var samples = ReadInput(sampleFile);
		var features = ReadFeature(featureDefinitionFile) ?? _featureManager.GetFeatureFromSampleVector(samples);

		_featureManager.PrepareCV(samples, foldCount, tvs, trainingData, validationData, testData);

		if (_normalize)
		{
			for (var i = 0; i < foldCount; i++)
			{
				NormalizeAll(trainingData, features);
				NormalizeAll(validationData, features);
				NormalizeAll(testData, features);
			}
		}

		var scoreOnTrain = 0.0;
		var scoreOnTest = 0.0;
		var totalScoreOnTest = 0.0;
		var totalTestSampleSize = 0;
		var scores = new double[foldCount][];

		for (var i = 0; i < foldCount; i++)
		{
			scores[i] = new double[2];
			var train = trainingData[i];
			var validation = tvs > 0 ? validationData[i] : null;
			var test = testData[i];

			var ranker = await _trainer.TrainAsync(rankerType, train, validation, features, _trainScorer, parameters)
				.ConfigureAwait(false);

			var testScore = Evaluate(ranker, test);
			scoreOnTrain += ranker.GetTrainingDataScore();
			scoreOnTest += testScore;
			totalScoreOnTest += testScore * test.Count;
			totalTestSampleSize += test.Count;

			scores[i][0] = ranker.GetTrainingDataScore();
			scores[i][1] = testScore;

			if (!string.IsNullOrEmpty(modelDir))
			{
				var foldModelFile = Path.Combine(modelDir, $"f{i + 1}.{modelFile}");
				await ranker.SaveAsync(foldModelFile);
				_logger.LogInformation($"Fold-{i + 1} model saved to: {foldModelFile}");
			}
		}

		_logger.LogInformation("Summary:");
		_logger.LogInformation("{Scorer}\t|   Train\t| Test", _testScorer.Name);

		for (var i = 0; i < foldCount; i++)
			_logger.LogInformation($"Fold {i + 1}\t|   {Math.Round(scores[i][0], 4)}\t|  {Math.Round(scores[i][1], 4)}\t");

		_logger.LogInformation($"Avg.\t|   {Math.Round(scoreOnTrain / foldCount, 4)}\t|  {Math.Round(scoreOnTest / foldCount, 4)}\t");
		_logger.LogInformation($"Total\t|   \t\t|  {Math.Round(totalScoreOnTest / totalTestSampleSize, 4)}\t");
	}

	public Task EvaluateAsync<TRanker, TRankerParameters>(
		string sampleFile,
		string? featureDefinitionFile,
		int foldCount,
		float tvs,
		string modelDir,
		string modelFile,
		TRankerParameters? parameters = default)
		where TRanker : IRanker<TRankerParameters>
		where TRankerParameters : IRankerParameters =>
		EvaluateAsync(typeof(TRanker), sampleFile, featureDefinitionFile, foldCount, tvs, modelDir, modelFile, parameters);

	public void Test(string testFile)
	{
		var test = ReadInput(testFile);
		var rankScore = Evaluate(null, test);
		_logger.LogInformation("{TestScorerName} on test data: {RoundedRankScore}", _testScorer.Name,
			Math.Round(rankScore, 4));
	}

	public void Test(string testFile, string? performanceOutputFile)
	{
		var test = ReadInput(testFile);
		var rankScore = 0.0;
		var ids = new List<string>();
		var scores = new List<double>();

		foreach (var l in test)
		{
			var score = _testScorer.Score(l);
			ids.Add(l.Id);
			scores.Add(score);
			rankScore += score;
		}

		rankScore /= test.Count;
		ids.Add("all");
		scores.Add(rankScore);

		_logger.LogInformation("{TestScorerName} on test data: {RoundedRankScore}", _testScorer.Name,
			Math.Round(rankScore, 4));

		if (!string.IsNullOrEmpty(performanceOutputFile))
		{
			SavePerRankListPerformanceFile(ids, scores, performanceOutputFile);
			_logger.LogInformation($"Per-ranked list performance saved to: {performanceOutputFile}");
		}
	}

	public void Test(string modelFile, string testFile, string? performanceOutputFile)
	{
		var ranker = _rankerFactory.LoadRankerFromFile(modelFile);
		var features = ranker.Features;
		var test = ReadInput(testFile);

		if (_normalize)
			Normalize(test, features);

		var rankScore = 0.0;
		var ids = new List<string>();
		var scores = new List<double>();

		foreach (var aTest in test)
		{
			var rankedList = ranker.Rank(aTest);
			var score = _testScorer.Score(rankedList);
			ids.Add(rankedList.Id);
			scores.Add(score);
			rankScore += score;
		}

		rankScore /= test.Count;
		ids.Add("all");
		scores.Add(rankScore);

		_logger.LogInformation("{TestScorerName} on test data: {RoundedRankScore}", _testScorer.Name,
			Math.Round(rankScore, 4));

		if (!string.IsNullOrEmpty(performanceOutputFile))
		{
			SavePerRankListPerformanceFile(ids, scores, performanceOutputFile);
			_logger.LogInformation($"Per-ranked list performance saved to: {performanceOutputFile}");
		}
	}

	public void Test(List<string> modelFiles, string testFile, string performanceOutputFile)
	{
		var trainingData = new List<List<RankList>>();
		var testData = new List<List<RankList>>();

		var foldCount = modelFiles.Count;
		var samples = ReadInput(testFile);

		_logger.LogInformation($"Preparing {foldCount}-fold test data... ");
		_featureManager.PrepareCV(samples, foldCount, trainingData, testData);

		var rankScore = 0.0;
		var ids = new List<string>();
		var scores = new List<double>();

		for (var f = 0; f < foldCount; f++)
		{
			var test = testData[f];
			var ranker = _rankerFactory.LoadRankerFromFile(modelFiles[f]);
			var features = ranker.Features;

			if (_normalize)
				Normalize(test, features);

			foreach (var aTest in test)
			{
				var rankedList = ranker.Rank(aTest);
				var score = _testScorer.Score(rankedList);
				ids.Add(rankedList.Id);
				scores.Add(score);
				rankScore += score;
			}
		}

		rankScore /= ids.Count;
		ids.Add("all");
		scores.Add(rankScore);

		_logger.LogInformation("{TestScorerName} on test data: {RoundedRankScore}", _testScorer.Name,
			Math.Round(rankScore, 4));

		if (!string.IsNullOrEmpty(performanceOutputFile))
		{
			SavePerRankListPerformanceFile(ids, scores, performanceOutputFile);
			_logger.LogInformation($"Per-ranked list performance saved to: {performanceOutputFile}");
		}
	}

	public void Test(List<string> modelFiles, List<string> testFiles, string performanceOutputFile)
	{
		var foldCount = modelFiles.Count;
		var rankScore = 0.0;
		var ids = new List<string>();
		var scores = new List<double>();

		for (var f = 0; f < foldCount; f++)
		{
			var testRankLists = ReadInput(testFiles[f]);
			var ranker = _rankerFactory.LoadRankerFromFile(modelFiles[f]);
			var features = ranker.Features;

			if (_normalize)
				Normalize(testRankLists, features);

			foreach (var rankList in testRankLists)
			{
				var rankedList = ranker.Rank(rankList);
				var score = _testScorer.Score(rankedList);
				ids.Add(rankedList.Id);
				scores.Add(score);
				rankScore += score;
			}
		}

		rankScore /= ids.Count;
		ids.Add("all");
		scores.Add(rankScore);

		_logger.LogInformation("{TestScorerName} on test data: {RoundedRankScore}", _testScorer.Name,
			Math.Round(rankScore, 4));

		if (!string.IsNullOrEmpty(performanceOutputFile))
		{
			SavePerRankListPerformanceFile(ids, scores, performanceOutputFile);
			_logger.LogInformation("Per-ranked list performance saved to: {PerformanceOutputFile}", performanceOutputFile);
		}
	}

	public void TestWithScoreFile(string testFile, string scoreFile)
	{
		try
		{
			var test = ReadInput(testFile);
			var scores = new List<double>();
			using (var reader = SmartReader.OpenText(scoreFile))
			{
				while (reader.ReadLine() is { } content)
				{
					content = content.Trim();
					if (!string.IsNullOrEmpty(content))
						scores.Add(double.Parse(content));
				}
			}

			var k = 0;
			for (var i = 0; i < test.Count; i++)
			{
				var rl = test[i];
				var scoreArray = new double[rl.Count];

				for (var j = 0; j < rl.Count; j++)
					scoreArray[j] = scores[k++];

				test[i] = new RankList(rl, MergeSorter.Sort(scoreArray, false));
			}

			var rankScore = Evaluate(null, test);
			_logger.LogInformation("{TestScorerName} on test data: {RoundedRankScore}", _testScorer.Name,
				Math.Round(rankScore, 4));

		}
		catch (IOException e)
		{
			throw RankLibException.Create(e);
		}
	}

	public void Score(string modelFile, string testFile, string outputFile)
	{
		var ranker = _rankerFactory.LoadRankerFromFile(modelFile);
		var features = ranker.Features;
		var testRankLists = ReadInput(testFile);

		if (_normalize)
			Normalize(testRankLists, features);

		try
		{
			using var outWriter = new StreamWriter(new FileStream(outputFile, FileMode.Create), Encoding.UTF8);
			foreach (var rankList in testRankLists)
			{
				var rankListId = rankList.Id;
				for (var j = 0; j < rankList.Count; j++)
					outWriter.WriteLine($"{rankListId}\t{j}\t{ranker.Eval(rankList[j])}");
			}
		}
		catch (IOException ex)
		{
			throw RankLibException.Create("Error scoring and writing output file", ex);
		}
	}

	public void Score(List<string> modelFiles, string testFile, string outputFile)
	{
		var foldCount = modelFiles.Count;
		var trainingData = new List<List<RankList>>();
		var testData = new List<List<RankList>>(foldCount);
		var samples = ReadInput(testFile);

		_logger.LogInformation("Preparing {FoldCount}-fold test data...", foldCount);
		_featureManager.PrepareCV(samples, foldCount, trainingData, testData);

		try
		{
			using var outWriter = new StreamWriter(new FileStream(outputFile, FileMode.Create), Encoding.UTF8);
			for (var f = 0; f < foldCount; f++)
			{
				var testRankLists = testData[f];
				var ranker = _rankerFactory.LoadRankerFromFile(modelFiles[f]);
				var features = ranker.Features;

				if (_normalize)
					Normalize(testRankLists, features);

				foreach (var rankList in testRankLists)
				{
					var rankListId = rankList.Id;
					for (var j = 0; j < rankList.Count; j++)
						outWriter.WriteLine($"{rankListId}\t{j}\t{ranker.Eval(rankList[j])}");
				}
			}
		}
		catch (IOException ex)
		{
			throw RankLibException.Create("Error scoring and writing output file", ex);
		}
	}

	public void Score(List<string> modelFiles, List<string> testFiles, string outputFile)
	{
		try
		{
			using var outWriter = new StreamWriter(new FileStream(outputFile, FileMode.Create), Encoding.UTF8);
			for (var f = 0; f < modelFiles.Count; f++)
			{
				var testRankLists = ReadInput(testFiles[f]);
				var ranker = _rankerFactory.LoadRankerFromFile(modelFiles[f]);
				var features = ranker.Features;

				if (_normalize)
					Normalize(testRankLists, features);

				foreach (var rankList in testRankLists)
				{
					var rankListId = rankList.Id;
					for (var j = 0; j < rankList.Count; j++)
						outWriter.WriteLine($"{rankListId}\t{j}\t{ranker.Eval(rankList[j])}");
				}
			}
		}
		catch (IOException ex)
		{
			throw RankLibException.Create("Error scoring and writing output file", ex);
		}
	}

	public void Rank(string modelFile, string testFile, string indriRankingFile)
	{
		var ranker = _rankerFactory.LoadRankerFromFile(modelFile);
		var features = ranker.Features;
		var test = ReadInput(testFile);

		if (_normalize)
			Normalize(test, features);

		try
		{
			using var outWriter = new StreamWriter(new FileStream(indriRankingFile, FileMode.Create), Encoding.UTF8);
			foreach (var l in test)
			{
				var scores = new double[l.Count];
				for (var j = 0; j < l.Count; j++)
					scores[j] = ranker.Eval(l[j]);

				var idx = MergeSorter.Sort(scores, false);
				for (var j = 0; j < idx.Length; j++)
				{
					var k = idx[j];
					var str = $"{l.Id} Q0 {l[k].Description.Replace("#", "").Trim()} {j + 1} {SimpleMath.Round(scores[k], 5)} indri";
					outWriter.WriteLine(str);
				}
			}
		}
		catch (IOException ex)
		{
			throw RankLibException.Create("Error ranking and writing indri ranking file", ex);
		}
	}

	public void Rank(string testFile, string indriRankingFile)
	{
		var test = ReadInput(testFile);

		try
		{
			using var outWriter = new StreamWriter(new FileStream(indriRankingFile, FileMode.Create), Encoding.UTF8);
			foreach (var l in test)
			{
				for (var j = 0; j < l.Count; j++)
				{
					var str = $"{l.Id} Q0 {l[j].Description.Replace("#", "").Trim()} {j + 1} {SimpleMath.Round(1.0 - 0.0001 * j, 5)} indri";
					outWriter.WriteLine(str);
				}
			}
		}
		catch (IOException ex)
		{
			throw RankLibException.Create("Error ranking and writing indri ranking file", ex);
		}
	}

	public void Rank(List<string> modelFiles, string testFile, string indriRankingFile)
	{
		var trainingData = new List<List<RankList>>();
		var testData = new List<List<RankList>>();
		var foldCount = modelFiles.Count;
		var samples = ReadInput(testFile);

		_logger.LogInformation($"Preparing {foldCount}-fold test data...");
		_featureManager.PrepareCV(samples, foldCount, trainingData, testData);

		try
		{
			using var outWriter = new StreamWriter(new FileStream(indriRankingFile, FileMode.Create), Encoding.UTF8);
			for (var f = 0; f < foldCount; f++)
			{
				var test = testData[f];
				var ranker = _rankerFactory.LoadRankerFromFile(modelFiles[f]);
				var features = ranker.Features;

				if (_normalize)
					Normalize(test, features);

				foreach (var rankList in test)
				{
					var scores = new double[rankList.Count];
					for (var j = 0; j < rankList.Count; j++)
						scores[j] = ranker.Eval(rankList[j]);

					var idx = MergeSorter.Sort(scores, false);
					for (var j = 0; j < idx.Length; j++)
					{
						var k = idx[j];
						outWriter.WriteLine($"{rankList.Id} Q0 {rankList[k].Description.AsSpan().Trim("#").Trim().ToString()} {j + 1} {SimpleMath.Round(scores[k], 5)} indri");
					}
				}
			}
		}
		catch (IOException ex)
		{
			throw RankLibException.Create("Error in Evaluator::Rank(): ", ex);
		}
	}

	public void Rank(List<string> modelFiles, List<string> testFiles, string indriRankingFile)
	{
		var foldCount = modelFiles.Count;

		try
		{
			using var outWriter = new StreamWriter(new FileStream(indriRankingFile, FileMode.Create), Encoding.UTF8);
			for (var f = 0; f < foldCount; f++)
			{
				var test = ReadInput(testFiles[f]);
				var ranker = _rankerFactory.LoadRankerFromFile(modelFiles[f]);
				var features = ranker.Features;

				if (_normalize)
					Normalize(test, features);

				foreach (var l in test)
				{
					var scores = new double[l.Count];
					for (var j = 0; j < l.Count; j++)
						scores[j] = ranker.Eval(l[j]);

					var idx = MergeSorter.Sort(scores, false);
					for (var j = 0; j < idx.Length; j++)
					{
						var k = idx[j];
						var str = $"{l.Id} Q0 {l[k].Description.Replace("#", "").Trim()} {j + 1} {SimpleMath.Round(scores[k], 5)} indri";
						outWriter.WriteLine(str);
					}
				}
			}
		}
		catch (IOException ex)
		{
			throw RankLibException.Create("Error in Evaluator::Rank(): ", ex);
		}
	}

	private int[] PrepareSplit(string sampleFile, string? featureDefFile, double percentTrain, bool normalize, List<RankList> trainingData, List<RankList> testData)
	{
		var data = ReadInput(sampleFile);
		var features = ReadFeature(featureDefFile) ?? _featureManager.GetFeatureFromSampleVector(data);

		if (normalize)
			Normalize(data, features);

		_featureManager.PrepareSplit(data, percentTrain, trainingData, testData);
		return features;
	}

	private void SavePerRankListPerformanceFile(List<string> ids, List<double> scores, string performanceOutputFile)
	{
		using var writer = new StreamWriter(performanceOutputFile);
		for (var i = 0; i < ids.Count; i++)
			writer.WriteLine($"{_testScorer.Name}   {ids[i]}   {scores[i]}");
	}
}
