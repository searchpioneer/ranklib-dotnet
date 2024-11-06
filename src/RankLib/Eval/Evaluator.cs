using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Features;
using RankLib.Learning;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Eval;

public class Evaluator
{
	private readonly ILogger<Evaluator> _logger;
	private readonly RankerFactory _rankerFactory;
	private readonly MetricScorer _trainScorer;
	private readonly MetricScorer _testScorer;
	private readonly RankerTrainer _trainer;
	private readonly bool _mustHaveRelDoc;
	private readonly bool _useSparseRepresentation;
	private readonly FeatureManager _featureManager;
	private readonly bool _normalize;
	private readonly Normalizer _normalizer;

	public Evaluator(
		RankerFactory rankerFactory,
		FeatureManager featureManager,
		MetricScorer trainScorer,
		MetricScorer testScorer,
		RankerTrainer trainer,
		Normalizer? normalizer = null,
		bool mustHaveRelDoc = false,
		bool useSparseRepresentation = false,
		ILogger<Evaluator>? logger = null
	)
	{
		_rankerFactory = rankerFactory;
		_featureManager = featureManager;
		_trainScorer = trainScorer;
		_testScorer = testScorer;
		_trainer = trainer;
		_mustHaveRelDoc = mustHaveRelDoc;
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
		bool mustHaveRelDoc = false,
		bool useSparseRepresentation = false,
		ILogger<Evaluator>? logger = null
	)
	{
		_rankerFactory = rankerFactory;
		_featureManager = featureManager;
		_trainScorer = scorer;
		_testScorer = scorer;
		_trainer = trainer;
		_mustHaveRelDoc = mustHaveRelDoc;
		_useSparseRepresentation = useSparseRepresentation;
		_logger = logger ?? NullLogger<Evaluator>.Instance;
		_normalize = normalizer != null;
		_normalizer = normalizer ?? SumNormalizer.Instance;
	}

	private List<RankList> ReadInput(string inputFile) =>
		_featureManager.ReadInput(inputFile, _mustHaveRelDoc, _useSparseRepresentation);

	private void Normalize(List<RankList> samples, int[] fids)
	{
		foreach (var sample in samples)
			_normalizer.Normalize(sample, fids);
	}

	private void NormalizeAll(List<List<RankList>> samples, int[] fids)
	{
		foreach (var sample in samples)
			Normalize(sample, fids);
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
		string? featureDefFile,
		string? modelFile = null,
		IRankerParameters? parameters = default)
	{
		if (!typeof(IRanker).IsAssignableFrom(rankerType))
			throw new ArgumentException($"Ranker type {rankerType} is not a ranker");

		var train = ReadInput(trainFile);
		var validation = !string.IsNullOrEmpty(validationFile) ? ReadInput(validationFile) : null;
		var test = !string.IsNullOrEmpty(testFile) ? ReadInput(testFile) : null;
		var features = ReadFeature(featureDefFile) ?? _featureManager.GetFeatureFromSampleVector(train);

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
		string? featureDefFile,
		string? modelFile = null,
		TRankerParameters? parameters = default)
		where TRanker : IRanker<TRankerParameters>
		where TRankerParameters : IRankerParameters =>
		EvaluateAsync(typeof(TRanker), trainFile, validationFile, testFile, featureDefFile, modelFile, parameters);

	public async Task EvaluateAsync(
		Type rankerType,
		string sampleFile,
		string? validationFile,
		string? featureDefFile,
		double percentTrain,
		string? modelFile = null,
		IRankerParameters? parameters = default)
	{
		var train = new List<RankList>();
		var test = new List<RankList>();
		var features = PrepareSplit(sampleFile, featureDefFile, percentTrain, _normalize, train, test);
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
		string featureDefFile,
		double percentTrain,
		string? modelFile = null,
		TRankerParameters? parameters = default)
		where TRanker : IRanker<TRankerParameters>
		where TRankerParameters : IRankerParameters =>
		EvaluateAsync(typeof(TRanker), sampleFile, validationFile, featureDefFile, percentTrain, modelFile, parameters);

	public async Task EvaluateAsync(
		Type rankerType,
		string trainFile,
		double percentTrain,
		string? testFile,
		string? featureDefFile,
		string? modelFile = null,
		IRankerParameters? parameters = default)
	{
		var train = new List<RankList>();
		var validation = new List<RankList>();
		var features = PrepareSplit(trainFile, featureDefFile, percentTrain, _normalize, train, validation);
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
		string featureDefFile,
		string? modelFile = null,
		TRankerParameters? parameters = default)
		where TRanker : IRanker<TRankerParameters>
		where TRankerParameters : IRankerParameters =>
		EvaluateAsync(typeof(TRanker), trainFile, percentTrain, testFile, featureDefFile, modelFile, parameters);

	public Task EvaluateAsync<TRanker, TRankerParameters>(
		string sampleFile,
		string featureDefFile,
		int nFold,
		string modelDir,
		string modelFile,
		TRankerParameters? parameters = default)
		where TRanker : IRanker<TRankerParameters>
		where TRankerParameters : IRankerParameters =>
		EvaluateAsync(typeof(TRanker), sampleFile, featureDefFile, nFold, -1, modelDir, modelFile, parameters);

	public async Task EvaluateAsync(
		Type rankerType,
		string sampleFile,
		string? featureDefFile,
		int nFold,
		float tvs,
		string modelDir,
		string modelFile,
		IRankerParameters? parameters = default)
	{
		var trainingData = new List<List<RankList>>();
		var validationData = new List<List<RankList>>();
		var testData = new List<List<RankList>>();
		var samples = ReadInput(sampleFile);
		var features = ReadFeature(featureDefFile) ?? _featureManager.GetFeatureFromSampleVector(samples);

		_featureManager.PrepareCV(samples, nFold, tvs, trainingData, validationData, testData);

		if (_normalize)
		{
			for (var i = 0; i < nFold; i++)
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
		var scores = new double[nFold][];

		for (var i = 0; i < nFold; i++)
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

		for (var i = 0; i < nFold; i++)
			_logger.LogInformation($"Fold {i + 1}\t|   {Math.Round(scores[i][0], 4)}\t|  {Math.Round(scores[i][1], 4)}\t");

		_logger.LogInformation($"Avg.\t|   {Math.Round(scoreOnTrain / nFold, 4)}\t|  {Math.Round(scoreOnTest / nFold, 4)}\t");
		_logger.LogInformation($"Total\t|   \t\t|  {Math.Round(totalScoreOnTest / totalTestSampleSize, 4)}\t");
	}

	public Task EvaluateAsync<TRanker, TRankerParameters>(
		string sampleFile,
		string? featureDefFile,
		int nFold,
		float tvs,
		string modelDir,
		string modelFile,
		TRankerParameters? parameters = default)
		where TRanker : IRanker<TRankerParameters>
		where TRankerParameters : IRankerParameters =>
		EvaluateAsync(typeof(TRanker), sampleFile, featureDefFile, nFold, tvs, modelDir, modelFile, parameters);

	public void Test(string testFile)
	{
		var test = ReadInput(testFile);
		var rankScore = Evaluate(null, test);
		_logger.LogInformation("{TestScorerName} on test data: {RoundedRankScore}", _testScorer.Name,
			Math.Round(rankScore, 4));
	}

	public void Test(string testFile, string? prpFile)
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

		if (!string.IsNullOrEmpty(prpFile))
		{
			SavePerRankListPerformanceFile(ids, scores, prpFile);
			_logger.LogInformation($"Per-ranked list performance saved to: {prpFile}");
		}
	}

	public void Test(string modelFile, string testFile, string? prpFile)
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

		if (!string.IsNullOrEmpty(prpFile))
		{
			SavePerRankListPerformanceFile(ids, scores, prpFile);
			_logger.LogInformation($"Per-ranked list performance saved to: {prpFile}");
		}
	}

	public void Test(List<string> modelFiles, string testFile, string prpFile)
	{
		var trainingData = new List<List<RankList>>();
		var testData = new List<List<RankList>>();

		var nFold = modelFiles.Count;
		var samples = ReadInput(testFile);

		_logger.LogInformation($"Preparing {nFold}-fold test data... ");
		_featureManager.PrepareCV(samples, nFold, trainingData, testData);

		var rankScore = 0.0;
		var ids = new List<string>();
		var scores = new List<double>();

		for (var f = 0; f < nFold; f++)
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

		if (!string.IsNullOrEmpty(prpFile))
		{
			SavePerRankListPerformanceFile(ids, scores, prpFile);
			_logger.LogInformation($"Per-ranked list performance saved to: {prpFile}");
		}
	}

	public void Test(List<string> modelFiles, List<string> testFiles, string prpFile)
	{
		var nFold = modelFiles.Count;
		var rankScore = 0.0;
		var ids = new List<string>();
		var scores = new List<double>();

		for (var f = 0; f < nFold; f++)
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

		if (!string.IsNullOrEmpty(prpFile))
		{
			SavePerRankListPerformanceFile(ids, scores, prpFile);
			_logger.LogInformation("Per-ranked list performance saved to: {PrpFile}", prpFile);
		}
	}

	public void TestWithScoreFile(string testFile, string scoreFile)
	{
		try
		{
			var test = ReadInput(testFile);
			var scores = new List<double>();
			using (var reader = FileUtils.SmartReader(scoreFile))
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
			throw RankLibException.Create("Error in Evaluator::Score(): ", ex);
		}
	}

	public void Score(List<string> modelFiles, string testFile, string outputFile)
	{
		var nFold = modelFiles.Count;
		var trainingData = new List<List<RankList>>();
		var testData = new List<List<RankList>>(nFold);
		var samples = ReadInput(testFile);

		_logger.LogInformation("Preparing {NFold}-fold test data...", nFold);
		_featureManager.PrepareCV(samples, nFold, trainingData, testData);

		try
		{
			using var outWriter = new StreamWriter(new FileStream(outputFile, FileMode.Create), Encoding.UTF8);
			for (var f = 0; f < nFold; f++)
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
			throw RankLibException.Create("Error in Evaluator::Score(): ", ex);
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
			throw RankLibException.Create("Error in Evaluator::Score(): ", ex);
		}
	}

	public void Rank(string modelFile, string testFile, string indriRanking)
	{
		var ranker = _rankerFactory.LoadRankerFromFile(modelFile);
		var features = ranker.Features;
		var test = ReadInput(testFile);

		if (_normalize)
			Normalize(test, features);

		try
		{
			using var outWriter = new StreamWriter(new FileStream(indriRanking, FileMode.Create), Encoding.UTF8);
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
			throw RankLibException.Create("Error in Evaluator::Rank(): ", ex);
		}
	}

	public void Rank(string testFile, string indriRanking)
	{
		var test = ReadInput(testFile);

		try
		{
			using var outWriter = new StreamWriter(new FileStream(indriRanking, FileMode.Create), Encoding.UTF8);
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
			throw RankLibException.Create("Error in Evaluator::Rank(): ", ex);
		}
	}

	public void Rank(List<string> modelFiles, string testFile, string indriRanking)
	{
		var trainingData = new List<List<RankList>>();
		var testData = new List<List<RankList>>();
		var nFold = modelFiles.Count;
		var samples = ReadInput(testFile);

		_logger.LogInformation($"Preparing {nFold}-fold test data...");
		_featureManager.PrepareCV(samples, nFold, trainingData, testData);

		try
		{
			using var outWriter = new StreamWriter(new FileStream(indriRanking, FileMode.Create), Encoding.UTF8);
			for (var f = 0; f < nFold; f++)
			{
				var test = testData[f];
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

	public void Rank(List<string> modelFiles, List<string> testFiles, string indriRanking)
	{
		var nFold = modelFiles.Count;

		try
		{
			using var outWriter = new StreamWriter(new FileStream(indriRanking, FileMode.Create), Encoding.UTF8);
			for (var f = 0; f < nFold; f++)
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

	private void SavePerRankListPerformanceFile(List<string> ids, List<double> scores, string prpFile)
	{
		using var writer = new StreamWriter(prpFile);
		for (var i = 0; i < ids.Count; i++)
			writer.WriteLine($"{_testScorer.Name}   {ids[i]}   {scores[i]}");
	}
}
