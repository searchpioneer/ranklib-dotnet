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
	internal static ILoggerFactory LoggerFactory = new NullLoggerFactory();
	internal static ILogger<Evaluator> Logger = NullLogger<Evaluator>.Instance;

	// main settings
	public static bool MustHaveRelDoc = false;
	public static bool UseSparseRepresentation = false;
	public static bool normalize = false;
	public static Normalizer Normalizer = new SumNormalizer();
	public static string ModelFile = "";

	public static string QrelFile = ""; // measure such as NDCG and MAP requires "complete" judgment.

	// tmp settings, for personal use
	public static string NewFeatureFile = "";
	public static bool KeepOrigFeatures = false;
	public static int TopNew = 2000;

	protected RankerFactory RankerFactory = new();
	protected MetricScorerFactory MetricScorerFactory = new();

	protected MetricScorer? TrainScorer;
	protected MetricScorer? TestScorer;
	protected RankerType RankerType = RankerType.MART;

	public Evaluator(RankerType rType, Metric.Metric trainMetric, Metric.Metric testMetric, ILoggerFactory? loggerFactory = null)
	{
		loggerFactory ??= new NullLoggerFactory();

		RankerFactory = new(loggerFactory);
		MetricScorerFactory = new(loggerFactory);

		RankerType = rType;
		TrainScorer = MetricScorerFactory.CreateScorer(trainMetric);
		TestScorer = MetricScorerFactory.CreateScorer(testMetric);

		if (!string.IsNullOrEmpty(QrelFile))
		{
			TrainScorer.LoadExternalRelevanceJudgment(QrelFile);
			TestScorer.LoadExternalRelevanceJudgment(QrelFile);
		}
	}

	public Evaluator(RankerType rType, Metric.Metric trainMetric, int trainK, Metric.Metric testMetric, int testK)
	{
		RankerType = rType;
		TrainScorer = MetricScorerFactory.CreateScorer(trainMetric, trainK);
		TestScorer = MetricScorerFactory.CreateScorer(testMetric, testK);

		if (!string.IsNullOrEmpty(QrelFile))
		{
			TrainScorer.LoadExternalRelevanceJudgment(QrelFile);
			TestScorer.LoadExternalRelevanceJudgment(QrelFile);
		}
	}

	public Evaluator(RankerType rType, Metric.Metric trainMetric, Metric.Metric testMetric, int k)
	{
		RankerType = rType;
		TrainScorer = MetricScorerFactory.CreateScorer(trainMetric, k);
		TestScorer = MetricScorerFactory.CreateScorer(testMetric, k);

		if (!string.IsNullOrEmpty(QrelFile))
		{
			TrainScorer.LoadExternalRelevanceJudgment(QrelFile);
			TestScorer.LoadExternalRelevanceJudgment(QrelFile);
		}
	}

	public Evaluator(RankerType rType, Metric.Metric metric, int k)
	{
		RankerType = rType;
		TrainScorer = MetricScorerFactory.CreateScorer(metric, k);

		if (!string.IsNullOrEmpty(QrelFile))
		{
			TrainScorer.LoadExternalRelevanceJudgment(QrelFile);
		}
		TestScorer = TrainScorer;
	}

	public Evaluator(RankerType rType, string trainMetric, string testMetric)
	{
		RankerType = rType;
		TrainScorer = MetricScorerFactory.CreateScorer(trainMetric);
		TestScorer = MetricScorerFactory.CreateScorer(testMetric);

		if (!string.IsNullOrEmpty(QrelFile))
		{
			TrainScorer.LoadExternalRelevanceJudgment(QrelFile);
			TestScorer.LoadExternalRelevanceJudgment(QrelFile);
		}
	}

	public List<RankList> ReadInput(string inputFile) => FeatureManager.ReadInput(inputFile, MustHaveRelDoc, UseSparseRepresentation);

	public void Normalize(List<RankList> samples)
	{
		foreach (var sample in samples)
		{
			Normalizer.Normalize(sample);
		}
	}

	public void Normalize(List<RankList> samples, int[] fids)
	{
		foreach (var sample in samples)
		{
			Normalizer.Normalize(sample, fids);
		}
	}

	public void NormalizeAll(List<List<RankList>> samples, int[] fids)
	{
		foreach (var sample in samples)
		{
			Normalize(sample, fids);
		}
	}

	public int[]? ReadFeature(string featureDefFile)
	{
		if (string.IsNullOrEmpty(featureDefFile))
		{
			return null;
		}
		return FeatureManager.ReadFeature(featureDefFile);
	}

	public double Evaluate(Ranker? ranker, List<RankList> rl)
	{
		var rankedList = ranker != null ? ranker.Rank(rl) : rl;
		return TestScorer.Score(rankedList);
	}

	public void Evaluate(string trainFile, string? validationFile, string? testFile, string? featureDefFile)
	{
		var train = ReadInput(trainFile);
		var validation = !string.IsNullOrEmpty(validationFile) ? ReadInput(validationFile) : null;
		var test = !string.IsNullOrEmpty(testFile) ? ReadInput(testFile) : null;

		var features = ReadFeature(featureDefFile) ?? FeatureManager.GetFeatureFromSampleVector(train);

		if (normalize)
		{
			Normalize(train, features);
			if (validation != null)
				Normalize(validation, features);
			if (test != null)
				Normalize(test, features);
		}

		var trainer = new RankerTrainer(LoggerFactory);
		var ranker = trainer.Train(RankerType, train, validation, features, TestScorer);

		if (test != null)
		{
			var rankScore = Evaluate(ranker, test);
			Logger.LogInformation($"{TestScorer.Name} on test data: {Math.Round(rankScore, 4)}");
		}

		if (!string.IsNullOrEmpty(ModelFile))
		{
			ranker.Save(ModelFile);
			Logger.LogInformation($"Model saved to: {ModelFile}");
		}
	}

	public void Evaluate(string sampleFile, string validationFile, string featureDefFile, double percentTrain)
	{
		var trainingData = new List<RankList>();
		var testData = new List<RankList>();
		var features = PrepareSplit(sampleFile, featureDefFile, percentTrain, normalize, trainingData, testData);
		var validation = !string.IsNullOrEmpty(validationFile) ? ReadInput(validationFile) : null;

		if (normalize && validation != null)
		{
			Normalize(validation, features);
		}

		var trainer = new RankerTrainer(LoggerFactory);
		var ranker = trainer.Train(RankerType, trainingData, validation, features, TestScorer);

		var rankScore = Evaluate(ranker, testData);
		Logger.LogInformation($"{TestScorer.Name} on test data: {Math.Round(rankScore, 4)}");

		if (!string.IsNullOrEmpty(ModelFile))
		{
			ranker.Save(ModelFile);
			Logger.LogInformation($"Model saved to: {ModelFile}");
		}
	}

	public void Evaluate(string trainFile, double percentTrain, string testFile, string featureDefFile)
	{
		var train = new List<RankList>();
		var validation = new List<RankList>();
		var features = PrepareSplit(trainFile, featureDefFile, percentTrain, normalize, train, validation);
		var test = !string.IsNullOrEmpty(testFile) ? ReadInput(testFile) : null;

		if (normalize && test != null)
		{
			Normalize(test, features);
		}

		var trainer = new RankerTrainer(LoggerFactory);
		var ranker = trainer.Train(RankerType, train, validation, features, TestScorer);

		if (test != null)
		{
			var rankScore = Evaluate(ranker, test);
			Logger.LogInformation($"{TestScorer.Name} on test data: {Math.Round(rankScore, 4)}");
		}

		if (!string.IsNullOrEmpty(ModelFile))
		{
			ranker.Save(ModelFile);
			Logger.LogInformation($"Model saved to: {ModelFile}");
		}
	}

	public void Evaluate(string sampleFile, string featureDefFile, int nFold, string modelDir, string modelFile) => Evaluate(sampleFile, featureDefFile, nFold, -1, modelDir, modelFile);

	public void Evaluate(string sampleFile, string featureDefFile, int nFold, float tvs, string modelDir, string modelFile)
	{
		var trainingData = new List<List<RankList>>();
		var validationData = new List<List<RankList>>();
		var testData = new List<List<RankList>>();
		var samples = ReadInput(sampleFile);
		var features = ReadFeature(featureDefFile) ?? FeatureManager.GetFeatureFromSampleVector(samples);

		FeatureManager.PrepareCV(samples, nFold, tvs, trainingData, validationData, testData);

		if (normalize)
		{
			for (var i = 0; i < nFold; i++)
			{
				NormalizeAll(trainingData, features);
				NormalizeAll(validationData, features);
				NormalizeAll(testData, features);
			}
		}

		double scoreOnTrain = 0.0, scoreOnTest = 0.0, totalScoreOnTest = 0.0;
		var totalTestSampleSize = 0;

		var scores = new double[nFold][];

		for (var i = 0; i < nFold; i++)
		{
			scores[i] = new double[2];
			var train = trainingData[i];
			var validation = tvs > 0 ? validationData[i] : null;
			var test = testData[i];

			var trainer = new RankerTrainer(LoggerFactory);
			var ranker = trainer.Train(RankerType, train, validation, features, TestScorer);

			var testScore = Evaluate(ranker, test);
			scoreOnTrain += ranker.GetScoreOnTrainingData();
			scoreOnTest += testScore;
			totalScoreOnTest += testScore * test.Count;
			totalTestSampleSize += test.Count;

			scores[i][0] = ranker.GetScoreOnTrainingData();
			scores[i][1] = testScore;

			if (!string.IsNullOrEmpty(modelDir))
			{
				ranker.Save(Path.Combine(modelDir, $"f{i + 1}.{modelFile}"));
				Logger.LogInformation($"Fold-{i + 1} model saved to: {modelFile}");
			}
		}

		Logger.LogInformation("Summary:");
		Logger.LogInformation($"{TestScorer.Name}\t|   Train\t| Test");

		for (var i = 0; i < nFold; i++)
		{
			Logger.LogInformation($"Fold {i + 1}\t|   {Math.Round(scores[i][0], 4)}\t|  {Math.Round(scores[i][1], 4)}\t");
		}

		Logger.LogInformation($"Avg.\t|   {Math.Round(scoreOnTrain / nFold, 4)}\t|  {Math.Round(scoreOnTest / nFold, 4)}\t");
		Logger.LogInformation($"Total\t|   \t\t|  {Math.Round(totalScoreOnTest / totalTestSampleSize, 4)}\t");
	}

	public void Test(string testFile)
	{
		var test = ReadInput(testFile);
		var rankScore = Evaluate(null, test);
		Logger.LogInformation($"{TestScorer.Name} on test data: {Math.Round(rankScore, 4)}");
	}

	public void Test(string testFile, string prpFile)
	{
		var test = ReadInput(testFile);
		var rankScore = 0.0;
		var ids = new List<string>();
		var scores = new List<double>();

		foreach (var l in test)
		{
			var score = TestScorer.Score(l);
			ids.Add(l.Id);
			scores.Add(score);
			rankScore += score;
		}

		rankScore /= test.Count;
		ids.Add("all");
		scores.Add(rankScore);

		Logger.LogInformation($"{TestScorer.Name} on test data: {Math.Round(rankScore, 4)}");

		if (!string.IsNullOrEmpty(prpFile))
		{
			SavePerRankListPerformanceFile(ids, scores, prpFile);
			Logger.LogInformation($"Per-ranked list performance saved to: {prpFile}");
		}
	}

	public void Test(string modelFile, string testFile, string prpFile)
	{
		var ranker = RankerFactory.LoadRankerFromFile(modelFile);
		var features = ranker.Features;
		var test = ReadInput(testFile);

		if (normalize)
		{
			Normalize(test, features);
		}

		var rankScore = 0.0;
		var ids = new List<string>();
		var scores = new List<double>();

		foreach (var aTest in test)
		{
			var rankedList = ranker.Rank(aTest);
			var score = TestScorer.Score(rankedList);
			ids.Add(rankedList.Id);
			scores.Add(score);
			rankScore += score;
		}

		rankScore /= test.Count;
		ids.Add("all");
		scores.Add(rankScore);

		Logger.LogInformation($"{TestScorer.Name} on test data: {SimpleMath.Round(rankScore, 4)}");

		if (!string.IsNullOrEmpty(prpFile))
		{
			SavePerRankListPerformanceFile(ids, scores, prpFile);
			Logger.LogInformation($"Per-ranked list performance saved to: {prpFile}");
		}
	}

	public void Test(List<string> modelFiles, string testFile, string prpFile)
	{
		var trainingData = new List<List<RankList>>();
		var testData = new List<List<RankList>>();

		var nFold = modelFiles.Count;
		var samples = ReadInput(testFile);

		Logger.LogInformation($"Preparing {nFold}-fold test data... ");
		FeatureManager.PrepareCV(samples, nFold, trainingData, testData);

		var rankScore = 0.0;
		var ids = new List<string>();
		var scores = new List<double>();

		for (var f = 0; f < nFold; f++)
		{
			var test = testData[f];
			var ranker = RankerFactory.LoadRankerFromFile(modelFiles[f]);
			var features = ranker.Features;

			if (normalize)
			{
				Normalize(test, features);
			}

			foreach (var aTest in test)
			{
				var rankedList = ranker.Rank(aTest);
				var score = TestScorer.Score(rankedList);
				ids.Add(rankedList.Id);
				scores.Add(score);
				rankScore += score;
			}
		}

		rankScore /= ids.Count;
		ids.Add("all");
		scores.Add(rankScore);

		Logger.LogInformation($"{TestScorer.Name} on test data: {Math.Round(rankScore, 4)}");

		if (!string.IsNullOrEmpty(prpFile))
		{
			SavePerRankListPerformanceFile(ids, scores, prpFile);
			Logger.LogInformation($"Per-ranked list performance saved to: {prpFile}");
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
			var test = ReadInput(testFiles[f]);
			var ranker = RankerFactory.LoadRankerFromFile(modelFiles[f]);
			var features = ranker.Features;

			if (normalize)
			{
				Normalize(test, features);
			}

			foreach (var aTest in test)
			{
				var rankedList = ranker.Rank(aTest);
				var score = TestScorer.Score(rankedList);
				ids.Add(rankedList.Id);
				scores.Add(score);
				rankScore += score;
			}
		}

		rankScore /= ids.Count;
		ids.Add("all");
		scores.Add(rankScore);

		Logger.LogInformation($"{TestScorer.Name} on test data: {Math.Round(rankScore, 4)}");

		if (!string.IsNullOrEmpty(prpFile))
		{
			SavePerRankListPerformanceFile(ids, scores, prpFile);
			Logger.LogInformation($"Per-ranked list performance saved to: {prpFile}");
		}
	}

	public void TestWithScoreFile(string testFile, string scoreFile)
	{
		try
		{
			using (var inReader = FileUtils.SmartReader(scoreFile))
			{
				var test = ReadInput(testFile);
				var scores = new List<double>();

				while (inReader.ReadLine() is { } content)
				{
					content = content.Trim();
					if (!string.IsNullOrEmpty(content))
					{
						scores.Add(double.Parse(content));
					}
				}

				var k = 0;
				for (var i = 0; i < test.Count; i++)
				{
					var rl = test[i];
					var scoreArray = new double[rl.Count];

					for (var j = 0; j < rl.Count; j++)
					{
						scoreArray[j] = scores[k++];
					}

					test[i] = new RankList(rl, MergeSorter.Sort(scoreArray, false));
				}

				var rankScore = Evaluate(null, test);
				Logger.LogInformation($"{TestScorer.Name} on test data: {Math.Round(rankScore, 4)}");
			}
		}
		catch (IOException e)
		{
			throw RankLibError.Create(e);
		}
	}

	public void Score(string modelFile, string testFile, string outputFile)
	{
		var ranker = RankerFactory.LoadRankerFromFile(modelFile);
		var features = ranker.Features;
		var test = ReadInput(testFile);

		if (normalize)
		{
			Normalize(test, features);
		}

		try
		{
			using (var outWriter = new StreamWriter(new FileStream(outputFile, FileMode.Create), System.Text.Encoding.UTF8))
			{
				foreach (var l in test)
				{
					for (var j = 0; j < l.Count; j++)
					{
						outWriter.WriteLine($"{l.Id}\t{j}\t{ranker.Eval(l[j])}");
					}
				}
			}
		}
		catch (IOException ex)
		{
			throw RankLibError.Create("Error in Evaluator::Score(): ", ex);
		}
	}

	public void Score(List<string> modelFiles, string testFile, string outputFile)
	{
		var trainingData = new List<List<RankList>>();
		var testData = new List<List<RankList>>();
		var nFold = modelFiles.Count;
		var samples = ReadInput(testFile);

		Logger.LogInformation($"Preparing {nFold}-fold test data...");
		FeatureManager.PrepareCV(samples, nFold, trainingData, testData);

		try
		{
			using (var outWriter = new StreamWriter(new FileStream(outputFile, FileMode.Create), System.Text.Encoding.UTF8))
			{
				for (var f = 0; f < nFold; f++)
				{
					var test = testData[f];
					var ranker = RankerFactory.LoadRankerFromFile(modelFiles[f]);
					var features = ranker.Features;

					if (normalize)
					{
						Normalize(test, features);
					}

					foreach (var l in test)
					{
						for (var j = 0; j < l.Count; j++)
						{
							outWriter.WriteLine($"{l.Id}\t{j}\t{ranker.Eval(l[j])}");
						}
					}
				}
			}
		}
		catch (IOException ex)
		{
			throw RankLibError.Create("Error in Evaluator::Score(): ", ex);
		}
	}

	public void Score(List<string> modelFiles, List<string> testFiles, string outputFile)
	{
		var nFold = modelFiles.Count;

		try
		{
			using (var outWriter = new StreamWriter(new FileStream(outputFile, FileMode.Create), System.Text.Encoding.UTF8))
			{
				for (var f = 0; f < nFold; f++)
				{
					var test = ReadInput(testFiles[f]);
					var ranker = RankerFactory.LoadRankerFromFile(modelFiles[f]);
					var features = ranker.Features;

					if (normalize)
					{
						Normalize(test, features);
					}

					foreach (var l in test)
					{
						for (var j = 0; j < l.Count; j++)
						{
							outWriter.WriteLine($"{l.Id}\t{j}\t{ranker.Eval(l[j])}");
						}
					}
				}
			}
		}
		catch (IOException ex)
		{
			throw RankLibError.Create("Error in Evaluator::Score(): ", ex);
		}
	}

	public void Rank(string modelFile, string testFile, string indriRanking)
	{
		var ranker = RankerFactory.LoadRankerFromFile(modelFile);
		var features = ranker.Features;
		var test = ReadInput(testFile);

		if (normalize)
		{
			Normalize(test, features);
		}

		try
		{
			using var outWriter = new StreamWriter(new FileStream(indriRanking, FileMode.Create), Encoding.UTF8);
			foreach (var l in test)
			{
				var scores = new double[l.Count];
				for (var j = 0; j < l.Count; j++)
				{
					scores[j] = ranker.Eval(l[j]);
				}

				var idx = MergeSorter.Sort(scores, false);
				for (var j = 0; j < idx.Length; j++)
				{
					var k = idx[j];
					var str = $"{l.Id} Q0 {l[k].GetDescription().Replace("#", "").Trim()} {(j + 1)} {SimpleMath.Round(scores[k], 5)} indri";
					outWriter.WriteLine(str);
				}
			}
		}
		catch (IOException ex)
		{
			throw RankLibError.Create("Error in Evaluator::Rank(): ", ex);
		}
	}

	public void Rank(string testFile, string indriRanking)
	{
		var test = ReadInput(testFile);

		try
		{
			using (var outWriter = new StreamWriter(new FileStream(indriRanking, FileMode.Create), System.Text.Encoding.UTF8))
			{
				foreach (var l in test)
				{
					for (var j = 0; j < l.Count; j++)
					{
						var str = $"{l.Id} Q0 {l[j].GetDescription().Replace("#", "").Trim()} {(j + 1)} {SimpleMath.Round(1.0 - 0.0001 * j, 5)} indri";
						outWriter.WriteLine(str);
					}
				}
			}
		}
		catch (IOException ex)
		{
			throw RankLibError.Create("Error in Evaluator::Rank(): ", ex);
		}
	}

	public void Rank(List<string> modelFiles, string testFile, string indriRanking)
	{
		var trainingData = new List<List<RankList>>();
		var testData = new List<List<RankList>>();
		var nFold = modelFiles.Count;
		var samples = ReadInput(testFile);

		Logger.LogInformation($"Preparing {nFold}-fold test data...");
		FeatureManager.PrepareCV(samples, nFold, trainingData, testData);

		try
		{
			using (var outWriter = new StreamWriter(new FileStream(indriRanking, FileMode.Create), System.Text.Encoding.UTF8))
			{
				for (var f = 0; f < nFold; f++)
				{
					var test = testData[f];
					var ranker = RankerFactory.LoadRankerFromFile(modelFiles[f]);
					var features = ranker.Features;

					if (normalize)
					{
						Normalize(test, features);
					}

					foreach (var l in test)
					{
						var scores = new double[l.Count];
						for (var j = 0; j < l.Count; j++)
						{
							scores[j] = ranker.Eval(l[j]);
						}

						var idx = MergeSorter.Sort(scores, false);
						for (var j = 0; j < idx.Length; j++)
						{
							var k = idx[j];
							var str = $"{l.Id} Q0 {l[k].GetDescription().Replace("#", "").Trim()} {(j + 1)} {SimpleMath.Round(scores[k], 5)} indri";
							outWriter.WriteLine(str);
						}
					}
				}
			}
		}
		catch (IOException ex)
		{
			throw RankLibError.Create("Error in Evaluator::Rank(): ", ex);
		}
	}

	public void Rank(List<string> modelFiles, List<string> testFiles, string indriRanking)
	{
		var nFold = modelFiles.Count;

		try
		{
			using (var outWriter = new StreamWriter(new FileStream(indriRanking, FileMode.Create), System.Text.Encoding.UTF8))
			{
				for (var f = 0; f < nFold; f++)
				{
					var test = ReadInput(testFiles[f]);
					var ranker = RankerFactory.LoadRankerFromFile(modelFiles[f]);
					var features = ranker.Features;

					if (normalize)
					{
						Normalize(test, features);
					}

					foreach (var l in test)
					{
						var scores = new double[l.Count];
						for (var j = 0; j < l.Count; j++)
						{
							scores[j] = ranker.Eval(l[j]);
						}

						var idx = MergeSorter.Sort(scores, false);
						for (var j = 0; j < idx.Length; j++)
						{
							var k = idx[j];
							var str = $"{l.Id} Q0 {l[k].GetDescription().Replace("#", "").Trim()} {(j + 1)} {SimpleMath.Round(scores[k], 5)} indri";
							outWriter.WriteLine(str);
						}
					}
				}
			}
		}
		catch (IOException ex)
		{
			throw RankLibError.Create("Error in Evaluator::Rank(): ", ex);
		}
	}

	private int[] PrepareSplit(string sampleFile, string featureDefFile, double percentTrain, bool normalize, List<RankList> trainingData, List<RankList> testData)
	{
		var data = ReadInput(sampleFile);
		var features = ReadFeature(featureDefFile) ?? FeatureManager.GetFeatureFromSampleVector(data);

		if (normalize)
		{
			Normalize(data, features);
		}

		FeatureManager.PrepareSplit(data, percentTrain, trainingData, testData);
		return features;
	}

	public void SavePerRankListPerformanceFile(List<string> ids, List<double> scores, string prpFile)
	{
		using (var writer = new StreamWriter(prpFile))
		{
			for (var i = 0; i < ids.Count; i++)
			{
				writer.WriteLine($"{TestScorer.Name}   {ids[i]}   {scores[i]}");
			}
		}
	}
}
