using Meziantou.Extensions.Logging.Xunit;
using Microsoft.Extensions.Logging;
using RankLib.Eval;
using RankLib.Learning;
using RankLib.Tests.Utilities;
using Xunit;
using Xunit.Abstractions;

namespace RankLib.Tests;

public class EvaluatorTest
{
	private readonly ITestOutputHelper _testOutputHelper;

	public EvaluatorTest(ITestOutputHelper testOutputHelper) => _testOutputHelper = testOutputHelper;

	private static readonly object DataPointLock = new();

	[Fact]
	public void TestCLINoArgs()
	{
		lock (DataPointLock)
		{
			Evaluator.Main([]);
		}
	}

	[Fact]
	public void TestCoorAscent()
	{
		using var dataFile = new TmpFile();
		using var modelFile = new TmpFile();
		WriteRandomData(dataFile);

		lock (DataPointLock)
		{
			Evaluator.Main([
				"-train",
				dataFile.Path,
				"-metric2t",
				"map",
				"-ranker",
				"4",
				"-save",
				modelFile.Path
			]);
		}

		var rf = new RankerFactory();
		var model = rf.LoadRankerFromFile(modelFile.Path);

		Assert.IsType<CoorAscent>(model);
		var cmodel = (CoorAscent)model;
		_testOutputHelper.WriteLine(string.Join(",", cmodel.weight));

		Assert.True(cmodel.weight[0] > cmodel.weight[1], $"Computed weight vector doesn't make sense with our fake data: {string.Join(",", cmodel.weight)}");
		Assert.True(cmodel.weight[0] > 0.9, $"Computed weight vector doesn't make sense with our fake data: {string.Join(",", cmodel.weight)}");
		Assert.True(cmodel.weight[1] < 0.1, $"Computed weight vector doesn't make sense with our fake data: {string.Join(",", cmodel.weight)}");
	}

	private void WriteRandomData(TmpFile dataFile)
	{
		using var outWriter = new StreamWriter(dataFile.Path);
		var rand = new Random();
		for (var i = 0; i < 100; i++)
		{
			var w1 = rand.Next(2) == 0 ? "-1.0" : "1.0";
			var w2 = rand.Next(2) == 0 ? "-1.0" : "1.0";
			outWriter.WriteLine($"1 qid:x 1:1.0 2:{w1} # P{i}");
			outWriter.WriteLine($"0 qid:x 1:0.9 2:{w2} # N{i}");
		}
	}

	private void WriteRandomDataCount(TmpFile dataFile, int numQ, int numD)
	{
		using var outWriter = new StreamWriter(dataFile.Path);
		var rand = new Random();
		for (var q = 0; q < numQ; q++)
		{
			var qid = q.ToString();
			for (var i = 0; i < numD; i++)
			{
				var w1 = rand.Next(2) == 0 ? "-1.0" : "1.0";
				var w2 = rand.Next(2) == 0 ? "-1.0" : "1.0";
				outWriter.WriteLine($"1 qid:{qid} 1:1.0 2:{w1} # P{i}");
				outWriter.WriteLine($"0 qid:{qid} 1:0.9 2:{w2} # N{i}");
			}
		}
	}

	[Fact]
	public void TestRF()
	{
		using var dataFile = new TmpFile();
		using var modelFile = new TmpFile();
		using var rankFile = new TmpFile();
		WriteRandomData(dataFile);
		TestRanker(dataFile, modelFile, rankFile, 8, "map");
	}

	[Fact]
	public void TestLinearReg()
	{
		using var dataFile = new TmpFile();
		using var modelFile = new TmpFile();
		using var rankFile = new TmpFile();
		WriteRandomData(dataFile);
		TestRanker(dataFile, modelFile, rankFile, 9, "map");
	}

	[Fact]
	public void TestCAscent()
	{
		using var dataFile = new TmpFile();
		using var modelFile = new TmpFile();
		using var rankFile = new TmpFile();
		WriteRandomData(dataFile);
		TestRanker(dataFile, modelFile, rankFile, 4, "map");
	}

	[Fact]
	public void TestMART()
	{
		using var dataFile = new TmpFile();
		using var modelFile = new TmpFile();
		using var rankFile = new TmpFile();
		WriteRandomData(dataFile);
		TestRanker(dataFile, modelFile, rankFile, 0, "map");
	}

	[Fact(Skip = "Fails with NaN")]
	public void TestRankBoost()
	{
		using var dataFile = new TmpFile();
		using var modelFile = new TmpFile();
		using var rankFile = new TmpFile();
		WriteRandomData(dataFile);
		TestRanker(dataFile, modelFile, rankFile, 1, "map");
	}

	[Fact(Skip = "Fails with NaN")]
	public void TestRankNet()
	{
		using var dataFile = new TmpFile();
		using var modelFile = new TmpFile();
		using var rankFile = new TmpFile();
		WriteRandomData(dataFile);
		TestRanker(dataFile, modelFile, rankFile, 2, "map");
	}

	[Fact(Skip = "Fails with Infinity or doesn't learn")]
	public void TestAdaRank()
	{
		using var dataFile = new TmpFile();
		using var modelFile = new TmpFile();
		using var rankFile = new TmpFile();
		WriteRandomDataCount(dataFile, 20, 20);
		TestRanker(dataFile, modelFile, rankFile, 3, "map");
	}

	[Fact(Skip = "Unstable based on initial conditions")]
	public void TestLambdaRank()
	{
		using var dataFile = new TmpFile();
		using var modelFile = new TmpFile();
		using var rankFile = new TmpFile();
		WriteRandomDataCount(dataFile, 10, 50);
		TestRanker(dataFile, modelFile, rankFile, 5, "map");
	}

	[Fact]
	public void TestLambdaMART()
	{
		using var dataFile = new TmpFile();
		using var modelFile = new TmpFile();
		using var rankFile = new TmpFile();
		WriteRandomData(dataFile);
		TestRanker(dataFile, modelFile, rankFile, 6, "map");
	}

	[Fact]
	public void TestListNet()
	{
		using var dataFile = new TmpFile();
		using var modelFile = new TmpFile();
		using var rankFile = new TmpFile();
		WriteRandomData(dataFile);
		TestRanker(dataFile, modelFile, rankFile, 6, "map");
	}

	private void TestRanker(TmpFile dataFile, TmpFile modelFile, TmpFile rankFile, int rnum, string measure)
	{
		_testOutputHelper.WriteLine($"Test Ranker: {rnum}");
		Evaluator.LoggerFactory = new LoggerFactory([new XUnitLoggerProvider(_testOutputHelper)]);

		lock (DataPointLock)
		{
			Evaluator.Main([
				"-train",
				dataFile.Path,
				"-metric2t",
				measure,
				"-ranker",
				rnum.ToString(),
				"-frate",
				"1.0",
				"-bag",
				"10",
				"-round",
				"10",
				"-epoch",
				"10",
				"-save",
				modelFile.Path
			]);
		}

		lock (DataPointLock)
		{
			Evaluator.Main([
				"-rank",
				dataFile.Path,
				"-load",
				modelFile.Path,
				"-indri",
				rankFile.Path
			]);
		}

		var pRank = int.MaxValue;
		var nRank = int.MaxValue;

		var trecrun = File.ReadAllLines(rankFile.Path);
		foreach (var line in trecrun)
		{
			var row = line.Split([' '], StringSplitOptions.RemoveEmptyEntries);
			Assert.Equal("Q0", row[1]); // unused
			var dname = row[2];
			var rank = int.Parse(row[3]);
			var score = double.Parse(row[4]);

			Assert.False(double.IsNaN(score));
			Assert.True(double.IsFinite(score));
			Assert.True(rank > 0);

			if (dname.StartsWith("P"))
			{
				pRank = Math.Min(rank, pRank);
			}
			else
			{
				nRank = Math.Min(rank, nRank);
			}

			Assert.True(pRank < nRank);
			Assert.Equal(1, pRank);
		}
	}
}
