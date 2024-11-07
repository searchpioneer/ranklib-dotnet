using Meziantou.Extensions.Logging.Xunit;
using Microsoft.Extensions.Logging;
using RankLib.Cli.Tests.Utilities;
using RankLib.Learning;
using Xunit;
using Xunit.Abstractions;

namespace RankLib.Cli.Tests;

public class ProgramTests
{
	private readonly ITestOutputHelper _testOutputHelper;

	public ProgramTests(ITestOutputHelper testOutputHelper) => _testOutputHelper = testOutputHelper;

	private static readonly object DataPointLock = new();

	[Fact]
	public void TestCLINoArgs()
	{
		lock (DataPointLock)
		{
			Program.Main([]);
		}
	}

	[Fact]
	public void TestCoorAscent()
	{
		using var dataFile = new TempFile();
		using var modelFile = new TempFile();
		WriteRandomData(dataFile);

		Program.ConfigureLogging = logging =>
		{
			logging.ClearProviders();
			logging.AddProvider(new XUnitLoggerProvider(_testOutputHelper));
		};

		lock (DataPointLock)
		{
			Program.Main([
				"eval",
				"--train",
				dataFile.Path,
				"--metric2t",
				"map",
				"--ranker",
				"4",
				"--save",
				modelFile.Path
			]);
		}

		var rankerFactory = new RankerFactory();
		var model = rankerFactory.LoadRankerFromFile(modelFile.Path);

		Assert.IsType<CoordinateAscent>(model);
		var cmodel = (CoordinateAscent)model;
		_testOutputHelper.WriteLine(string.Join(",", cmodel.Weight));

		Assert.True(cmodel.Weight[0] > cmodel.Weight[1], $"Computed weight vector doesn't make sense with our fake data: {string.Join(",", cmodel.Weight)}");
		Assert.True(cmodel.Weight[0] > 0.9, $"Computed weight vector doesn't make sense with our fake data: {string.Join(",", cmodel.Weight)}");
		Assert.True(cmodel.Weight[1] < 0.1, $"Computed weight vector doesn't make sense with our fake data: {string.Join(",", cmodel.Weight)}");
	}

	private void WriteRandomData(TempFile dataFile)
	{
		using var outWriter = dataFile.GetWriter();
		var rand = Random.Shared;
		for (var i = 0; i < 100; i++)
		{
			var w1 = rand.Next(2) == 0 ? "-1.0" : "1.0";
			var w2 = rand.Next(2) == 0 ? "-1.0" : "1.0";
			outWriter.WriteLine($"1 qid:x 1:1.0 2:{w1} # P{i}");
			outWriter.WriteLine($"0 qid:x 1:0.9 2:{w2} # N{i}");
		}
	}

	private void WriteRandomDataCount(TempFile dataFile, int numQ, int numD)
	{
		using var outWriter = dataFile.GetWriter();
		var rand = Random.Shared;
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
		using var dataFile = new TempFile();
		using var modelFile = new TempFile();
		using var rankFile = new TempFile();
		WriteRandomData(dataFile);
		TestRanker(dataFile, modelFile, rankFile, 8, "map");
	}

	[Fact]
	public void TestLinearReg()
	{
		using var dataFile = new TempFile();
		using var modelFile = new TempFile();
		using var rankFile = new TempFile();
		WriteRandomData(dataFile);
		TestRanker(dataFile, modelFile, rankFile, 9, "map");
	}

	[Fact]
	public void TestCAscent()
	{
		using var dataFile = new TempFile();
		using var modelFile = new TempFile();
		using var rankFile = new TempFile();
		WriteRandomData(dataFile);
		TestRanker(dataFile, modelFile, rankFile, 4, "map");
	}

	[Fact]
	public void TestMART()
	{
		using var dataFile = new TempFile();
		using var modelFile = new TempFile();
		using var rankFile = new TempFile();
		WriteRandomData(dataFile);
		TestRanker(dataFile, modelFile, rankFile, 0, "map");
	}

	[Fact(Skip = "Fails with NaN")]
	public void TestRankBoost()
	{
		using var dataFile = new TempFile();
		using var modelFile = new TempFile();
		using var rankFile = new TempFile();
		WriteRandomData(dataFile);
		TestRanker(dataFile, modelFile, rankFile, 1, "map");
	}

	[Fact(Skip = "Fails with NaN")]
	public void TestRankNet()
	{
		using var dataFile = new TempFile();
		using var modelFile = new TempFile();
		using var rankFile = new TempFile();
		WriteRandomData(dataFile);
		TestRanker(dataFile, modelFile, rankFile, 2, "map");
	}

	[Fact(Skip = "Fails with Infinity or doesn't learn")]
	public void TestAdaRank()
	{
		using var dataFile = new TempFile();
		using var modelFile = new TempFile();
		using var rankFile = new TempFile();
		WriteRandomDataCount(dataFile, 20, 20);
		TestRanker(dataFile, modelFile, rankFile, 3, "map");
	}

	[Fact(Skip = "Unstable based on initial conditions")]
	public void TestLambdaRank()
	{
		using var dataFile = new TempFile();
		using var modelFile = new TempFile();
		using var rankFile = new TempFile();
		WriteRandomDataCount(dataFile, 10, 50);
		TestRanker(dataFile, modelFile, rankFile, 5, "map");
	}

	[Fact]
	public void TestLambdaMART()
	{
		using var dataFile = new TempFile();
		using var modelFile = new TempFile();
		using var rankFile = new TempFile();
		WriteRandomData(dataFile);
		TestRanker(dataFile, modelFile, rankFile, 6, "map");
	}

	[Fact(Skip = "Sometimes fails Assert.True(pRank < nRank)")]
	public void TestListNet()
	{
		using var dataFile = new TempFile();
		using var modelFile = new TempFile();
		using var rankFile = new TempFile();
		WriteRandomData(dataFile);
		TestRanker(dataFile, modelFile, rankFile, 7, "map");
	}

	private void TestRanker(TempFile dataFile, TempFile modelFile, TempFile rankFile, int rnum, string measure)
	{
		_testOutputHelper.WriteLine($"Test Ranker: {rnum}");
		Program.ConfigureLogging = logging =>
		{
			logging.ClearProviders();
			logging.AddProvider(new XUnitLoggerProvider(_testOutputHelper));
		};

		lock (DataPointLock)
		{
			Program.Main([
				"eval",
				"--train",
				dataFile.Path,
				"--metric2t",
				measure,
				"--ranker",
				rnum.ToString(),
				"--frate",
				"1.0",
				"--bag",
				"10",
				"--round",
				"10",
				"--epoch",
				"10",
				"--save",
				modelFile.Path
			]);
		}

		lock (DataPointLock)
		{
			Program.Main([
				"eval",
				"--rank",
				dataFile.Path,
				"--load",
				modelFile.Path,
				"--indri",
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
				pRank = Math.Min(rank, pRank);
			else
				nRank = Math.Min(rank, nRank);

			Assert.True(pRank < nRank);
			Assert.Equal(1, pRank);
		}
	}
}
