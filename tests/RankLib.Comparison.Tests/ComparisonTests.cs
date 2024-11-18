
using Microsoft.Extensions.Logging;
using RankLib.Cli;
using RankLib.Learning;
using Xunit;
using Xunit.Abstractions;

namespace RankLib.Comparison.Tests;

public class ComparisonTests
{
	private readonly ITestOutputHelper _output;

	public ComparisonTests(ITestOutputHelper output)
	{
		_output = output;

		// Force RankLib.Cli to be copied over.
		Program.ConfigureLogging = logging =>
		{
			logging.ClearProviders();
		};
	}

	[Theory]
	[MemberData(nameof(RankerTypes))]
	public void CompareToRankyMcRankFace(RankerType rankerType)
	{
		// Same (may have same double formatted differently in file)
		// - 0: MART
		// - 9: Linear Regression

		// Differ by small amounts in some values (differences in floating point precision calculations?)
		// - 2: RankBoost
		// - 3: AdaRank

		// Different
		// - 1: RankNet - (maybe use of random values?)
		// - 4: Coordinate Ascent (use of random values)
		// - 5: LambdaRank - (maybe use of random values?)
		// - 6: LambdaMART - (Same with different double string representation up to tree 404. Compounding differences with floating point precision calculations?).
		// - 7: ListNet - (maybe use of random values?)
		// - 8: Random Forests - (Different splits from random sampling).

		var javaExecutable = new JavaExecutable("RankyMcRankFace-0.2.0.jar");
		var dotnetExecutable = new DotnetExecutable("RankLib.Cli.dll");
		var trainFile = "sample_judgments_with_features.txt";
		var outputDir = Path.Combine(SolutionPaths.Root, "test_output");
		var ranker = ((int)rankerType).ToString();

		_output.WriteLine($"Run java executable for {rankerType}");
		var (output, error) = javaExecutable.Execute(
			"-ranker", ranker,
			"-train", trainFile,
			"-save", Path.Combine(outputDir, $"model_java_{ranker}.txt"),
			"-frate", "1.0");

		_output.WriteLine(output);

		if (!string.IsNullOrEmpty(error))
		{
			_output.WriteLine("");
			_output.WriteLine(error);
			Assert.Fail();
		}

		_output.WriteLine($"Run dotnet executable for {rankerType}");
		(output, error) = dotnetExecutable.Execute(
			"eval",
			"-ranker", ranker,
			"-train", trainFile,
			"-save", Path.Combine(outputDir, $"model_dotnet_{ranker}.txt"),
			"-frate", "1.0");

		_output.WriteLine(output);

		if (!string.IsNullOrEmpty(error))
		{
			_output.WriteLine("");
			_output.WriteLine(error);
			Assert.Fail();
		}

		// No assertions, check the outputs.
	}

	public static TheoryData<RankerType> RankerTypes => new(Enum.GetValues<RankerType>());

	[Theory]
	[MemberData(nameof(RankerTypes))]
	public void SingleThreadedAndMultiThreadedOutputAreTheSame(RankerType rankerType)
	{
		var trainFile = "sample_judgments_with_features.txt";
		var dotnetExecutable = new DotnetExecutable("RankLib.Cli.dll");
		var ranker = ((int)rankerType).ToString();
		var singleThreadedOutput = $"model_dotnet_{ranker}_single.txt";
		var (output, error) = dotnetExecutable.Execute(
			"eval",
			"--ranker", ranker,
			"--train-input-file", trainFile,
			"--model-output-file", singleThreadedOutput,
			"--feature-sampling-rate", "1.0",
			"--thread", "1",
			"--random-seed", "42");

		_output.WriteLine(output);

		if (!string.IsNullOrEmpty(error))
		{
			_output.WriteLine("");
			_output.WriteLine(error);
			Assert.Fail();
		}

		var multiThreadedOutput = $"model_dotnet_{ranker}_multi.txt";
		(output, error) = dotnetExecutable.Execute(
			"eval",
			"--ranker", ranker,
			"--train-input-file", trainFile,
			"--model-output-file", multiThreadedOutput,
			"--feature-sampling-rate", "1.0",
			"--random-seed", "42");

		_output.WriteLine(output);

		if (!string.IsNullOrEmpty(error))
		{
			_output.WriteLine("");
			_output.WriteLine(error);
			Assert.Fail();
		}

		Assert.True(FilesAreEqual(singleThreadedOutput, multiThreadedOutput));
	}

	private static bool FilesAreEqual(string file1, string file2)
	{
		var fileInfo1 = new FileInfo(file1);
		var fileInfo2 = new FileInfo(file2);
		var length = fileInfo1.Length;
		if (length != fileInfo2.Length)
			return false;

		var buf1 = new byte[4096];
		var buf2 = new byte[4096];

		using var stream1 = File.OpenRead(file1);
		using var stream2 = File.OpenRead(file2);
		while (length > 0)
		{
			var toRead = buf1.Length;
			if (toRead > length)
				toRead = (int)length;
			length -= toRead;

			var b1 = stream1.Read(buf1, 0, toRead);
			var b2 = stream2.Read(buf2, 0, toRead);
			for (var i = 0; i < toRead; ++i)
			{
				if (buf1[i] != buf2[i])
					return false;
			}
		}

		return true;
	}
}
