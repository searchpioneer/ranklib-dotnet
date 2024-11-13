using RankLib.Learning;
using Xunit;
using Xunit.Abstractions;

namespace RankLib.Comparison.Tests;

public class ComparisonTests
{
	private readonly ITestOutputHelper _output;

	public ComparisonTests(ITestOutputHelper output) => _output = output;

	[Fact]
	public void CompareToRankyMcRankFace()
	{
		// Same (may have same double formatted differently in file)
		// - Coordinate Ascent
		// - Linear Regression
		// - MART

		// Differ by small amounts in some values (differences in floating point precision calculations?)
		// - RankBoost
		// - AdaRank

		// Different
		// - RankNet - (maybe use of random values?)
		// - ListNet - (maybe use of random values?)
		// - LambdaRank - (maybe use of random values?)
		// - LambdaMART - (Same with different double string representation up to tree 404. Compounding differences with floating point precision calculations?).
		// - Random Forests - (Same with different double string representation up to tree 404. Compounding differences with floating point precision calculations?).

		var javaExecutable = new JavaExecutable("RankyMcRankFace-0.2.0.jar");
		var dotnetExecutable = new DotnetExecutable("RankLib.Cli.dll");
		var trainFile = "sample_judgments_with_features.txt";
		var outputDir = Path.Combine(SolutionPaths.Root, "test_output");

		foreach (var rankerType in Enum.GetValues<RankerType>())
		{
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
		}

		// No assertions, check the outputs.
	}

	public static IEnumerable<object[]> RankerTypes =>
		Enum.GetValues<RankerType>().Select(rankerType => new object[] { rankerType });

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
