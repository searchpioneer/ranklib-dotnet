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
		var javaExecutable = new JavaExecutable("RankyMcRankFace-0.2.0.jar");
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
				"-frate", "1.0",
				"-thread", "1");

			if (!string.IsNullOrEmpty(error))
			{
				_output.WriteLine(error);
				_output.WriteLine("");
				_output.WriteLine(output);
				Assert.Fail();
			}

			_output.WriteLine($"Run dotnet executable for {rankerType}");
			Console.Program.Main([
				"eval",
				"--ranker", ranker,
				"--train", trainFile,
				"--save", Path.Combine(outputDir, $"model_dotnet_{ranker}.txt"),
				"--frate", "1.0",
				"--thread", "1"]);
		}
	}

	[Fact]
	public void SingleThreadedAndMultiThreadedOutputAreTheSame()
	{
		var trainFile = "sample_judgments_with_features.txt";
		var outputNotEqual = new List<RankerType>();

		var dotnetExecutable = new DotnetExecutable("RankLib.Console.dll");

		foreach (var rankerType in Enum.GetValues<RankerType>())
		{
			var ranker = ((int)rankerType).ToString();
			var singleThreadedOutput = $"model_dotnet_{ranker}_single.txt";
			var (output, error) = dotnetExecutable.Execute(
				"eval",
				"--ranker", ranker,
				"--train", trainFile,
				"--save", singleThreadedOutput,
				"--frate", "1.0",
				"--thread", "1",
				"--randomSeed", "42");

			if (!string.IsNullOrEmpty(error))
			{
				_output.WriteLine(error);
				_output.WriteLine("");
				_output.WriteLine(output);
				Assert.Fail();
			}

			var multiThreadedOutput = $"model_dotnet_{ranker}_multi.txt";
			(output, error) = dotnetExecutable.Execute(
				"eval",
				"--ranker", ranker,
				"--train", trainFile,
				"--save", multiThreadedOutput,
				"--frate", "1.0",
				"--thread", "-1",
				"--randomSeed", "42");

			if (!string.IsNullOrEmpty(error))
			{
				_output.WriteLine(error);
				_output.WriteLine("");
				_output.WriteLine(output);
				Assert.Fail();
			}

			if (!FilesAreEqual(singleThreadedOutput, multiThreadedOutput))
				outputNotEqual.Add(rankerType);
		}

		Assert.Empty(outputNotEqual);
	}

	private static bool FilesAreEqual(string f1, string f2)
	{
		var length = new FileInfo(f1).Length;
		if (length != new FileInfo(f2).Length)
			return false;

		var buf1 = new byte[4096];
		var buf2 = new byte[4096];

		using var stream1 = File.OpenRead(f1);
		using var stream2 = File.OpenRead(f2);
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
