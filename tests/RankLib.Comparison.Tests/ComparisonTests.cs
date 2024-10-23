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
}
