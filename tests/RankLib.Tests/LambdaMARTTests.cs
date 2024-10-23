using RankLib.Learning;
using Xunit;
using Xunit.Abstractions;

namespace RankLib.Tests;

public class LambdaMARTTests
{
	private static readonly int[] Features = [7, 2, 6, 8, 9, 5];

	private readonly ITestOutputHelper _testOutputHelper;

	public LambdaMARTTests(ITestOutputHelper testOutputHelper) => _testOutputHelper = testOutputHelper;

	[Fact]
	public void LoadFromFile()
	{
		var rankerFactory = new RankerFactory(new XUnitLoggerFactory(_testOutputHelper));
		var ranker = rankerFactory.LoadRankerFromFile("lambdamart.model");
		Assert.Equal("LambdaMART", ranker.Name);
		Assert.Equal(Features, ranker.Features);
	}
}
