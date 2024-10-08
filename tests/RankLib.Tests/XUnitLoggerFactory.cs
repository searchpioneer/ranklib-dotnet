using Meziantou.Extensions.Logging.Xunit;
using Microsoft.Extensions.Logging;
using Xunit.Abstractions;

namespace RankLib.Tests;

public class XUnitLoggerFactory : ILoggerFactory
{
	private readonly ITestOutputHelper _testOutputHelper;
	private readonly LoggerExternalScopeProvider _scopeProvider = new();
	public XUnitLoggerFactory(ITestOutputHelper testOutputHelper) => _testOutputHelper = testOutputHelper;

	public ILogger CreateLogger(string categoryName) =>
		new XUnitLogger(_testOutputHelper, _scopeProvider, categoryName);

	public void AddProvider(ILoggerProvider provider)
	{
	}

	public void Dispose()
	{
	}
}
