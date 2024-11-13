using System.CommandLine;
using Microsoft.Extensions.Logging;
using RankLib.Eval;
using RankLib.Features;
using RankLib.Learning;
using RankLib.Stats;

namespace RankLib.Cli;

public class StatsCommandOptions : ICommandOptions
{
	public FileInfo Model { get; set; } = default!;
}

public class StatsCommand : Command<StatsCommandOptions, StatsCommandOptionsHandler>
{
	public StatsCommand()
	: base("stats", "Feature statistics for the given model") =>
		AddArgument(new Argument<FileInfo>("model", "The path to the model file").ExistingOnly());
}

public class StatsCommandOptionsHandler : ICommandOptionsHandler<StatsCommandOptions>
{
	private readonly ILoggerFactory _loggerFactory;

	public StatsCommandOptionsHandler(ILoggerFactory loggerFactory) => _loggerFactory = loggerFactory;

	public Task<int> HandleAsync(StatsCommandOptions options, CancellationToken cancellationToken)
	{
		try
		{
			var featureStats = new FeatureStats(options.Model.FullName, _loggerFactory.CreateLogger<FeatureStats>());
			featureStats.WriteFeatureStats();
			return Task.FromResult(0);
		}
		catch (Exception e)
		{
			_loggerFactory.CreateLogger<StatsCommand>().LogCritical(e, "Failed processing saved {Model} model file. {Message}", options.Model, e.Message);
			return Task.FromResult(1);
		}
	}
}
