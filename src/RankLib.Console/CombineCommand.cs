using System.CommandLine;
using Microsoft.Extensions.Logging;
using RankLib.Eval;
using RankLib.Learning;
using RankLib.Stats;

namespace RankLib.Console;

public class CombineCommandOptions : ICommandOptions
{
	public DirectoryInfo Directory { get; set; } = default!;

	public FileInfo Output { get; set; } = default!;
}

public class CombineCommand : Command<CombineCommandOptions, CombineCommandOptionsHandler>
{
	public CombineCommand()
	: base("combine", "Combines ensembles from files in a directory into one file")
	{
		AddArgument(new Argument<DirectoryInfo>("directory", "The directory of files to combine"));
		AddArgument(new Argument<FileInfo>("output", "The combined output file path"));
	}
}

public class CombineCommandOptionsHandler : ICommandOptionsHandler<CombineCommandOptions>
{
	private readonly ILoggerFactory _loggerFactory;

	public CombineCommandOptionsHandler(ILoggerFactory loggerFactory) => _loggerFactory = loggerFactory;

	public Task<int> HandleAsync(CombineCommandOptions options, CancellationToken cancellationToken)
	{
		try
		{
			var rankerFactory = new RankerFactory(_loggerFactory);
			var combiner = new Combiner(rankerFactory);
			combiner.Combine(options.Directory.FullName, options.Output.FullName);
			return Task.FromResult(0);
		}
		catch (Exception e)
		{
			_loggerFactory.CreateLogger<Combiner>().LogCritical(e, "Failed to combine files");
			return Task.FromResult(1);
		}
	}
}
