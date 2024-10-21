using System.CommandLine;
using Microsoft.Extensions.Logging;
using RankLib.Eval;
using RankLib.Stats;

namespace RankLib.Console;

public class AnalyzeCommandOptions : ICommandOptions
{
	public DirectoryInfo All { get; set; } = default!;

	public FileInfo Base { get; set; } = default!;

	public int? Np { get; set; }
}

public class AnalyzeCommand : Command<AnalyzeCommandOptions, AnalyzeCommandOptionsHandler>
{
	public AnalyzeCommand()
	: base("analyze", "Analyze performance comparison of saved models against a baseline")
	{
		AddOption(new Option<DirectoryInfo>("--all", "Directory of performance files (one per system)").ExistingOnly());
		AddOption(new Option<FileInfo>("--base", "Performance file for the baseline (MUST be in the same directory)").ExistingOnly());
		AddOption(new Option<int?>("--np", () => RandomPermutationTest.DefaultPermutationCount, "Number of permutation (Fisher randomization test)"));
	}
}

public class AnalyzeCommandOptionsHandler : ICommandOptionsHandler<AnalyzeCommandOptions>
{
	private readonly ILoggerFactory _loggerFactory;

	public AnalyzeCommandOptionsHandler(ILoggerFactory loggerFactory) => _loggerFactory = loggerFactory;

	public Task<int> HandleAsync(AnalyzeCommandOptions options, CancellationToken cancellationToken)
	{
		var test = options.Np != null
			? new RandomPermutationTest(options.Np.Value)
			: new RandomPermutationTest();

		var analyzer = new Analyzer(test, _loggerFactory.CreateLogger<Analyzer>());
		analyzer.Compare(options.All.FullName, options.Base.FullName);
		return Task.FromResult(0);
	}
}
