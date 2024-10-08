using System.CommandLine;
using Microsoft.Extensions.Logging;
using RankLib.Eval;
using RankLib.Stats;

namespace RankLib.Console;

public class AnalyzeCommandOptions : ICommandOptions
{
	public DirectoryInfo All { get; set; }

	public FileInfo Base { get; set; }

	public int? Np { get; set; } = RandomPermutationTest.NPermutation;
}

public class AnalyzeCommand : Command<AnalyzeCommandOptions, AnalyzeCommandOptionsHandler>
{
	public AnalyzeCommand()
	: base("analyze", "Analyze performance comparison of saved models against a baseline")
	{
		AddOption(new Option<DirectoryInfo>("--all", "Directory of performance files (one per system)"));
		AddOption(new Option<FileInfo>("--base", "Performance file for the baseline (MUST be in the same directory)"));
		AddOption(new Option<int?>("--np", () => RandomPermutationTest.NPermutation, "Number of permutation (Fisher randomization test)"));
	}
}

public class AnalyzeCommandOptionsHandler : ICommandOptionsHandler<AnalyzeCommandOptions>
{
	private readonly ILoggerFactory _loggerFactory;

	public AnalyzeCommandOptionsHandler(ILoggerFactory loggerFactory) => _loggerFactory = loggerFactory;

	public Task<int> HandleAsync(AnalyzeCommandOptions options, CancellationToken cancellationToken)
	{
		if (options.Np != null)
		{
			RandomPermutationTest.NPermutation = options.Np.Value;
		}

		var analyzer = new Analyzer(_loggerFactory.CreateLogger<Analyzer>());
		analyzer.Compare(options.All.FullName, options.Base.FullName);
		return Task.FromResult(0);
	}
}
