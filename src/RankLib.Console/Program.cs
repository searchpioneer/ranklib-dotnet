using System.CommandLine;
using System.CommandLine.Builder;
using System.CommandLine.Parsing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using RankLib.Console;
using RankLib.Eval;
using RankLib.Features;
using RankLib.Learning;
using RankLib.Metric;

internal class Program
{
	internal static Action<ILoggingBuilder>? ConfigureLogging { get; set; }

	public static int Main(string[] args)
	{
		var rootCommand = new RootCommand
		{
			new EvaluateCommand(),
			new AnalyzeCommand(),
			new CombineCommand(),
			new PrepareCommand(),
			new StatsCommand()
		};

		var builder = new CommandLineBuilder(rootCommand)
			.UseDefaults()
			.UseDependencyInjection(services =>
			{
				services.AddLogging(builder =>
				{
					builder.AddConsole();
					ConfigureLogging?.Invoke(builder);
				});

				services.AddSingleton<EvaluatorFactory>();
				services.AddSingleton<RankerFactory>();
				services.AddSingleton<MetricScorerFactory>();
				services.AddSingleton<FeatureManager>();
			});

		return builder.Build().Invoke(args);
	}
}
