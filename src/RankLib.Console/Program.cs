using System.CommandLine;
using System.CommandLine.Builder;
using System.CommandLine.Parsing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using RankLib.Console;

public class Program
{
	internal static Action<ILoggingBuilder>? ConfigureLogging { get; set; }

	public static int Main(string[] args)
	{
		var rootCommand = new RootCommand
		{
			new EvaluateCommand()
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
			});

		return builder.Build().Invoke(args);
	}
}
