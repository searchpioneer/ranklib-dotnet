using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Logging.Console;
using Microsoft.Extensions.Options;

namespace RankLib.Cli;

internal sealed class CustomFormatter : ConsoleFormatter, IDisposable
{
	private readonly IDisposable? _optionsReloadToken;
	private ConsoleFormatterOptions _formatterOptions;

	public CustomFormatter(IOptionsMonitor<ConsoleFormatterOptions> options) : base("custom") =>
		(_optionsReloadToken, _formatterOptions) =
		(options.OnChange(ReloadLoggerOptions), options.CurrentValue);

	private void ReloadLoggerOptions(ConsoleFormatterOptions options) =>
		_formatterOptions = options;

	public override void Write<TState>(
		in LogEntry<TState> logEntry,
		IExternalScopeProvider? scopeProvider,
		TextWriter textWriter)
	{
		var message = logEntry.Formatter.Invoke(logEntry.State, logEntry.Exception);
		textWriter.WriteLine(logEntry.LogLevel >= LogLevel.Error ? $"\x1B[1m\x1B[31m{message}\x1B[0m" : message);
	}

	public void Dispose() => _optionsReloadToken?.Dispose();
}
