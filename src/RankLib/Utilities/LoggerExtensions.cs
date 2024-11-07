using System.Text;
using Microsoft.Extensions.Logging;

namespace RankLib.Utilities;


internal interface IBufferedLogger : ILogger
{
	StringBuilder Buffer { get; }
}

internal class BufferedLogger : IBufferedLogger
{
	private readonly ILogger _logger;

	public BufferedLogger(ILogger logger, StringBuilder buffer)
	{
		_logger = logger;
		Buffer = buffer;
	}

	public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter) =>
		_logger.Log(logLevel, eventId, state, exception, formatter);

	public bool IsEnabled(LogLevel logLevel) => _logger.IsEnabled(logLevel);

	public IDisposable? BeginScope<TState>(TState state) where TState : notnull => _logger.BeginScope(state);

	public StringBuilder Buffer { get; }
}

internal static class LoggerExtensions
{
	public static void PrintLog(this ILogger logger, int[] len, string[] messages)
	{
		if (logger.IsEnabled(LogLevel.Information))
		{
			var builder = new StringBuilder();
			for (var i = 0; i < messages.Length; i++)
			{
				var msg = messages[i];
				if (msg.Length > len[i])
					builder.Append(msg.AsSpan(0, len[i]));
				else
					builder.Append(msg.PadRight(len[i], ' '));

				builder.Append(" | ");
			}

			logger.LogInformation("{Message}", builder.ToString());
		}
	}


	public static void PrintLog(this IBufferedLogger logger, int[] len, string[] messages)
	{
		if (logger.IsEnabled(LogLevel.Information))
		{
			for (var i = 0; i < messages.Length; i++)
			{
				var msg = messages[i];
				if (msg.Length > len[i])
					logger.Buffer.Append(msg.AsSpan(0, len[i]));
				else
					logger.Buffer.Append(msg.PadRight(len[i], ' '));

				logger.Buffer.Append(" | ");
			}
		}
	}

	public static void PrintLogLn(this IBufferedLogger logger, int[] len, string[] messages)
	{
		if (logger.IsEnabled(LogLevel.Information))
		{
			logger.PrintLog(len, messages);
			logger.FlushLog();
		}
	}

	public static void FlushLog(this IBufferedLogger logger)
	{
		if (logger.IsEnabled(LogLevel.Information))
		{
			if (logger.Buffer.Length > 0)
			{
				logger.LogInformation("{Message}", logger.Buffer.ToString());
				logger.Buffer.Clear();
			}
		}
	}
}
