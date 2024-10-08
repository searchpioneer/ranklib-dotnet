namespace RankLib.Utilities;

public class RankLibException : Exception
{
	private RankLibException(Exception e)
		: base(e.Message, e)
	{
	}

	private RankLibException(string message)
		: base(message)
	{
	}

	private RankLibException(string message, Exception cause)
		: base(message, cause)
	{
	}

	public static RankLibException Create(Exception e)
	{
		if (e is RankLibException error)
		{
			return error;
		}
		return new RankLibException(e);
	}

	public static RankLibException Create(string message) => new(message);

	public static RankLibException Create(string message, Exception cause)
	{
		if (cause is RankLibException error)
		{
			return error;
		}
		return new RankLibException(message, cause);
	}
}
