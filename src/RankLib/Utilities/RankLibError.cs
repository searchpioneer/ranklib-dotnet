namespace RankLib.Utilities;

public class RankLibError : Exception
{
	private RankLibError(Exception e)
		: base(e.Message, e)
	{
	}

	private RankLibError(string message)
		: base(message)
	{
	}

	private RankLibError(string message, Exception cause)
		: base(message, cause)
	{
	}

	public static RankLibError Create(Exception e)
	{
		if (e is RankLibError error)
		{
			return error;
		}
		return new RankLibError(e);
	}

	public static RankLibError Create(string message) => new RankLibError(message);

	public static RankLibError Create(string message, Exception cause)
	{
		if (cause is RankLibError error)
		{
			return error;
		}
		return new RankLibError(message, cause);
	}
}
