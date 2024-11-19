namespace RankLib.Utilities;

/// <summary>
/// An exception thrown by this library.
/// </summary>
public class RankLibException : Exception
{
	private RankLibException(Exception innerException)
		: base(innerException.Message, innerException)
	{
	}

	private RankLibException(string message)
		: base(message)
	{
	}

	private RankLibException(string message, Exception innerException)
		: base(message, innerException)
	{
	}

	/// <summary>
	/// Instantiates a new instance of <see cref="RankLibException"/>, or returns
	/// the inner exception if it is an instance of <see cref="RankLibException"/>.
	/// </summary>
	/// <param name="innerException">The inner exception</param>
	/// <returns>An instance of <see cref="RankLibException"/></returns>
	public static RankLibException Create(Exception innerException)
	{
		if (innerException is RankLibException rankLibException)
			return rankLibException;

		return new RankLibException(innerException);
	}

	/// <summary>
	/// Instantiates a new instance of <see cref="RankLibException"/> with the specified message.
	/// </summary>
	/// <param name="message">The exception message</param>
	/// <returns>A new instance of <see cref="RankLibException"/></returns>
	public static RankLibException Create(string message) => new(message);

	/// <summary>
	/// Instantiates a new instance of <see cref="RankLibException"/>, or returns
	/// the inner exception if it is an instance of <see cref="RankLibException"/>.
	/// </summary>
	/// <param name="message">The exception message</param>
	/// <param name="innerException">The inner exception</param>
	/// <returns>An instance of <see cref="RankLibException"/></returns>
	public static RankLibException Create(string message, Exception innerException)
	{
		if (innerException is RankLibException rankLibException)
			return rankLibException;

		return new RankLibException(message, innerException);
	}
}
