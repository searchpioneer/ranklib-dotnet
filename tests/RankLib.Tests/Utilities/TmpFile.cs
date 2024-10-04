using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace RankLib.Tests.Utilities;

public class TmpFile : IDisposable
{
	private static readonly ILogger<TmpFile> logger = NullLogger<TmpFile>.Instance;
	private readonly FileInfo _fileInfo;

	public TmpFile()
	{
		var tempFileName = System.IO.Path.GetTempFileName();
		_fileInfo = new FileInfo(tempFileName);
	}

	public FileInfo GetFile() => _fileInfo;

	public string Path => _fileInfo.FullName;

	public StreamWriter GetWriter() => new StreamWriter(_fileInfo.FullName, false); // Opens the file in write mode

	public void Dispose()
	{
		try
		{
			_fileInfo.Refresh(); // Ensure latest state
			if (_fileInfo.Exists)
			{
				_fileInfo.Delete();
			}
		}
		catch (Exception ex)
		{
			logger.LogWarning("Couldn't delete temporary file: {FilePath}. Error: {Error}", _fileInfo.FullName, ex.Message);
		}
	}
}
