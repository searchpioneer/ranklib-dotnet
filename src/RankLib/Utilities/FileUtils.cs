using System.IO.Compression;
using System.Text;

namespace RankLib.Utilities;

public static class FileUtils
{
	public static StreamReader SmartReader(string inputFile) => SmartReader(inputFile, Encoding.UTF8);

	public static StreamReader SmartReader(string inputFile, Encoding encoding)
	{
		Stream input = new FileStream(inputFile, FileMode.Open, FileAccess.Read);

		if (inputFile.EndsWith(".gz"))
			input = new GZipStream(input, CompressionMode.Decompress);

		return new StreamReader(input, encoding);
	}

	public static string Read(string filename, Encoding encoding)
	{
		try
		{
			using var reader = SmartReader(filename, encoding);
			return reader.ReadToEnd();
		}
		catch (Exception e)
		{
			throw new InvalidOperationException("Error reading file", e);
		}
	}

	public static void Write(string filename, Encoding encoding, string content)
	{
		try
		{
			File.WriteAllText(filename, content, encoding);
		}
		catch (Exception e)
		{
			throw RankLibException.Create("Error writing to file", e);
		}
	}
}
