using System.IO.Compression;
using System.Text;

namespace RankLib.Utilities;

internal static class SmartReader
{
	public static StreamReader OpenText(string path) => OpenText(path, Encoding.UTF8);

	public static StreamReader OpenText(string path, Encoding encoding)
	{
		Stream input = File.OpenRead(path);

		if (path.EndsWith(".gz"))
			input = new GZipStream(input, CompressionMode.Decompress);

		return new StreamReader(input, encoding);
	}

	public static string ReadToEnd(string path, Encoding encoding)
	{
		using var reader = OpenText(path, encoding);
		return reader.ReadToEnd();
	}
}
