using System.IO.Compression;
using System.Text;

namespace RankLib.Utilities;

public static class FileUtils
{
	private const int BufSize = 51200;

	public static StreamReader SmartReader(string inputFile, string encoding = "UTF-8")
	{
		Stream input = new FileStream(inputFile, FileMode.Open, FileAccess.Read);

		if (inputFile.EndsWith(".gz"))
		{
			input = new GZipStream(input, CompressionMode.Decompress);
		}

		return new StreamReader(input, Encoding.GetEncoding(encoding));
	}

	public static string Read(string filename, string encoding)
	{
		var content = new StringBuilder(1000);

		try
		{
			using var reader = SmartReader(filename, encoding);
			var buffer = new char[BufSize];
			int numRead;

			while ((numRead = reader.Read(buffer, 0, buffer.Length)) != -1)
			{
				content.Append(buffer, 0, numRead);
			}
		}
		catch (Exception e)
		{
			throw new InvalidOperationException("Error reading file", e);
		}

		return content.ToString();
	}

	public static List<string> ReadLine(string filename, string encoding)
	{
		var lines = new List<string>();

		try
		{
			using var reader = SmartReader(filename, encoding);
			while (reader.ReadLine() is { } line)
			{
				line = line.Trim();

				if (line.Length == 0)
				{
					continue;
				}

				lines.Add(line);
			}
		}
		catch (Exception e)
		{
			throw new InvalidOperationException("Error reading lines from file", e);
		}

		return lines;
	}

	public static void Write(string filename, string encoding, string content)
	{
		try
		{
			using var writer = new StreamWriter(new FileStream(filename, FileMode.Create), Encoding.GetEncoding(encoding));
			writer.Write(content);
		}
		catch (Exception e)
		{
			throw new InvalidOperationException("Error writing to file", e);
		}
	}

	public static string[] GetAllFiles(string directory) => Directory.GetFiles(directory);

	public static bool Exists(string file) => File.Exists(file);

	public static void CopyFile(string srcFile, string dstFile)
	{
		try
		{
			File.Copy(srcFile, dstFile, overwrite: true);
		}
		catch (IOException e)
		{
			throw new InvalidOperationException("Error copying file", e);
		}
	}

	public static void CopyFiles(string srcDir, string dstDir, List<string> files)
	{
		foreach (var file in files)
		{
			CopyFile(Path.Combine(srcDir, file), Path.Combine(dstDir, file));
		}
	}

	public static int GunzipFile(FileInfo fileInput, DirectoryInfo dirOutput)
	{
		var fileOutputName = Path.GetFileNameWithoutExtension(fileInput.Name);
		var outputFile = new FileInfo(Path.Combine(dirOutput.FullName, fileOutputName));

		var buffer = new byte[BufSize];

		try
		{
			using (var gzipStream = new GZipStream(fileInput.OpenRead(), CompressionMode.Decompress))
			using (var destinationStream = outputFile.OpenWrite())
			{
				int len;

				while ((len = gzipStream.Read(buffer, 0, buffer.Length)) > 0)
				{
					destinationStream.Write(buffer, 0, len);
				}

				destinationStream.Flush();
			}
		}
		catch (IOException e)
		{
			throw new InvalidOperationException("Error decompressing file", e);
		}

		return 1;
	}

	public static int GzipFile(string inputFile, string gzipFilename)
	{
		try
		{
			using (var inStream = new FileStream(inputFile, FileMode.Open, FileAccess.Read))
			using (var outStream = new GZipStream(new FileStream(gzipFilename, FileMode.Create), CompressionMode.Compress))
			{
				inStream.CopyTo(outStream);
			}
		}
		catch (Exception e)
		{
			throw new InvalidOperationException("Error compressing file", e);
		}

		return 1;
	}

	public static string GetFileName(string pathName) => Path.GetFileName(pathName);

	public static string MakePathStandard(string directory)
	{
		if (!directory.EndsWith(Path.DirectorySeparatorChar.ToString()))
		{
			return directory + Path.DirectorySeparatorChar;
		}

		return directory;
	}
}
