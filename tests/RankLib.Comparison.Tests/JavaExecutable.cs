using System.Diagnostics;

namespace RankLib.Comparison.Tests;

public class JavaExecutable
{
	private readonly string _javaPath;
	private readonly string _jarPath;

	public JavaExecutable(string javaPath, string jarPath)
	{
		_javaPath = javaPath;
		_jarPath = jarPath;
	}

	public JavaExecutable(string jarPath) : this(FindJavaPath(), jarPath)
	{
	}

	private static string FindJavaPath()
	{
		var javaHome = Environment.GetEnvironmentVariable("JAVA_HOME");
		if (!string.IsNullOrEmpty(javaHome))
		{
			var javaExePath = Path.Combine(javaHome, "bin", "java");
			if (File.Exists(javaExePath) || File.Exists($"{javaExePath}.exe"))
				return javaExePath;
		}

		var processStartInfo = new ProcessStartInfo
		{
			FileName = Environment.OSVersion.Platform == PlatformID.Win32NT ? "where" : "which",
			Arguments = "java",
			RedirectStandardOutput = true,
			UseShellExecute = false,
			ErrorDialog = false,
			CreateNoWindow = true
		};

		using var process = new Process();
		process.StartInfo = processStartInfo;
		process.Start();
		var result = process.StandardOutput.ReadToEnd().Trim();
		process.WaitForExit();

		if (!string.IsNullOrEmpty(result))
			return result;

		throw new Exception("Could not find java executable.");
	}

	public (string output, string error) Execute(params string[] arguments)
	{
		var startInfo = new ProcessStartInfo
		{
			FileName = _javaPath,
			Arguments = $"-jar \"{_jarPath}\" {string.Join(" ", arguments)}",
			RedirectStandardOutput = true,
			RedirectStandardError = true,
			ErrorDialog = false,
			UseShellExecute = false,
			CreateNoWindow = true
		};

		using var process = new Process();
		process.StartInfo = startInfo;
		process.Start();
		var output = process.StandardOutput.ReadToEnd();
		var error = process.StandardError.ReadToEnd();
		process.WaitForExit();
		return (output, error);
	}
}
