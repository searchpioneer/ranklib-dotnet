using System.Diagnostics;

namespace RankLib.Comparison.Tests;

public class DotnetExecutable
{
	private readonly string _dllPath;

	public DotnetExecutable(string dllPath) => _dllPath = dllPath;

	public (string output, string error) Execute(params string[] arguments)
	{
		var startInfo = new ProcessStartInfo
		{
			FileName = "dotnet",
			Arguments = $"\"{_dllPath}\" {string.Join(" ", arguments)}",
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
