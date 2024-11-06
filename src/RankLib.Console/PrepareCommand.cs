using System.CommandLine;
using Microsoft.Extensions.Logging;
using RankLib.Features;
using RankLib.Learning;
using RankLib.Utilities;

namespace RankLib.Console;

public class PrepareCommandOptions : ICommandOptions
{
	public IEnumerable<FileInfo> Input { get; set; } = default!;

	public DirectoryInfo Output { get; set; } = default!;

	public bool Shuffle { get; set; }

	public float? Tvs { get; set; }

	public float? Tts { get; set; }

	public int? K { get; set; }
}

public class PrepareCommand : Command<PrepareCommandOptions, PrepareCommandOptionsHandler>
{
	public PrepareCommand()
	: base("prepare", "Split the input sample set into k chunks (folds) of roughly equal " +
					  "size and create train/ test data for each fold")
	{
		AddOption(new Option<IEnumerable<FileInfo>>("--input", "Source data (ranked lists)") { IsRequired = true }.ExistingOnly());
		AddOption(new Option<DirectoryInfo>("--output", "The output directory") { IsRequired = true });
		AddOption(new Option<bool>("--shuffle", "Create a copy of the input file in which the ordering of all ranked lists (e.g. queries) is randomized."));
		AddOption(new Option<float?>("--tvs", "Train-validation split ratio (x)(1.0-x)"));
		AddOption(new Option<float?>("--tts", "Train-test split ratio (x)(1.0-x)"));
		AddOption(new Option<int?>("--k", "The number of folds"));
	}
}

public class PrepareCommandOptionsHandler : ICommandOptionsHandler<PrepareCommandOptions>
{
	private readonly ILoggerFactory _loggerFactory;
	private readonly FeatureManager _featureManager;

	public PrepareCommandOptionsHandler(ILoggerFactory loggerFactory, FeatureManager featureManager)
	{
		_loggerFactory = loggerFactory;
		_featureManager = featureManager;
	}

	public Task<int> HandleAsync(PrepareCommandOptions options, CancellationToken cancellationToken)
	{
		var logger = _loggerFactory.CreateLogger<PrepareCommand>();

		if (options.K is not null && options is { K: > 0, Tts: not null })
		{
			logger.LogCritical("Error: Only one of k or tts should be specified.");
			return Task.FromResult(1);
		}

		if (options.Shuffle || options.K > 0 || options.Tts is not null)
		{
			var nFold = options.K ?? 0;
			var shuffle = options.Shuffle;
			var outputDir = options.Output.FullName;
			var rankingFiles = options.Input.Select(f => f.FullName).ToList();
			var tvs = options.Tvs ?? -1;

			var samples = _featureManager.ReadInput(rankingFiles);

			if (samples.Count == 0)
			{
				logger.LogInformation("Error: The input file is empty.");
				return Task.FromResult(1);
			}

			var fn = Path.GetFileName(rankingFiles[0]);

			Directory.CreateDirectory(outputDir);

			if (shuffle)
			{
				fn += ".shuffled";
				logger.LogInformation("Shuffling... ");
				samples.Shuffle();
				logger.LogInformation("Saving... ");
				_featureManager.Save(samples, Path.Combine(outputDir, fn));
			}

			if (options.Tts.HasValue)
			{
				var trains = new List<RankList>();
				var tests = new List<RankList>();

				logger.LogInformation("Splitting... ");
				_featureManager.PrepareSplit(samples, options.Tts.Value, trains, tests);

				try
				{
					logger.LogInformation("Saving splits...");
					_featureManager.Save(trains, Path.Combine(outputDir, $"train.{fn}"));
					_featureManager.Save(tests, Path.Combine(outputDir, $"test.{fn}"));
				}
				catch (Exception ex)
				{
					throw RankLibException.Create("Cannot save partition data.\nOccurred in FeatureManager::main(): ", ex);
				}
			}

			if (nFold > 0)
			{
				var trains = new List<List<RankList>>();
				var tests = new List<List<RankList>>();
				var valis = new List<List<RankList>>();
				logger.LogInformation("Partitioning... ");
				_featureManager.PrepareCV(samples, nFold, tvs, trains, valis, tests);

				try
				{
					for (var i = 0; i < trains.Count; i++)
					{
						logger.LogInformation($"Saving fold {i + 1}/{nFold}... ");
						_featureManager.Save(trains[i], Path.Combine(outputDir, $"f{i + 1}.train.{fn}"));
						_featureManager.Save(tests[i], Path.Combine(outputDir, $"f{i + 1}.test.{fn}"));
						if (tvs > 0)
						{
							_featureManager.Save(valis[i], Path.Combine(outputDir, $"f{i + 1}.validation.{fn}"));
						}
					}
				}
				catch (Exception ex)
				{
					throw RankLibException.Create("Cannot save partition data.\nOccurred in FeatureManager::main(): ", ex);
				}
			}
		}

		return Task.FromResult(0);
	}
}
