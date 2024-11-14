using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Learning.Tree;
using RankLib.Utilities;

namespace RankLib.Learning;

/// <summary>
/// Combines <see cref="RandomForests"/> ensembles into a new model.
/// </summary>
public class Combiner
{
	private readonly RankerFactory _rankerFactory;
	private readonly ILogger<Combiner> _logger;

	/// <summary>
	/// Instantiates a new instance of <see cref="Combiner"/>
	/// </summary>
	/// <param name="rankerFactory">The ranker factory to use to read ranker models</param>
	/// <param name="logger">Logger to log messages</param>
	public Combiner(RankerFactory rankerFactory, ILogger<Combiner>? logger = null)
	{
		_rankerFactory = rankerFactory;
		_logger = logger ?? NullLogger<Combiner>.Instance;
	}

	/// <summary>
	/// Combines the first <see cref="Ensemble"/> from each <see cref="RandomForests"/>
	/// ranker model in the given directory.
	/// </summary>
	/// <param name="directory">The directory containing the ranker models</param>
	/// <param name="outputFile">The file to which to write the output model.</param>
	/// <exception cref="RankLibException"></exception>
	public void Combine(string directory, string outputFile)
	{
		try
		{
			var files = Directory.GetFiles(directory);
			using var writer = new StreamWriter(outputFile, false, Encoding.ASCII);
			writer.WriteLine("## " + RandomForests.RankerName);

			foreach (var file in files)
			{
				if (file.Contains(".progress"))
					continue;

				var ranker = _rankerFactory.LoadRankerFromFile(file);
				if (ranker is RandomForests randomForests)
				{
					var ensemble = randomForests.Ensembles[0];
					writer.Write(ensemble.ToString());
				}
				else
					_logger.LogError("{File} is not a a RandomForests ranker. skipping", file);
			}
		}
		catch (Exception ex)
		{
			throw RankLibException.Create("Error combining files", ex);
		}
	}
}
