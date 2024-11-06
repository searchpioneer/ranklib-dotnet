using System.Text;
using RankLib.Learning.Tree;
using RankLib.Utilities;

namespace RankLib.Learning;

public class Combiner
{
	private readonly RankerFactory _rankerFactory;

	public Combiner(RankerFactory rankerFactory) => _rankerFactory = rankerFactory;

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

				var ranker = (RandomForests)_rankerFactory.LoadRankerFromFile(file);
				var ensemble = ranker.Ensembles[0];
				writer.Write(ensemble.ToString());
			}
		}
		catch (Exception ex)
		{
			throw RankLibException.Create("Error combining files", ex);
		}
	}
}
