using System.Text;
using RankLib.Learning.Tree;
using RankLib.Utilities;

namespace RankLib.Learning;

public class Combiner
{
	public static void Main(string[] args)
	{
		var combiner = new Combiner();
		combiner.Combine(args[0], args[1]);
	}

	public void Combine(string directory, string outputFile)
	{
		var rankerFactory = new RankerFactory();
		var files = Directory.GetFiles(directory);

		try
		{
			using var writer = new StreamWriter(outputFile, false, Encoding.ASCII);
			writer.WriteLine("## " + new RFRanker().Name());

			foreach (var file in files)
			{
				if (file.Contains(".progress"))
				{
					continue;
				}

				var ranker = (RFRanker)rankerFactory.LoadRankerFromFile(file);
				var ensemble = ranker.GetEnsembles()[0];
				writer.Write(ensemble.ToString());
			}
		}
		catch (Exception ex)
		{
			throw RankLibError.Create($"Error in Combiner::Combine(): {ex.Message}", ex);
		}
	}
}
