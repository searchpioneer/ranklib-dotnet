using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Learning.Boosting;
using RankLib.Learning.NeuralNet;
using RankLib.Learning.Tree;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning;

public class RankerFactory
{
	// Factory for creating Ranker instances
	protected Ranker[] Rankers;

	// Map for custom registered rankers
	protected Dictionary<string, string> map = new();


	protected readonly ILoggerFactory LoggerFactory;
	private readonly ILogger<RankerFactory> _logger;

	public RankerFactory(ILoggerFactory? loggerFactory = null)
	{
		LoggerFactory = loggerFactory ?? NullLoggerFactory.Instance;
		_logger = LoggerFactory.CreateLogger<RankerFactory>();

		Rankers =
		[
			new MART(LoggerFactory.CreateLogger<MART>()),
			new RankBoost(LoggerFactory.CreateLogger<RankBoost>()),
			new RankNet(LoggerFactory.CreateLogger<RankNet>()),
			new AdaRank(LoggerFactory.CreateLogger<AdaRank>()),
			new CoorAscent(),
			new LambdaRank(),
			new LambdaMART(LoggerFactory.CreateLogger<LambdaMART>()),
			new ListNet(),
			new RFRanker(),
			new LinearRegRank()
		];

		// Register all predefined rankers
		map[CreateRanker(RankerType.MART).Name.ToUpper()] = RankerType.MART.ToString();
		map[CreateRanker(RankerType.RANKNET).Name.ToUpper()] = RankerType.RANKNET.ToString();
		map[CreateRanker(RankerType.RANKBOOST).Name.ToUpper()] = RankerType.RANKBOOST.ToString();
		map[CreateRanker(RankerType.ADARANK).Name.ToUpper()] = RankerType.ADARANK.ToString();
		map[CreateRanker(RankerType.COOR_ASCENT).Name.ToUpper()] = RankerType.COOR_ASCENT.ToString();
		map[CreateRanker(RankerType.LAMBDARANK).Name.ToUpper()] = RankerType.LAMBDARANK.ToString();
		map[CreateRanker(RankerType.LAMBDAMART).Name.ToUpper()] = RankerType.LAMBDAMART.ToString();
		map[CreateRanker(RankerType.LISTNET).Name.ToUpper()] = RankerType.LISTNET.ToString();
		map[CreateRanker(RankerType.RANDOM_FOREST).Name.ToUpper()] = RankerType.RANDOM_FOREST.ToString();
		map[CreateRanker(RankerType.LINEAR_REGRESSION).Name.ToUpper()] = RankerType.LINEAR_REGRESSION.ToString();
	}

	public void Register(string name, string className) => map[name] = className;

	public Ranker CreateRanker(RankerType type) => Rankers[(int)type - (int)RankerType.MART].CreateNew();

	public Ranker CreateRanker(RankerType type, List<RankList> samples, int[] features, MetricScorer scorer)
	{
		var r = CreateRanker(type);
		r.SetTrainingSet(samples);
		r.Features = features;
		r.Scorer = scorer;
		return r;
	}

	public Ranker CreateRanker(string typeName)
	{
		if (Enum.TryParse(typeName, out RankerType rankerType))
		{
			return CreateRanker(rankerType);
		}

		Ranker ranker;
		try
		{
			var type = Type.GetType(typeName);
			if (type == null)
			{
				throw RankLibError.Create($"Type '{typeName}' does not exist.");
			}

			ranker = (Ranker)Activator.CreateInstance(type, LoggerFactory.CreateLogger(type))!;
		}
		catch (TypeLoadException e)
		{
			throw RankLibError.Create($"Could not find the type \"{typeName}\" specified. Make sure the assembly is referenced.", e);
		}
		catch (MissingMethodException e)
		{
			throw RankLibError.Create($"Cannot create an instance from the type \"{typeName}\".", e);
		}
		catch (InvalidCastException e)
		{
			throw RankLibError.Create($"The class \"{typeName}\" does not derive from \"{typeof(Ranker).FullName}\".", e);
		}
		return ranker;
	}

	public Ranker CreateRanker(string typeName, List<RankList> samples, int[] features, MetricScorer scorer)
	{
		var r = CreateRanker(typeName);
		r.SetTrainingSet(samples);
		r.Features = features;
		r.Scorer = scorer;
		return r;
	}

	public Ranker LoadRankerFromFile(string modelFile)
	{
		var fullText = FileUtils.Read(modelFile, "ASCII");
		return LoadRankerFromString(fullText);
	}

	public Ranker LoadRankerFromString(string fullText)
	{
		try
		{
			using var reader = new StringReader(fullText);
			var content = reader.ReadLine().Replace("## ", "").Trim(); // read the first line to get the ranker name
			_logger.LogInformation($"Model: {content}");
			var r = CreateRanker(map[content.ToUpper()]);
			r.LoadFromString(fullText);
			return r;
		}
		catch (Exception ex)
		{
			throw RankLibError.Create(ex);
		}
	}
}
