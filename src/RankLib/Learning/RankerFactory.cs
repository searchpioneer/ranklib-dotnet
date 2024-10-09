using System.Text;
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
	private readonly ILoggerFactory _loggerFactory;
	private readonly ILogger<RankerFactory> _logger;
	private readonly Dictionary<string, string> _map = new();

	public RankerFactory(ILoggerFactory? loggerFactory = null)
	{
		_loggerFactory = loggerFactory ?? NullLoggerFactory.Instance;
		_logger = _loggerFactory.CreateLogger<RankerFactory>();

		// Register all predefined rankers
		_map[CreateRanker(RankerType.MART).Name.ToUpper()] = RankerType.MART.ToString();
		_map[CreateRanker(RankerType.RANKNET).Name.ToUpper()] = RankerType.RANKNET.ToString();
		_map[CreateRanker(RankerType.RANKBOOST).Name.ToUpper()] = RankerType.RANKBOOST.ToString();
		_map[CreateRanker(RankerType.ADARANK).Name.ToUpper()] = RankerType.ADARANK.ToString();
		_map[CreateRanker(RankerType.COOR_ASCENT).Name.ToUpper()] = RankerType.COOR_ASCENT.ToString();
		_map[CreateRanker(RankerType.LAMBDARANK).Name.ToUpper()] = RankerType.LAMBDARANK.ToString();
		_map[CreateRanker(RankerType.LAMBDAMART).Name.ToUpper()] = RankerType.LAMBDAMART.ToString();
		_map[CreateRanker(RankerType.LISTNET).Name.ToUpper()] = RankerType.LISTNET.ToString();
		_map[CreateRanker(RankerType.RANDOM_FOREST).Name.ToUpper()] = RankerType.RANDOM_FOREST.ToString();
		_map[CreateRanker(RankerType.LINEAR_REGRESSION).Name.ToUpper()] = RankerType.LINEAR_REGRESSION.ToString();
	}

	public void Register(string name, string className) => _map[name] = className;

	public Ranker CreateRanker(RankerType type) =>
		type switch
		{
			RankerType.MART => new MART(_loggerFactory.CreateLogger<MART>()),
			RankerType.RANKBOOST => new RankBoost(_loggerFactory.CreateLogger<RankBoost>()),
			RankerType.RANKNET => new RankNet(_loggerFactory.CreateLogger<RankNet>()),
			RankerType.ADARANK => new AdaRank(_loggerFactory.CreateLogger<AdaRank>()),
			RankerType.COOR_ASCENT => new CoorAscent(_loggerFactory.CreateLogger<CoorAscent>()),
			RankerType.LAMBDARANK => new LambdaRank(_loggerFactory.CreateLogger<LambdaRank>()),
			RankerType.LAMBDAMART => new LambdaMART(_loggerFactory.CreateLogger<LambdaMART>()),
			RankerType.LISTNET => new ListNet(_loggerFactory.CreateLogger<ListNet>()),
			RankerType.RANDOM_FOREST => new RFRanker(_loggerFactory),
			RankerType.LINEAR_REGRESSION => new LinearRegRank(_loggerFactory.CreateLogger<LinearRegRank>()),
			_ => throw new ArgumentOutOfRangeException(nameof(type), type, null)
		};

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
				throw RankLibException.Create($"Type '{typeName}' does not exist.");
			}

			ranker = (Ranker)Activator.CreateInstance(type, _loggerFactory.CreateLogger(type))!;
		}
		catch (TypeLoadException e)
		{
			throw RankLibException.Create($"Could not find the type \"{typeName}\" specified. Make sure the assembly is referenced.", e);
		}
		catch (MissingMethodException e)
		{
			throw RankLibException.Create($"Cannot create an instance from the type \"{typeName}\".", e);
		}
		catch (InvalidCastException e)
		{
			throw RankLibException.Create($"The class \"{typeName}\" does not derive from \"{typeof(Ranker).FullName}\".", e);
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
		var fullText = FileUtils.Read(modelFile, Encoding.ASCII);
		return LoadRankerFromString(fullText);
	}

	public Ranker LoadRankerFromString(string fullText)
	{
		try
		{
			using var reader = new StringReader(fullText);
			var rankerName = reader.ReadLine().Replace("## ", "").Trim(); // read the first line to get the ranker name
			_logger.LogInformation("Model: {RankerName}", rankerName);
			var r = CreateRanker(_map[rankerName.ToUpper()]);
			r.LoadFromString(fullText);
			return r;
		}
		catch (Exception ex)
		{
			throw RankLibException.Create(ex);
		}
	}
}
