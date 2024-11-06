using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Learning.Boosting;
using RankLib.Learning.NeuralNet;
using RankLib.Learning.Tree;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning;

/// <summary>
/// Factory for creating <see cref="IRanker"/> instances.
/// </summary>
public class RankerFactory
{
	private readonly ILoggerFactory _loggerFactory;
	private readonly ILogger<RankerFactory> _logger;

	private readonly Dictionary<string, Type> _map = new(StringComparer.OrdinalIgnoreCase)
	{
		[MART.RankerName] = typeof(MART),
		[RankNet.RankerName] = typeof(RankNet),
		[RankBoost.RankerName] = typeof(RankBoost),
		[AdaRank.RankerName] = typeof(AdaRank),
		[CoordinateAscent.RankerName] = typeof(CoordinateAscent),
		[LambdaRank.RankerName] = typeof(LambdaRank),
		[LambdaMART.RankerName] = typeof(LambdaMART),
		[ListNet.RankerName] = typeof(ListNet),
		[RFRanker.RankerName] = typeof(RFRanker),
		[LinearRegRank.RankerName] = typeof(LinearRegRank),
	};

	private readonly Dictionary<Type, Func<ILoggerFactory, IRanker>> _rankers = new()
	{
		[typeof(MART)] = loggerFactory => new MART(loggerFactory.CreateLogger<MART>()),
		[typeof(RankNet)] = loggerFactory => new RankNet(loggerFactory.CreateLogger<RankNet>()),
		[typeof(RankBoost)] = loggerFactory => new RankBoost(loggerFactory.CreateLogger<RankBoost>()),
		[typeof(AdaRank)] = loggerFactory => new AdaRank(loggerFactory.CreateLogger<AdaRank>()),
		[typeof(CoordinateAscent)] = loggerFactory => new CoordinateAscent(loggerFactory.CreateLogger<CoordinateAscent>()),
		[typeof(LambdaRank)] = loggerFactory => new LambdaRank(loggerFactory.CreateLogger<LambdaRank>()),
		[typeof(LambdaMART)] = loggerFactory => new LambdaMART(loggerFactory.CreateLogger<LambdaMART>()),
		[typeof(ListNet)] = loggerFactory => new ListNet(loggerFactory.CreateLogger<ListNet>()),
		[typeof(RFRanker)] = loggerFactory => new RFRanker(loggerFactory),
		[typeof(LinearRegRank)] = loggerFactory => new LinearRegRank(loggerFactory.CreateLogger<LinearRegRank>()),
	};

	public RankerFactory(ILoggerFactory? loggerFactory = null)
	{
		_loggerFactory = loggerFactory ?? NullLoggerFactory.Instance;
		_logger = _loggerFactory.CreateLogger<RankerFactory>();
	}

	public void AddRanker<TRanker, TRankerParameters>(Func<ILoggerFactory, TRanker> factory)
		where TRanker : class, IRanker<TRankerParameters>
		where TRankerParameters : IRankerParameters
	{
		if (!_rankers.TryAdd(typeof(TRanker), factory))
			throw new ArgumentException($"Ranker of type '{typeof(TRanker).Name}' is already registered.");
	}

	public IRanker CreateRanker(RankerType type) => CreateRanker(type.GetRankerType());

	public TRanker CreateRanker<TRanker>()
		where TRanker : IRanker
	{
		if (!_rankers.TryGetValue(typeof(TRanker), out var factory))
			throw new ArgumentException($"Ranker of type '{typeof(TRanker).Name}' is not registered.");

		return (TRanker)factory(_loggerFactory);
	}

	public IRanker CreateRanker(Type rankerType)
	{
		if (!typeof(IRanker).IsAssignableFrom(rankerType))
			throw new ArgumentException($"Ranker of type '{rankerType.Name}' is not a ranker. It must implement {typeof(IRanker<>).FullName}.", nameof(rankerType));

		if (!_rankers.TryGetValue(rankerType, out var factory))
			throw new ArgumentException($"Ranker of type '{rankerType.Name}' is not registered.", nameof(rankerType));

		return factory(_loggerFactory);
	}

	public TRanker CreateRanker<TRanker, TRankerParameters>(List<RankList> samples, int[] features, MetricScorer scorer, TRankerParameters? parameters = default)
		where TRankerParameters : IRankerParameters
		where TRanker : IRanker<TRankerParameters>
	{
		var ranker = CreateRanker<TRanker>();
		ranker.Features = features;
		ranker.Samples = samples;
		ranker.Scorer = scorer;
		if (parameters != null)
			ranker.Parameters = parameters;

		return ranker;
	}

	public IRanker CreateRanker(Type rankerType, List<RankList> samples, int[] features, MetricScorer scorer, IRankerParameters? parameters = default)
	{
		var ranker = CreateRanker(rankerType);
		ranker.Features = features;
		ranker.Samples = samples;
		ranker.Scorer = scorer;
		if (parameters != null)
		{
			if (!ranker.Parameters.GetType().IsInstanceOfType(parameters))
			{
				throw new ArgumentException(
					$"Parameters of type '{parameters.GetType().Name}' is not assignable to " +
					$"ranker parameters '{ranker.Parameters.GetType().Name}'.");
			}

			ranker.Parameters = parameters;
		}

		return ranker;
	}

	public IRanker CreateRanker(RankerType type, List<RankList> samples, int[] features, MetricScorer scorer, IRankerParameters? parameters = default) =>
		CreateRanker(type.GetRankerType(), samples, features, scorer, parameters);

	public IRanker LoadRankerFromFile(string modelFile)
	{
		var fullText = FileUtils.Read(modelFile, Encoding.ASCII);
		return LoadRankerFromString(fullText);
	}

	public IRanker LoadRankerFromString(string model)
	{
		// read the first line to get the ranker name
		var modelSpan = model.AsSpan();
		var endOfLine = modelSpan.IndexOfAny('\r', '\n');

		if (endOfLine == -1)
			throw new ArgumentException($"Invalid model '{model}'.", nameof(model));

		var firstLine = modelSpan.Slice(0, endOfLine);
		var lastHash = firstLine.LastIndexOf('#');

		if (lastHash == -1)
			throw new ArgumentException($"Expected to find model name on first line, but found '{firstLine}'.", nameof(model));

		var rankerName = firstLine.Slice(lastHash + 1).Trim().ToString();
		_logger.LogInformation("Model: {RankerName}", rankerName);

		if (!_map.TryGetValue(rankerName, out var rankerType))
			throw new ArgumentException($"Ranker with name '{rankerName}' is not registered.");

		if (!_rankers.TryGetValue(rankerType, out var factory))
			throw new ArgumentException($"Ranker of type '{rankerName}' is not registered.");

		var ranker = factory(_loggerFactory);
		ranker.LoadFromString(model);
		return ranker;
	}
}
