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
    private static readonly ILogger logger = NullLogger.Instance;

    // Factory for creating Ranker instances
    protected Ranker[] rFactory = new Ranker[]
    {
        new MART(),
        new RankBoost(),
        new RankNet(),
        new AdaRank(),
        new CoorAscent(),
        new LambdaRank(),
        new LambdaMART(),
        new ListNet(),
        new RFRanker(),
        new LinearRegRank()
    };

    // Map for custom registered rankers
    protected Dictionary<string, string> map = new();

    public RankerFactory()
    {
        // Register all predefined rankers
        map[CreateRanker(RankerType.MART).Name().ToUpper()] = RankerType.MART.ToString();
        map[CreateRanker(RankerType.RANKNET).Name().ToUpper()] = RankerType.RANKNET.ToString();
        map[CreateRanker(RankerType.RANKBOOST).Name().ToUpper()] = RankerType.RANKBOOST.ToString();
        map[CreateRanker(RankerType.ADARANK).Name().ToUpper()] = RankerType.ADARANK.ToString();
        map[CreateRanker(RankerType.COOR_ASCENT).Name().ToUpper()] = RankerType.COOR_ASCENT.ToString();
        map[CreateRanker(RankerType.LAMBDARANK).Name().ToUpper()] = RankerType.LAMBDARANK.ToString();
        map[CreateRanker(RankerType.LAMBDAMART).Name().ToUpper()] = RankerType.LAMBDAMART.ToString();
        map[CreateRanker(RankerType.LISTNET).Name().ToUpper()] = RankerType.LISTNET.ToString();
        map[CreateRanker(RankerType.RANDOM_FOREST).Name().ToUpper()] = RankerType.RANDOM_FOREST.ToString();
        map[CreateRanker(RankerType.LINEAR_REGRESSION).Name().ToUpper()] = RankerType.LINEAR_REGRESSION.ToString();
    }

    public void Register(string name, string className)
    {
        map[name] = className;
    }

    public Ranker CreateRanker(RankerType type)
    {
        return rFactory[(int)type - (int)RankerType.MART].CreateNew();
    }

    public Ranker CreateRanker(RankerType type, List<RankList> samples, int[] features, MetricScorer scorer)
    {
        Ranker r = CreateRanker(type);
        r.SetTrainingSet(samples);
        r.SetFeatures(features);
        r.SetMetricScorer(scorer);
        return r;
    }

    public Ranker CreateRanker(string className)
    {
        try
        {
            RankerType rankerType = (RankerType)Enum.Parse(typeof(RankerType), className);
            return CreateRanker(rankerType);
        }
        catch
        {
            // ignored
        }

        Ranker r = null;
        try
        {
            Type type = Type.GetType(className);
            r = (Ranker)Activator.CreateInstance(type);
        }
        catch (TypeLoadException e)
        {
            throw RankLibError.Create($"Could not find the class \"{className}\" specified. Make sure the assembly is referenced.", e);
        }
        catch (MissingMethodException e)
        {
            throw RankLibError.Create($"Cannot create an instance from the class \"{className}\".", e);
        }
        catch (InvalidCastException e)
        {
            throw RankLibError.Create($"The class \"{className}\" does not implement the Ranker interface.", e);
        }
        return r;
    }

    public Ranker CreateRanker(string className, List<RankList> samples, int[] features, MetricScorer scorer)
    {
        Ranker r = CreateRanker(className);
        r.SetTrainingSet(samples);
        r.SetFeatures(features);
        r.SetMetricScorer(scorer);
        return r;
    }

    public Ranker LoadRankerFromFile(string modelFile)
    {
        string fullText = FileUtils.Read(modelFile, "ASCII");
        return LoadRankerFromString(fullText);
    }

    public Ranker LoadRankerFromString(string fullText)
    {
        try
        {
            using (StringReader reader = new StringReader(fullText))
            {
                string content = reader.ReadLine().Replace("## ", "").Trim(); // read the first line to get the ranker name
                logger.LogInformation($"Model: {content}");
                Ranker r = CreateRanker(map[content.ToUpper()]);
                r.LoadFromString(fullText);
                return r;
            }
        }
        catch (Exception ex)
        {
            throw RankLibError.Create(ex);
        }
    }
}