using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Parsing;
using RankLib.Utilities;

namespace RankLib.Learning.Tree;

public class RFRanker : Ranker
{
    private static readonly ILogger<RFRanker> logger = NullLogger<RFRanker>.Instance;

    // Parameters
    // [a] general bagging parameters
    public static int nBag = 300;
    public static float subSamplingRate = 1.0f; // sampling of samples (*WITH* replacement)
    public static float featureSamplingRate = 0.3f; // sampling of features (*WITHOUT* replacement)

    // [b] what to do in each bag
    public static RankerType rType = RankerType.MART; // which algorithm to bag
    public static int nTrees = 1; // how many trees in each bag
    public static int nTreeLeaves = 100;
    public static float learningRate = 0.1F; // or shrinkage, only matters if nTrees > 1
    public static int nThreshold = 256;
    public static int minLeafSupport = 1;

    // Variables
    protected Ensemble[] ensembles = null; // bag of ensembles

    public RFRanker() : base() { }

    public RFRanker(List<RankList> samples, int[] features, MetricScorer scorer) 
        : base(samples, features, scorer) { }

    public override void Init()
    {
        logger.LogInformation("Initializing...");
        ensembles = new Ensemble[nBag];

        // Initialize parameters for the tree(s) built in each bag
        LambdaMART.nTrees = nTrees;
        LambdaMART.nTreeLeaves = nTreeLeaves;
        LambdaMART.learningRate = learningRate;
        LambdaMART.nThreshold = nThreshold;
        LambdaMART.minLeafSupport = minLeafSupport;
        LambdaMART.nRoundToStopEarly = -1; // no early stopping since we're doing bagging

        // Turn on feature sampling
        FeatureHistogram.samplingRate = featureSamplingRate;
    }

    public override void Learn()
    {
        RankerFactory rf = new RankerFactory();
        logger.LogInformation("Training starts...");
        PrintLogLn(new int[] { 9, 9, 11 }, new string[] { "bag", _scorer.Name() + "-B", _scorer.Name() + "-OOB" });

        double[] impacts = null;

        // Start the bagging process
        for (int i = 0; i < nBag; i++)
        {
            Sampler sp = new Sampler();
            // Create a "bag" of samples by random sampling from the training set
            List<RankList> bag = sp.DoSampling(_samples, subSamplingRate, true);
            LambdaMART r = (LambdaMART)rf.CreateRanker(rType, bag, _features, _scorer);

            r.Init();
            r.Learn();

            // Accumulate impacts
            if (impacts == null)
            {
                impacts = r.impacts;
            }
            else
            {
                for (int ftr = 0; ftr < impacts.Length; ftr++)
                {
                    impacts[ftr] += r.impacts[ftr];
                }
            }
            PrintLogLn(new int[] { 9, 9 }, new string[] { "b[" + (i + 1) + "]", SimpleMath.Round(r.GetScoreOnTrainingData(), 4).ToString() });
            ensembles[i] = r.GetEnsemble();
        }

        // Finishing up
        _scoreOnTrainingData = _scorer.Score(Rank(_samples));
        logger.LogInformation("Finished successfully.");
        logger.LogInformation(_scorer.Name() + " on training data: " + SimpleMath.Round(_scoreOnTrainingData, 4));

        if (_validationSamples != null)
        {
            _bestScoreOnValidationData = _scorer.Score(Rank(_validationSamples));
            logger.LogInformation(_scorer.Name() + " on validation data: " + SimpleMath.Round(_bestScoreOnValidationData, 4));
        }

        // Print feature impacts
        logger.LogInformation("-- FEATURE IMPACTS");
        if (logger.IsEnabled(LogLevel.Information))
        {
            int[] ftrsSorted = MergeSorter.Sort(impacts, false);
            foreach (int ftr in ftrsSorted)
            {
                logger.LogInformation(" Feature " + _features[ftr] + " reduced error " + impacts[ftr]);
            }
        }
    }

    public override double Eval(DataPoint dp)
    {
        double s = 0;
        foreach (Ensemble ensemble in ensembles)
        {
            s += ensemble.Eval(dp);
        }
        return s / ensembles.Length;
    }

    public override Ranker CreateNew()
    {
        return new RFRanker();
    }

    public override string ToString()
    {
        StringBuilder str = new StringBuilder();
        for (int i = 0; i < nBag; i++)
        {
            str.Append(ensembles[i]).Append("\n");
        }
        return str.ToString();
    }

    public override string Model()
    {
        StringBuilder output = new StringBuilder();
        output.Append("## " + Name() + "\n");
        output.Append("## No. of bags = " + nBag + "\n");
        output.Append("## Sub-sampling = " + subSamplingRate + "\n");
        output.Append("## Feature-sampling = " + featureSamplingRate + "\n");
        output.Append("## No. of trees = " + nTrees + "\n");
        output.Append("## No. of leaves = " + nTreeLeaves + "\n");
        output.Append("## No. of threshold candidates = " + nThreshold + "\n");
        output.Append("## Learning rate = " + learningRate + "\n\n");
        output.Append(ToString());
        return output.ToString();
    }

    public override void LoadFromString(string fullText)
    {
        List<Ensemble> ens = new List<Ensemble>();
        ModelLineProducer lineByLine = new ModelLineProducer();

        lineByLine.Parse(fullText, (model, maybeEndEns) => 
        {
            if (maybeEndEns && model.ToString().EndsWith("</ensemble>"))
            {
                ens.Add(new Ensemble(model.ToString()));
                model.Clear();
            }
        });

        ISet<int> uniqueFeatures = new HashSet<int>();
        ensembles = new Ensemble[ens.Count];
        for (int i = 0; i < ens.Count; i++)
        {
            ensembles[i] = ens[i];

            // Obtain used features
            int[] fids = ens[i].GetFeatures();
            foreach (int fid in fids)
            {
                uniqueFeatures.Add(fid);
            }
        }

        _features = uniqueFeatures.ToArray();
    }

    public override void PrintParameters()
    {
        logger.LogInformation("No. of bags: " + nBag);
        logger.LogInformation("Sub-sampling: " + subSamplingRate);
        logger.LogInformation("Feature-sampling: " + featureSamplingRate);
        logger.LogInformation("No. of trees: " + nTrees);
        logger.LogInformation("No. of leaves: " + nTreeLeaves);
        logger.LogInformation("No. of threshold candidates: " + nThreshold);
        logger.LogInformation("Learning rate: " + learningRate);
    }

    public override string Name()
    {
        return "Random Forests";
    }

    public Ensemble[] GetEnsembles()
    {
        return ensembles;
    }
}