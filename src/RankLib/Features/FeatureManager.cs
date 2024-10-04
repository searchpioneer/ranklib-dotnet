﻿using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Learning;
using RankLib.Utilities;

namespace RankLib.Features;

public class FeatureManager
{
    // Fields
    private static readonly ILogger<FeatureManager> Logger = NullLogger<FeatureManager>.Instance;

    // Static main method
    public static void Main(string[] args)
    {
        var rankingFiles = new List<string>();
        string outputDir = "";
        string modelFileName = "";
        bool shuffle = false;
        bool doFeatureStats = false;

        int nFold = 0;
        float tvs = -1; // train-validation split in each fold
        float tts = -1; // train-test validation split of the whole dataset
        int argsLen = args.Length;

        if ((argsLen < 3 && !args.Contains("-feature_stats")) || (argsLen != 2 && args.Contains("-feature_stats")))
        {
            Logger.LogInformation("Usage: dotnet run ciir.umass.edu.features.FeatureManager <Params>");
            Logger.LogInformation("Params:");
            Logger.LogInformation("\t-input <file>\t\tSource data (ranked lists)");
            Logger.LogInformation("\t-output <dir>\t\tThe output directory");
            Logger.LogInformation("  [+] Shuffling");
            Logger.LogInformation("\t-shuffle\t\tCreate a copy of the input file in which the ordering of all ranked lists is randomized.");
            Logger.LogInformation("  [+] k-fold Partitioning (sequential split)");
            Logger.LogInformation("\t-k <fold>\t\tThe number of folds");
            Logger.LogInformation("\t[ -tvs <x \\in [0..1]> ] Train-validation split ratio (x)(1.0-x)");
            Logger.LogInformation("  [+] Train-test split");
            Logger.LogInformation("\t-tts <x \\in [0..1]> ] Train-test split ratio (x)(1.0-x)");

            Logger.LogInformation("  NOTE: If both -shuffle and -k are specified, the input data will be shuffled and then sequentially partitioned.");
            Logger.LogInformation("Feature Statistics -- Saved model feature use frequencies and statistics.");
            return;
        }

        for (int i = 0; i < args.Length; i++)
        {
            if (args[i].Equals("-input", StringComparison.OrdinalIgnoreCase))
            {
                rankingFiles.Add(args[++i]);
            }
            else if (args[i].Equals("-k", StringComparison.OrdinalIgnoreCase))
            {
                nFold = int.Parse(args[++i]);
            }
            else if (args[i].Equals("-shuffle", StringComparison.OrdinalIgnoreCase))
            {
                shuffle = true;
            }
            else if (args[i].Equals("-tvs", StringComparison.OrdinalIgnoreCase))
            {
                tvs = float.Parse(args[++i]);
            }
            else if (args[i].Equals("-tts", StringComparison.OrdinalIgnoreCase))
            {
                tts = float.Parse(args[++i]);
            }
            else if (args[i].Equals("-output", StringComparison.OrdinalIgnoreCase))
            {
                outputDir = FileUtils.MakePathStandard(args[++i]);
            }
            else if (args[i].Equals("-feature_stats", StringComparison.OrdinalIgnoreCase))
            {
                doFeatureStats = true;
                modelFileName = args[++i];
            }
        }

        if (nFold > 0 && tts != -1)
        {
            Logger.LogInformation("Error: Only one of k or tts should be specified.");
            return;
        }

        if (shuffle || nFold > 0 || tts != -1)
        {
            var samples = ReadInput(rankingFiles);

            if (!samples.Any())
            {
                Logger.LogInformation("Error: The input file is empty.");
                return;
            }

            string fn = FileUtils.GetFileName(rankingFiles[0]);

            if (shuffle)
            {
                fn += ".shuffled";
                Logger.LogInformation("Shuffling... ");
                samples.Shuffle();
                Logger.LogInformation("Saving... ");
                Save(samples, Path.Combine(outputDir, fn));
            }

            if (tts != -1)
            {
                var trains = new List<RankList>();
                var tests = new List<RankList>();

                Logger.LogInformation("Splitting... ");
                PrepareSplit(samples, tts, trains, tests);

                try
                {
                    Logger.LogInformation("Saving splits...");
                    Save(trains, Path.Combine(outputDir, $"train.{fn}"));
                    Save(tests, Path.Combine(outputDir, $"test.{fn}"));
                }
                catch (Exception ex)
                {
                    throw RankLibError.Create("Cannot save partition data.\nOccurred in FeatureManager::main(): ", ex);
                }
            }

            if (nFold > 0)
            {
                var trains = new List<List<RankList>>();
                var tests = new List<List<RankList>>();
                var valis = new List<List<RankList>>();
                Logger.LogInformation("Partitioning... ");
                PrepareCV(samples, nFold, tvs, trains, valis, tests);

                try
                {
                    for (int i = 0; i < trains.Count; i++)
                    {
                        Logger.LogInformation($"Saving fold {i + 1}/{nFold}... ");
                        Save(trains[i], Path.Combine(outputDir, $"f{i + 1}.train.{fn}"));
                        Save(tests[i], Path.Combine(outputDir, $"f{i + 1}.test.{fn}"));
                        if (tvs > 0)
                        {
                            Save(valis[i], Path.Combine(outputDir, $"f{i + 1}.validation.{fn}"));
                        }
                    }
                }
                catch (Exception ex)
                {
                    throw RankLibError.Create("Cannot save partition data.\nOccurred in FeatureManager::main(): ", ex);
                }
            }
        }
        else if (doFeatureStats)
        {
            try
            {
                var fs = new FeatureStats(modelFileName);
                fs.WriteFeatureStats();
            }
            catch (Exception ex)
            {
                throw RankLibError.Create($"Failure processing saved {modelFileName} model file.\nError occurred in FeatureManager::main(): ", ex);
            }
        }
    }

    public static List<RankList> ReadInput(string inputFile)
    {
        return ReadInput(inputFile, false, false);
    }

    public static List<RankList> ReadInput(string inputFile, bool mustHaveRelDoc, bool useSparseRepresentation)
    {
        var samples = new List<RankList>(1000);
        int countEntries = 0;

        try
        {
            using (var inStream = FileUtils.SmartReader(inputFile))
            {
                string content = null;
                string lastID = "";
                bool hasRel = false;
                var rl = new List<DataPoint>(10000);

                while ((content = inStream.ReadLine()) != null)
                {
                    content = content.Trim();
                    if (content.Length == 0 || content[0] == '#')
                    {
                        continue;
                    }

                    if (countEntries % 10000 == 0)
                    {
                        Logger.LogInformation($"Reading feature file [{inputFile}]: {countEntries}...");
                    }

                    DataPoint qp = useSparseRepresentation ? new SparseDataPoint(content) : new DenseDataPoint(content);

                    if (!string.IsNullOrEmpty(lastID) && !lastID.Equals(qp.GetID(), StringComparison.OrdinalIgnoreCase))
                    {
                        if (!mustHaveRelDoc || hasRel)
                        {
                            samples.Add(new RankList(rl));
                        }
                        rl = new List<DataPoint>();
                        hasRel = false;
                    }

                    if (qp.GetLabel() > 0)
                    {
                        hasRel = true;
                    }

                    lastID = qp.GetID();
                    rl.Add(qp);
                    countEntries++;
                }

                if (rl.Any() && (!mustHaveRelDoc || hasRel))
                {
                    samples.Add(new RankList(rl));
                }

                Logger.LogInformation($"Reading feature file [{inputFile}] completed. (Read {samples.Count} ranked lists, {countEntries} entries)");
            }
        }
        catch (Exception ex)
        {
            throw RankLibError.Create("Error in FeatureManager::readInput(): ", ex);
        }

        return samples;
    }

    public static List<RankList> ReadInput(List<string> inputFiles)
    {
        var samples = new List<RankList>();
        foreach (var inputFile in inputFiles)
        {
            var s = ReadInput(inputFile, false, false);
            samples.AddRange(s);
        }
        return samples;
    }

    public static int[] ReadFeature(string featureDefFile)
    {
        int[] features = null;
        var fids = new List<string>();

        try
        {
            using (var inStream = FileUtils.SmartReader(featureDefFile))
            {
                string content = null;
                while ((content = inStream.ReadLine()) != null)
                {
                    content = content.Trim();
                    if (content.Length == 0 || content[0] == '#')
                    {
                        continue;
                    }
                    fids.Add(content.Split('\t')[0].Trim());
                }
            }

            features = fids.Select(fid => int.Parse(fid)).ToArray();
        }
        catch (IOException ex)
        {
            throw RankLibError.Create("Error in FeatureManager::readFeature(): ", ex);
        }

        return features;
    }

    public static int[] GetFeatureFromSampleVector(List<RankList> samples)
    {
        if (!samples.Any())
        {
            throw RankLibError.Create("Error in FeatureManager::getFeatureFromSampleVector(): There are no training samples.");
        }

        int maxFeatureCount = samples.Max(rl => rl.GetFeatureCount());
        var features = Enumerable.Range(1, maxFeatureCount).ToArray();

        return features;
    }

    public static void PrepareCV(List<RankList> samples, int nFold, List<List<RankList>> trainingData, List<List<RankList>> testData)
    {
        PrepareCV(samples, nFold, -1, trainingData, null, testData);
    }

    public static void PrepareCV(List<RankList> samples, int nFold, float tvs, List<List<RankList>> trainingData, List<List<RankList>> validationData, List<List<RankList>> testData)
    {
        var trainSamplesIdx = new List<List<int>>();
        int size = samples.Count / nFold;
        int start = 0;
        int total = 0;

        for (int f = 0; f < nFold; f++)
        {
            var foldIndexes = new List<int>();
            for (int i = 0; i < size && start + i < samples.Count; i++)
            {
                foldIndexes.Add(start + i);
            }
            trainSamplesIdx.Add(foldIndexes);
            total += foldIndexes.Count;
            start += size;
        }

        while (total < samples.Count)
        {
            trainSamplesIdx.Last().Add(total++);
        }

        foreach (var indexes in trainSamplesIdx)
        {
            Logger.LogInformation($"Creating data for fold {trainSamplesIdx.IndexOf(indexes) + 1}/{nFold}...");
            var train = new List<RankList>();
            var test = new List<RankList>();
            var validation = new List<RankList>();

            foreach (var index in Enumerable.Range(0, samples.Count))
            {
                if (indexes.Contains(index))
                {
                    test.Add(new RankList(samples[index]));
                }
                else
                {
                    train.Add(new RankList(samples[index]));
                }
            }

            if (tvs > 0)
            {
                int validationSize = (int)(train.Count * (1.0f - tvs));
                for (int i = 0; i < validationSize; i++)
                {
                    validation.Add(train.Last());
                    train.RemoveAt(train.Count - 1);
                }
            }

            trainingData.Add(train);
            testData.Add(test);

            if (tvs > 0)
            {
                validationData.Add(validation);
            }
        }

        Logger.LogInformation($"Creating data for {nFold} folds completed.");
        PrintQueriesForSplit("Train", trainingData);
        PrintQueriesForSplit("Validate", validationData);
        PrintQueriesForSplit("Test", testData);
    }

    public static void PrintQueriesForSplit(string name, List<List<RankList>>? split)
    {
        if (split == null)
        {
            Logger.LogInformation($"No {name} split.");
            return;
        }

        foreach (var rankLists in split)
        {
            Logger.LogInformation($"{name} [{split.IndexOf(rankLists)}] = ");
            foreach (var rankList in rankLists)
            {
                Logger.LogInformation($" \"{rankList.GetID()}\"");
            }
        }
    }

    public static void PrepareSplit(List<RankList> samples, double percentTrain, List<RankList> trainingData, List<RankList> testData)
    {
        int size = (int)(samples.Count * percentTrain);

        for (int i = 0; i < size; i++)
        {
            trainingData.Add(new RankList(samples[i]));
        }

        for (int i = size; i < samples.Count; i++)
        {
            testData.Add(new RankList(samples[i]));
        }
    }

    public static void Save(List<RankList> samples, string outputFile)
    {
        try
        {
            using var outStream = new StreamWriter(outputFile);
            foreach (var sample in samples)
            {
                Save(sample, outStream);
            }
        }
        catch (Exception ex)
        {
            throw RankLibError.Create("Error in FeatureManager::save(): ", ex);
        }
    }

    private static void Save(RankList r, StreamWriter outStream)
    {
        for (int i = 0; i < r.Size(); i++)
        {
            var dataPoint = r.Get(i);
            outStream.WriteLine(dataPoint.ToString());
        }
    }
}