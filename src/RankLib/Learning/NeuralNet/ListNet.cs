using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Logging;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning.NeuralNet;

public class ListNet : RankNet
{
    private static readonly ILogger logger = NullLogger.Instance;

    // Parameters
    public static int nIteration = 1500;
    public static double learningRate = 0.00001;
    public static int nHiddenLayer = 0; // FIXED, it doesn't work with hidden layer

    public ListNet() { }

    public ListNet(List<RankList> samples, int[] features, MetricScorer scorer)
        : base(samples, features, scorer)
    { }

    protected float[] FeedForward(RankList rl)
    {
        var labels = new float[rl.Size()];
        for (int i = 0; i < rl.Size(); i++)
        {
            AddInput(rl.Get(i));
            Propagate(i);
            labels[i] = rl.Get(i).GetLabel();
        }
        return labels;
    }

    protected void BackPropagate(float[] labels)
    {
        // Back-propagate
        var p = new PropParameter(labels);
        _outputLayer.ComputeDelta(p); // Starting at the output layer

        // Weight update
        _outputLayer.UpdateWeight(p);
    }

    protected override void EstimateLoss()
    {
        _error = 0.0;
        double sumLabelExp = 0;
        double sumScoreExp = 0;
        for (int i = 0; i < _samples.Count; i++)
        {
            RankList rl = _samples[i];
            var scores = new double[rl.Size()];
            double err = 0;
            for (int j = 0; j < rl.Size(); j++)
            {
                scores[j] = Eval(rl.Get(j));
                sumLabelExp += Math.Exp(rl.Get(j).GetLabel());
                sumScoreExp += Math.Exp(scores[j]);
            }
            for (int j = 0; j < rl.Size(); j++)
            {
                double p1 = Math.Exp(rl.Get(j).GetLabel()) / sumLabelExp;
                double p2 = Math.Exp(scores[j]) / sumScoreExp;
                err += -p1 * SimpleMath.LogBase2(p2);
            }
            _error += err / rl.Size();
        }
        _lastError = _error;
    }

    public override void Init()
    {
        logger.LogInformation("Initializing...");

        // Set up the network
        SetInputOutput(_features.Length, 1, 1);
        Wire();

        if (_validationSamples != null)
        {
            for (int i = 0; i < _layers.Count; i++)
            {
                _bestModelOnValidation.Add(new List<double>());
            }
        }

        Neuron.LearningRate = learningRate;
    }

    public override void Learn()
    {
        logger.LogInformation("Training starts...");
        PrintLogLn(new[] { 7, 14, 9, 9 }, new[] { "#epoch", "C.E. Loss", _scorer.Name() + "-T", _scorer.Name() + "-V" });

        for (int i = 1; i <= nIteration; i++)
        {
            for (int j = 0; j < _samples.Count; j++)
            {
                float[] labels = FeedForward(_samples[j]);
                BackPropagate(labels);
                ClearNeuronOutputs();
            }

            PrintLog(new[] { 7, 14 }, new[] { i.ToString(), SimpleMath.Round(_error, 6).ToString() });

            if (i % 1 == 0)
            {
                _scoreOnTrainingData = _scorer.Score(Rank(_samples));
                PrintLog(new[] { 9 }, new[] { SimpleMath.Round(_scoreOnTrainingData, 4).ToString() });

                if (_validationSamples != null)
                {
                    double score = _scorer.Score(Rank(_validationSamples));
                    if (score > _bestScoreOnValidationData)
                    {
                        _bestScoreOnValidationData = score;
                        SaveBestModelOnValidation();
                    }
                    PrintLog(new[] { 9 }, new[] { SimpleMath.Round(score, 4).ToString() });
                }
            }
            FlushLog();
        }

        // Restore the best model if validation data is used
        if (_validationSamples != null)
        {
            RestoreBestModelOnValidation();
        }

        _scoreOnTrainingData = SimpleMath.Round(_scorer.Score(Rank(_samples)), 4);
        logger.LogInformation("Finished successfully.");
        logger.LogInformation($"{_scorer.Name} on training data: {_scoreOnTrainingData}");

        if (_validationSamples != null)
        {
            _bestScoreOnValidationData = _scorer.Score(Rank(_validationSamples));
            logger.LogInformation($"{_scorer.Name} on validation data: {SimpleMath.Round(_bestScoreOnValidationData, 4)}");
        }
    }

    public override double Eval(DataPoint p)
    {
        return base.Eval(p);
    }

    public override Ranker CreateNew()
    {
        return new ListNet();
    }

    public override string ToString()
    {
        return base.ToString();
    }

    public override string Model()
    {
        var output = new System.Text.StringBuilder();
        output.Append($"## {Name()}\n");
        output.Append($"## Epochs = {nIteration}\n");
        output.Append($"## No. of features = {_features.Length}\n");

        // Print used features
        for (int i = 0; i < _features.Length; i++)
        {
            output.Append(_features[i] + (i == _features.Length - 1 ? "" : " "));
        }
        output.Append("\n");

        // Print network information
        output.Append("0\n"); // [# hidden layers, *ALWAYS* 0 since we're using linear net]
        // Print learned weights
        output.Append(ToString());
        return output.ToString();
    }

    public override void LoadFromString(string fullText)
    {
        try
        {
            using (var inStream = new StringReader(fullText))
            {
                string content;
                var l = new List<string>();
                while ((content = inStream.ReadLine()) != null)
                {
                    content = content.Trim();
                    if (string.IsNullOrEmpty(content) || content.StartsWith("##")) continue;
                    l.Add(content);
                }

                // Load the network
                // The first line contains feature information
                var tmp = l[0].Split(' ');
                _features = new int[tmp.Length];
                for (int i = 0; i < tmp.Length; i++)
                {
                    _features[i] = int.Parse(tmp[i]);
                }

                // The 2nd line is a scalar indicating the number of hidden layers
                int nHiddenLayer = int.Parse(l[1]);
                var nn = new int[nHiddenLayer];

                // The next @nHiddenLayer lines contain the number of neurons in each layer
                int index = 2;
                for (; index < 2 + nHiddenLayer; index++)
                {
                    nn[index - 2] = int.Parse(l[index]);
                }

                // Create the network
                SetInputOutput(_features.Length, 1);
                for (int j = 0; j < nHiddenLayer; j++)
                {
                    AddHiddenLayer(nn[j]);
                }
                Wire();

                // Fill in weights
                for (; index < l.Count; index++) // Loop through all layers
                {
                    var s = l[index].Split(' ');
                    int iLayer = int.Parse(s[0]); // Which layer?
                    int iNeuron = int.Parse(s[1]); // Which neuron?
                    var n = _layers[iLayer].Get(iNeuron);
                    for (int k = 0; k < n.GetOutLinks().Count; k++)
                    {
                        n.GetOutLinks()[k].Weight = double.Parse(s[k + 2]);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            throw RankLibError.Create("Error in ListNet::LoadFromString(): ", ex);
        }
    }

    public override void PrintParameters()
    {
        logger.LogInformation($"No. of epochs: {nIteration}");
        logger.LogInformation($"Learning rate: {learningRate}");
    }

    public override string Name()
    {
        return "ListNet";
    }
}
