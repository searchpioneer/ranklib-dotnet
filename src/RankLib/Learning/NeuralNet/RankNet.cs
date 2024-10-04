using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning.NeuralNet;

public class RankNet : Ranker
{
    // TODO: logging
    private static readonly ILogger<RankNet> _logger = NullLogger<RankNet>.Instance;

    public static int NIteration { get; set; } = 100;
    public static int NHiddenLayer { get; set; } = 1;
    public static int NHiddenNodePerLayer { get; set; } = 10;
    public static double LearningRate { get; set; } = 0.00005;

    protected List<Layer> _layers = new();
    protected Layer _inputLayer;
    protected Layer _outputLayer;

    protected List<List<double>> _bestModelOnValidation = new();
    protected int _totalPairs;
    protected int _misorderedPairs;
    protected double _error;
    protected double _lastError = double.MaxValue;
    protected int _straightLoss = 0;

    public RankNet() { }

    public RankNet(List<RankList> samples, int[] features, MetricScorer scorer) : base(samples, features, scorer)
    {
    }

    protected void SetInputOutput(int nInput, int nOutput)
    {
        _inputLayer = new Layer(nInput + 1);
        _outputLayer = new Layer(nOutput);
        _layers.Clear();
        _layers.Add(_inputLayer);
        _layers.Add(_outputLayer);
    }
    
    protected void SetInputOutput(int nInput, int nOutput, int nType)
    {
        _inputLayer = new Layer(nInput + 1, nType);
        _outputLayer = new Layer(nOutput, nType);
        _layers.Clear();
        _layers.Add(_inputLayer);
        _layers.Add(_outputLayer);
    }

    protected void AddHiddenLayer(int size)
    {
        _layers.Insert(_layers.Count - 1, new Layer(size));
    }

    protected void Wire()
    {
        for (int i = 0; i < _inputLayer.Size() - 1; i++)
        {
            for (int j = 0; j < _layers[1].Size(); j++)
            {
                Connect(0, i, 1, j);
            }
        }

        for (int i = 1; i < _layers.Count - 1; i++)
        {
            for (int j = 0; j < _layers[i].Size(); j++)
            {
                for (int k = 0; k < _layers[i + 1].Size(); k++)
                {
                    Connect(i, j, i + 1, k);
                }
            }
        }

        for (int i = 1; i < _layers.Count; i++)
        {
            for (int j = 0; j < _layers[i].Size(); j++)
            {
                Connect(0, _inputLayer.Size() - 1, i, j);
            }
        }
    }

    protected void Connect(int sourceLayer, int sourceNeuron, int targetLayer, int targetNeuron)
    {
        new Synapse(_layers[sourceLayer].Get(sourceNeuron), _layers[targetLayer].Get(targetNeuron));
    }

    protected void AddInput(DataPoint p)
    {
        for (int k = 0; k < _inputLayer.Size() - 1; k++)
        {
            _inputLayer.Get(k).AddOutput(p.GetFeatureValue(_features[k]));
        }
        _inputLayer.Get(_inputLayer.Size() - 1).AddOutput(1.0);
    }

    protected void Propagate(int i)
    {
        for (int k = 1; k < _layers.Count; k++)
        {
            _layers[k].ComputeOutput(i);
        }
    }

    protected virtual int[][] BatchFeedForward(RankList rl)
    {
        var pairMap = new int[rl.Size()][];
        for (int i = 0; i < rl.Size(); i++)
        {
            AddInput(rl.Get(i));
            Propagate(i);

            var count = 0;
            for (int j = 0; j < rl.Size(); j++)
            {
                if (rl.Get(i).GetLabel() > rl.Get(j).GetLabel())
                {
                    count++;
                }
            }

            pairMap[i] = new int[count];
            var k = 0;
            for (int j = 0; j < rl.Size(); j++)
            {
                if (rl.Get(i).GetLabel() > rl.Get(j).GetLabel())
                {
                    pairMap[i][k++] = j;
                }
            }
        }
        return pairMap;
    }

    protected virtual void BatchBackPropagate(int[][] pairMap, float[][] pairWeight)
    {
        for (int i = 0; i < pairMap.Length; i++)
        {
            var p = new PropParameter(i, pairMap);
            _outputLayer.ComputeDelta(p);
            for (int j = _layers.Count - 2; j >= 1; j--)
            {
                _layers[j].UpdateDelta(p);
            }

            _outputLayer.UpdateWeight(p);
            for (int j = _layers.Count - 2; j >= 1; j--)
            {
                _layers[j].UpdateWeight(p);
            }
        }
    }

    protected void ClearNeuronOutputs()
    {
        foreach (var layer in _layers)
        {
            layer.ClearOutputs();
        }
    }
    
    // ----
    
    protected virtual float[][]? ComputePairWeight(int[][] pairMap, RankList rl) 
    {
        return null;
    }

    protected virtual RankList InternalReorder(RankList rl) {
        return rl;
    }

    /**
     * Model validation
     */
    protected void SaveBestModelOnValidation() {
        for (int i = 0; i < _layers.Count - 1; i++)//loop through all layers
        {
            List<double> l = _bestModelOnValidation[i];
            l.Clear();
            for (int j = 0; j < _layers[i].Size(); j++)//loop through all neurons on in the current layer
            {
                Neuron n = _layers[i].Get(j);
                for (int k = 0; k < n.GetOutLinks().Count; k++) {
                    l.Add(n.GetOutLinks()[k].Weight);
                }
            }
        }
    }

    protected void RestoreBestModelOnValidation() {
        try {
            for (int i = 0; i < _layers.Count - 1; i++)//loop through all layers
            {
                List<Double> l = _bestModelOnValidation[i];
                int c = 0;
                for (int j = 0; j < _layers[i].Size(); j++)//loop through all neurons on in the current layer
                {
                    Neuron n = _layers[i].Get(j);
                    for (int k = 0; k < n.GetOutLinks().Count; k++) {
                        n.GetOutLinks()[k].Weight = l[c++];
                    }
                }
            }
        } catch (Exception ex) 
        {
            throw RankLibError.Create("Error in NeuralNetwork.restoreBestModelOnValidation(): ", ex);
        }
    }
    
    protected double CrossEntropy(double o1, double o2, double targetValue)
    {
        var oij = o1 - o2;
        return -targetValue * oij + SimpleMath.LogBase2(1 + Math.Exp(oij));
    }
    
    protected virtual void EstimateLoss()
    {
        _misorderedPairs = 0;
        _error = 0.0;

        foreach (var rl in _samples)
        {
            for (int k = 0; k < rl.Size() - 1; k++)
            {
                double o1 = Eval(rl.Get(k));
                for (int l = k + 1; l < rl.Size(); l++)
                {
                    if (rl.Get(k).GetLabel() > rl.Get(l).GetLabel())
                    {
                        double o2 = Eval(rl.Get(l));
                        _error += CrossEntropy(o1, o2, 1.0);
                        if (o1 < o2) _misorderedPairs++;
                    }
                }
            }
        }
        _error = Math.Round(_error / _totalPairs, 4);
        _lastError = _error;
    }
    
    public override void Init()
    {
        _logger.LogInformation("Initializing...");

        SetInputOutput(_features.Length, 1);
        for (int i = 0; i < NHiddenLayer; i++)
        {
            AddHiddenLayer(NHiddenNodePerLayer);
        }
        Wire();

        _totalPairs = 0;
        foreach (var rl in _samples)
        {
            var correctRanking = rl.GetCorrectRanking();
            for (int j = 0; j < correctRanking.Size() - 1; j++)
            {
                for (int k = j + 1; k < correctRanking.Size(); k++)
                {
                    if (correctRanking.Get(j).GetLabel() > correctRanking.Get(k).GetLabel())
                    {
                        _totalPairs++;
                    }
                }
            }
        }

        Neuron.LearningRate = LearningRate;
    }

    public override void Learn()
    {
        _logger.LogInformation("Training starts...");
        PrintLogLn(new[] { 7, 14, 9, 9 },
            new[] { "#epoch", "% mis-ordered", _scorer.Name() + "-T", _scorer.Name() + "-V" });
        PrintLogLn(new[] { 7, 14, 9, 9 }, new[] { " ", "  pairs", " ", " " });

        for (int i = 1; i <= NIteration; i++)
        {
            for (int j = 0; j < _samples.Count; j++)
            {
                RankList rl = InternalReorder(_samples[j]);
                int[][] pairMap = BatchFeedForward(rl);
                float[][] pairWeight = ComputePairWeight(pairMap, rl)!;
                BatchBackPropagate(pairMap, pairWeight);
                ClearNeuronOutputs();
            }

            _scoreOnTrainingData = _scorer.Score(Rank(_samples));
            EstimateLoss();

            PrintLog(new[] { 7, 14 },
                new[] { i.ToString(), SimpleMath.Round((double)_misorderedPairs / _totalPairs, 4).ToString() });

            if (i % 1 == 0)
            {
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

        // Restore the best model if validation data was specified
        if (_validationSamples != null)
        {
            RestoreBestModelOnValidation();
        }

        _scoreOnTrainingData = SimpleMath.Round(_scorer.Score(Rank(_samples)), 4);
        _logger.LogInformation("Finished successfully.");
        _logger.LogInformation($"{_scorer.Name()} on training data: {_scoreOnTrainingData}");

        if (_validationSamples != null)
        {
            _bestScoreOnValidationData = _scorer.Score(Rank(_validationSamples));
            _logger.LogInformation($"{_scorer.Name()} on validation data: {SimpleMath.Round(_bestScoreOnValidationData, 4)}");
        }
    }


    public override double Eval(DataPoint p)
    {
        for (int k = 0; k < _inputLayer.Size() - 1; k++)
        {
            _inputLayer.Get(k).SetOutput(p.GetFeatureValue(_features[k]));
        }
        _inputLayer.Get(_inputLayer.Size() - 1).SetOutput(1.0);

        for (int k = 1; k < _layers.Count; k++)
        {
            _layers[k].ComputeOutput();
        }

        return _outputLayer.Get(0).GetOutput();
    }

    public override Ranker CreateNew()
    {
        return new RankNet();
    }

    public override string ToString()
    {
        var output = new System.Text.StringBuilder();
        for (int i = 0; i < _layers.Count - 1; i++)
        {
            for (int j = 0; j < _layers[i].Size(); j++)
            {
                var neuron = _layers[i].Get(j);
                output.AppendLine($"{i} {j} {string.Join(" ", neuron.GetOutLinks().Select(link => link.Weight))}");
            }
        }
        return output.ToString();
    }

    public override string Model()
    {
        var output = new System.Text.StringBuilder();
        output.AppendLine($"## {Name()}");
        output.AppendLine($"## Epochs = {NIteration}");
        output.AppendLine($"## No. of features = {_features.Length}");
        output.AppendLine($"## No. of hidden layers = {_layers.Count - 2}");
        for (int i = 1; i < _layers.Count - 1; i++)
        {
            output.AppendLine($"## Layer {i}: {_layers[i].Size()} neurons");
        }

        for (int i = 0; i < _features.Length; i++)
        {
            output.Append($"{_features[i]}{(i == _features.Length - 1 ? "" : " ")}");
        }
        output.AppendLine();
        output.AppendLine($"{_layers.Count - 2}");
        for (int i = 1; i < _layers.Count - 1; i++)
        {
            output.AppendLine($"{_layers[i].Size()}");
        }
        output.AppendLine(ToString());

        return output.ToString();
    }

    public override void LoadFromString(string fullText)
    {
        var lines = fullText.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

        var features = lines[0].Split(' ');
        _features = new int[features.Length];
        for (int i = 0; i < features.Length; i++)
        {
            _features[i] = int.Parse(features[i]);
        }

        var nhl = int.Parse(lines[1]);
        var nn = new int[nhl];
        int lineIndex = 2;
        for (; lineIndex < 2 + nhl; lineIndex++)
        {
            nn[lineIndex - 2] = int.Parse(lines[lineIndex]);
        }

        SetInputOutput(_features.Length, 1);
        for (int j = 0; j < nhl; j++)
        {
            AddHiddenLayer(nn[j]);
        }
        Wire();

        for (; lineIndex < lines.Length; lineIndex++)
        {
            var s = lines[lineIndex].Split(' ');
            var iLayer = int.Parse(s[0]);
            var iNeuron = int.Parse(s[1]);
            var neuron = _layers[iLayer].Get(iNeuron);
            for (int k = 0; k < neuron.GetOutLinks().Count; k++)
            {
                neuron.GetOutLinks()[k].Weight = double.Parse(s[k + 2]);
            }
        }
    }

    public override void PrintParameters()
    {
        _logger.LogInformation($"No. of epochs: {NIteration}");
        _logger.LogInformation($"No. of hidden layers: {NHiddenLayer}");
        _logger.LogInformation($"No. of hidden nodes per layer: {NHiddenNodePerLayer}");
        _logger.LogInformation($"Learning rate: {LearningRate}");
    }

    public override string Name() => "RankNet";
}