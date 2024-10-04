namespace RankLib.Learning.NeuralNet;

public class ListNeuron : Neuron
{
    private double[] d1;
    private double[] d2;

    public override void ComputeDelta(PropParameter param)
    {
        double sumLabelExp = 0;
        double sumScoreExp = 0;

        // Calculate sums of exponentiated labels and scores
        for (int i = 0; i < _outputs.Count; i++)
        {
            sumLabelExp += Math.Exp(param.Labels[i]);
            sumScoreExp += Math.Exp(_outputs[i]);
        }

        d1 = new double[_outputs.Count];
        d2 = new double[_outputs.Count];

        // Calculate d1 and d2 based on the above sums
        for (int i = 0; i < _outputs.Count; i++)
        {
            d1[i] = Math.Exp(param.Labels[i]) / sumLabelExp;
            d2[i] = Math.Exp(_outputs[i]) / sumScoreExp;
        }
    }

    public override void UpdateWeight(PropParameter param)
    {
        Synapse s;
        for (int k = 0; k < _inLinks.Count; k++)
        {
            s = _inLinks[k];
            double dw = 0;

            // Update weights based on the difference between d1 and d2
            for (int l = 0; l < d1.Length; l++)
            {
                dw += (d1[l] - d2[l]) * s.Source.GetOutput(l);
            }

            dw *= LearningRate;
            s.SetWeightAdjustment(dw);
            s.UpdateWeight();
        }
    }
}
