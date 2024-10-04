namespace RankLib.Learning.NeuralNet;

public interface ITransferFunction
{
    double Compute(double x);
    
    double ComputeDerivative(double x);
}