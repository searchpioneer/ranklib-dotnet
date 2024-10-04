using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Utilities;
using KeyValuePair = RankLib.Utilities.KeyValuePair;

namespace RankLib.Learning;

public class LinearRegRank : Ranker
{
	private static readonly ILogger<LinearRegRank> logger = NullLogger<LinearRegRank>.Instance;
	public static double lambda = 1E-10; // L2-norm regularization parameter

	// Local variables
	protected double[] weight = null;

	public LinearRegRank() { }

	public LinearRegRank(List<RankList> samples, int[] features, MetricScorer scorer)
		: base(samples, features, scorer) { }

	public override void Init() => logger.LogInformation("Initializing...");

	public override void Learn()
	{
		logger.LogInformation("Training starts...");
		logger.LogInformation("Learning the least square model...");

		// closed form solution: beta = ((xTx - lambda*I)^(-1)) * (xTy)
		var nVar = 0;
		foreach (var rl in _samples)
		{
			var c = rl.GetFeatureCount();
			if (c > nVar)
				nVar = c;
		}

		var xTx = new double[nVar][];
		for (var i = 0; i < nVar; i++)
		{
			xTx[i] = new double[nVar];
			Array.Fill(xTx[i], 0.0);
		}

		var xTy = new double[nVar];
		Array.Fill(xTy, 0.0);

		for (var s = 0; s < _samples.Count; s++)
		{
			var rl = _samples[s];
			for (var i = 0; i < rl.Count; i++)
			{
				var point = rl[i];
				xTy[nVar - 1] += point.Label;
				for (var j = 0; j < nVar - 1; j++)
				{
					xTy[j] += point.GetFeatureValue(j + 1) * point.Label;
					for (var k = 0; k < nVar; k++)
					{
						double t = (k < nVar - 1) ? point.GetFeatureValue(k + 1) : 1f;
						xTx[j][k] += point.GetFeatureValue(j + 1) * t;
					}
				}

				for (var k = 0; k < nVar - 1; k++)
					xTx[nVar - 1][k] += point.GetFeatureValue(k + 1);

				xTx[nVar - 1][nVar - 1] += 1f;
			}
		}

		if (lambda != 0.0)
		{
			for (var i = 0; i < xTx.Length; i++)
				xTx[i][i] += lambda;
		}

		weight = Solve(xTx, xTy);

		_scoreOnTrainingData = SimpleMath.Round(_scorer.Score(Rank(_samples)), 4);
		logger.LogInformation("Finished successfully.");
		logger.LogInformation($"{_scorer.Name()} on training data: {_scoreOnTrainingData}");

		if (_validationSamples != null)
		{
			_bestScoreOnValidationData = _scorer.Score(Rank(_validationSamples));
			logger.LogInformation($"{_scorer.Name()} on validation data: {SimpleMath.Round(_bestScoreOnValidationData, 4)}");
		}
	}

	public override double Eval(DataPoint p)
	{
		var score = weight[weight.Length - 1];
		for (var i = 0; i < _features.Length; i++)
			score += weight[i] * p.GetFeatureValue(_features[i]);
		return score;
	}

	public override Ranker CreateNew() => new LinearRegRank();

	public override string ToString()
	{
		var output = new StringBuilder();
		output.Append("0:" + weight[0] + " ");
		for (var i = 0; i < _features.Length; i++)
		{
			output.Append(_features[i] + ":" + weight[i]);
			if (i != weight.Length - 1)
				output.Append(" ");
		}
		return output.ToString();
	}

	public override string Model()
	{
		var output = new StringBuilder();
		output.Append("## " + Name() + "\n");
		output.Append("## Lambda = " + lambda + "\n");
		output.Append(ToString());
		return output.ToString();
	}

	public override void LoadFromString(string fullText)
	{
		try
		{
			using (var reader = new StringReader(fullText))
			{
				string content;
				KeyValuePair kvp = null;
				while ((content = reader.ReadLine()) != null)
				{
					content = content.Trim();
					if (content.Length == 0 || content.StartsWith("##"))
						continue;

					kvp = new KeyValuePair(content);
					break;
				}

				if (kvp == null)
					return;

				var keys = kvp.Keys();
				var values = kvp.Values();

				weight = new double[keys.Count];
				_features = new int[keys.Count - 1];

				var idx = 0;
				for (var i = 0; i < keys.Count; i++)
				{
					var fid = int.Parse(keys[i]);
					if (fid > 0)
					{
						_features[idx] = fid;
						weight[idx] = double.Parse(values[i]);
						idx++;
					}
					else
					{
						weight[weight.Length - 1] = double.Parse(values[i]);
					}
				}
			}
		}
		catch (Exception ex)
		{
			throw RankLibError.Create("Error in LinearRegRank::load(): ", ex);
		}
	}

	public override void PrintParameters() => logger.LogInformation("L2-norm regularization: lambda = " + lambda);

	public override string Name() => "Linear Regression";

	protected double[] Solve(double[][] A, double[] B)
	{
		if (A.Length == 0 || B.Length == 0)
			throw RankLibError.Create("Error: some of the input arrays is empty.");
		if (A[0].Length == 0)
			throw RankLibError.Create("Error: some of the input arrays is empty.");
		if (A.Length != B.Length)
			throw RankLibError.Create("Error: Solving Ax=B: A and B have different dimensions.");

		var a = new double[A.Length][];
		var b = new double[B.Length];
		Array.Copy(B, b, B.Length);
		for (var i = 0; i < a.Length; i++)
		{
			a[i] = new double[A[i].Length];
			if (i > 0 && a[i].Length != a[i - 1].Length)
				throw RankLibError.Create("Error: Solving Ax=B: A is NOT a square matrix.");
			Array.Copy(A[i], a[i], A[i].Length);
		}

		for (var j = 0; j < b.Length - 1; j++)
		{
			var pivot = a[j][j];
			for (var i = j + 1; i < b.Length; i++)
			{
				var multiplier = a[i][j] / pivot;
				for (var k = j + 1; k < b.Length; k++)
					a[i][k] -= a[j][k] * multiplier;
				b[i] -= b[j] * multiplier;
			}
		}

		var x = new double[b.Length];
		var n = b.Length;
		x[n - 1] = b[n - 1] / a[n - 1][n - 1];
		for (var i = n - 2; i >= 0; i--)
		{
			var val = b[i];
			for (var j = i + 1; j < n; j++)
				val -= a[i][j] * x[j];
			x[i] = val / a[i][i];
		}

		return x;
	}
}
