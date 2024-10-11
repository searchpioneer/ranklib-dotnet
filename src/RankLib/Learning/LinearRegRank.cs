using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning;

public class LinearRegRank : Ranker
{
	private readonly ILogger<LinearRegRank> _logger;
	public static double lambda = 1E-10; // L2-norm regularization parameter

	// Local variables
	protected double[] weight = null;

	public LinearRegRank(ILogger<LinearRegRank>? logger = null) : base(logger) =>
		_logger = logger ?? NullLogger<LinearRegRank>.Instance;

	public LinearRegRank(List<RankList> samples, int[] features, MetricScorer scorer,
		ILogger<LinearRegRank>? logger = null)
		: base(samples, features, scorer, logger) =>
		_logger = logger ?? NullLogger<LinearRegRank>.Instance;

	public override void Init() => _logger.LogInformation("Initializing...");

	public override void Learn()
	{
		_logger.LogInformation("Training starts...");
		_logger.LogInformation("Learning the least square model...");

		// closed form solution: beta = ((xTx - lambda*I)^(-1)) * (xTy)
		var nVar = 0;
		foreach (var rl in Samples)
		{
			var c = rl.FeatureCount;
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

		for (var s = 0; s < Samples.Count; s++)
		{
			var rl = Samples[s];
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

		ScoreOnTrainingData = SimpleMath.Round(Scorer.Score(Rank(Samples)), 4);
		_logger.LogInformation("Finished successfully.");
		_logger.LogInformation("{ScorerName} on training data: {ScoreOnTrainingData}", Scorer.Name, ScoreOnTrainingData);

		if (ValidationSamples != null)
		{
			BestScoreOnValidationData = Scorer.Score(Rank(ValidationSamples));
			_logger.LogInformation("{ScorerName} on validation data: {BestScoreOnValidationData}", Scorer.Name, SimpleMath.Round(BestScoreOnValidationData, 4));
		}
	}

	public override double Eval(DataPoint p)
	{
		var score = weight[^1];
		for (var i = 0; i < Features.Length; i++)
			score += weight[i] * p.GetFeatureValue(Features[i]);

		return score;
	}

	public virtual Ranker CreateNew() => new LinearRegRank();

	public override string ToString()
	{
		var output = new StringBuilder();
		output.Append($"0:{weight[0]} ");
		for (var i = 0; i < Features.Length; i++)
		{
			output.Append(Features[i] + ":" + weight[i]);
			if (i != weight.Length - 1)
				output.Append(' ');
		}
		return output.ToString();
	}

	public override string Model
	{
		get
		{
			var output = new StringBuilder()
				.AppendLine($"## {Name}")
				.AppendLine($"## Lambda = {lambda}")
				.Append(ToString());
			return output.ToString();
		}
	}

	public override void LoadFromString(string fullText)
	{
		try
		{
			using var reader = new StringReader(fullText);
			KeyValuePairs? kvp = null;
			while (reader.ReadLine() is { } content)
			{
				content = content.Trim();
				if (content.Length == 0 || content.StartsWith("##"))
					continue;

				kvp = new KeyValuePairs(content);
				break;
			}

			if (kvp == null)
				return;

			weight = new double[kvp.Count];
			Features = new int[kvp.Count - 1];

			var idx = 0;
			for (var i = 0; i < kvp.Count; i++)
			{
				var kv = kvp[i];

				var fid = int.Parse(kv.Key);
				if (fid > 0)
				{
					Features[idx] = fid;
					weight[idx] = double.Parse(kv.Value);
					idx++;
				}
				else
				{
					weight[^1] = double.Parse(kv.Value);
				}
			}
		}
		catch (Exception ex)
		{
			throw RankLibException.Create("Error in LinearRegRank::load(): ", ex);
		}
	}

	public override void PrintParameters() => _logger.LogInformation("L2-norm regularization: lambda = {Lambda}", lambda);

	public override string Name => "Linear Regression";

	protected double[] Solve(double[][] a, double[] b)
	{
		if (a.Length == 0 || b.Length == 0)
			throw RankLibException.Create("Error: some of the input arrays is empty.");
		if (a[0].Length == 0)
			throw RankLibException.Create("Error: some of the input arrays is empty.");
		if (a.Length != b.Length)
			throw RankLibException.Create("Error: Solving Ax=B: A and B have different dimensions.");

		var aCopy = new double[a.Length][];
		var bCopy = new double[b.Length];
		Array.Copy(b, bCopy, b.Length);
		for (var i = 0; i < aCopy.Length; i++)
		{
			aCopy[i] = new double[a[i].Length];
			if (i > 0 && aCopy[i].Length != aCopy[i - 1].Length)
				throw RankLibException.Create("Error: Solving Ax=B: A is NOT a square matrix.");

			Array.Copy(a[i], aCopy[i], a[i].Length);
		}

		for (var j = 0; j < bCopy.Length - 1; j++)
		{
			var pivot = aCopy[j][j];
			for (var i = j + 1; i < bCopy.Length; i++)
			{
				var multiplier = aCopy[i][j] / pivot;
				for (var k = j + 1; k < bCopy.Length; k++)
					aCopy[i][k] -= aCopy[j][k] * multiplier;
				bCopy[i] -= bCopy[j] * multiplier;
			}
		}

		var x = new double[bCopy.Length];
		var n = bCopy.Length;
		x[n - 1] = bCopy[n - 1] / aCopy[n - 1][n - 1];
		for (var i = n - 2; i >= 0; i--)
		{
			var val = bCopy[i];
			for (var j = i + 1; j < n; j++)
				val -= aCopy[i][j] * x[j];
			x[i] = val / aCopy[i][i];
		}

		return x;
	}
}
