using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using RankLib.Metric;
using RankLib.Utilities;

namespace RankLib.Learning;

/// <summary>
/// Parameters for <see cref="LinearRegression"/>.
/// </summary>
public class LinearRegressionParameters : IRankerParameters
{
	/// <summary>
	/// Default L2-norm regularization parameter
	/// </summary>
	public const double DefaultLambda = 1E-10;

	/// <summary>
	/// L2-norm regularization parameter
	/// </summary>
	public double Lambda { get; set; } = DefaultLambda;

	public override string ToString() => $"L2-norm regularization: lambda = {Lambda}";
}

/// <summary>
/// Linear Regression ranking model that applies linear regression
/// to predict relevance scores for items, using feature weights learned from
/// training data to produce a ranked list of items based on predicted relevance.
/// </summary>
public class LinearRegression : Ranker<LinearRegressionParameters>
{
	internal const string RankerName = "Linear Regression";

	private readonly ILogger<LinearRegression> _logger;
	private double[] _weight = [];

	public LinearRegression(ILogger<LinearRegression>? logger = null) =>
		_logger = logger ?? NullLogger<LinearRegression>.Instance;

	public LinearRegression(List<RankList> samples, int[] features, MetricScorer scorer,
		ILogger<LinearRegression>? logger = null)
		: base(samples, features, scorer) =>
		_logger = logger ?? NullLogger<LinearRegression>.Instance;

	/// <inheritdoc />
	public override string Name => RankerName;

	/// <inheritdoc />
	public override Task InitAsync(CancellationToken cancellationToken = default)
	{
		_logger.LogInformation("Initializing...");
		return Task.CompletedTask;
	}

	/// <inheritdoc />
	public override Task LearnAsync(CancellationToken cancellationToken = default)
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
			CheckCancellation(_logger, cancellationToken);

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

		if (Parameters.Lambda != 0)
		{
			for (var i = 0; i < xTx.Length; i++)
				xTx[i][i] += Parameters.Lambda;
		}

		CheckCancellation(_logger, cancellationToken);
		_weight = Solve(xTx, xTy);

		TrainingDataScore = SimpleMath.Round(Scorer.Score(Rank(Samples)), 4);
		_logger.LogInformation("Finished successfully.");
		_logger.LogInformation("{Scorer} on training data: {TrainingScore}", Scorer.Name, TrainingDataScore);

		if (ValidationSamples != null)
		{
			ValidationDataScore = Scorer.Score(Rank(ValidationSamples));
			_logger.LogInformation("{Scorer} on validation data: {ValidationScore}", Scorer.Name, SimpleMath.Round(ValidationDataScore, 4));
		}

		return Task.CompletedTask;
	}

	/// <inheritdoc />
	public override double Eval(DataPoint dataPoint)
	{
		var score = _weight[^1];
		for (var i = 0; i < Features.Length; i++)
			score += _weight[i] * dataPoint.GetFeatureValue(Features[i]);

		return score;
	}

	/// <inheritdoc />
	public override string GetModel()
	{
		var output = new StringBuilder()
			.AppendLine($"## {Name}")
			.AppendLine($"## Lambda = {Parameters.Lambda}");

		output.Append($"0:{_weight[0]} ");
		for (var i = 0; i < Features.Length; i++)
		{
			output.Append(Features[i] + ":" + _weight[i]);
			if (i != _weight.Length - 1)
				output.Append(' ');
		}

		return output.ToString();
	}

	/// <inheritdoc />
	public override void LoadFromString(string model)
	{
		try
		{
			using var reader = new StringReader(model);
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

			_weight = new double[kvp.Count];
			Features = new int[kvp.Count - 1];

			var idx = 0;
			for (var i = 0; i < kvp.Count; i++)
			{
				var kv = kvp[i];

				var fid = int.Parse(kv.Key);
				if (fid > 0)
				{
					Features[idx] = fid;
					_weight[idx] = double.Parse(kv.Value);
					idx++;
				}
				else
					_weight[^1] = double.Parse(kv.Value);
			}
		}
		catch (Exception ex)
		{
			throw RankLibException.Create("Error in LinearRegRank::load(): ", ex);
		}
	}

	private static double[] Solve(double[][] a, double[] b)
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
