using System.Collections;
using System.Runtime.Intrinsics.Arm;
using RankLib.Utilities;

namespace RankLib.Learning;

/// <summary>
/// A list of <see cref="DataPoint"/> to be ranked.
/// </summary>
public class RankList : IEnumerable<DataPoint>
{
	private readonly DataPoint[] _dataPoints;

	/// <summary>
	/// Initializes a new instance of <see cref="RankList"/> with the specified data points.
	/// </summary>
	/// <param name="dataPoints">The list of data points.</param>
	public RankList(List<DataPoint> dataPoints)
	{
		_dataPoints = dataPoints.ToArray();
		Init();
	}

	/// <summary>
	/// Initializes a new instance of <see cref="RankList"/> with a copy of the data points from the specified rank list.
	/// </summary>
	/// <param name="rankList">The rank list to copy data points from.</param>
	public RankList(RankList rankList)
	{
		_dataPoints = new DataPoint[rankList.Count];
		Array.Copy(rankList._dataPoints, _dataPoints, rankList.Count);
		Init();
	}

	/// <summary>
	/// Initializes a new instance of <see cref="RankList"/> with a copy of
	/// selected data points from the specified rank list specified by <paramref name="idx"/>
	/// </summary>
	/// <param name="rankList">The rank list to copy data points from.</param>
	/// <param name="idx">The indexes of data points to copy.</param>
	public RankList(RankList rankList, int[] idx)
	{
		_dataPoints = new DataPoint[rankList.Count];
		for (var i = 0; i < idx.Length; i++)
			_dataPoints[i] = rankList[idx[i]];

		Init();
	}

	/// <summary>
	/// Initializes a new instance of <see cref="RankList"/> with a copy of
	/// selected data points from the specified rank list specified by <paramref name="idx"/>
	/// minus the specified offset.
	/// </summary>
	/// <param name="rankList">The rank list to copy data points from.</param>
	/// <param name="idx">The indexes of data points to copy.</param>
	/// <param name="offset">The offset to apply to indexes of data points to copy.</param>
	public RankList(RankList rankList, int[] idx, int offset)
	{
		_dataPoints = new DataPoint[rankList.Count];
		for (var i = 0; i < idx.Length; i++)
			_dataPoints[i] = rankList[idx[i] - offset];

		Init();
	}

	// Initialize feature count
	private void Init()
	{
		foreach (var dp in _dataPoints)
		{
			var count = dp.FeatureCount;
			if (count > FeatureCount)
				FeatureCount = count;
		}
	}

	/// <summary>
	/// Gets the ID of the first data point in the rank list.
	/// </summary>
	public string Id => this[0].Id;

	/// <summary>
	/// Gets the count of data points in the rank list
	/// </summary>
	public int Count => _dataPoints.Length;

	/// <summary>
	/// Gets the feature count
	/// </summary>
	public int FeatureCount { get; private set; }

	/// <summary>
	/// Indexer into the data points
	/// </summary>
	/// <param name="index">The index of the data point to get.</param>
	public DataPoint this[int index] => _dataPoints[index];

	/// <summary>
	/// Get the correct ranking by label
	/// </summary>
	public RankList GetCorrectRanking()
	{
		var score = new double[_dataPoints.Length];
		for (var i = 0; i < _dataPoints.Length; i++)
			score[i] = _dataPoints[i].Label;

		var idx = Sorter.Sort(score, false);
		return new RankList(this, idx);
	}

	/// <summary>
	/// Gets the ranking based on a specific feature ID
	/// </summary>
	/// <param name="fid">The feature ID</param>
	/// <returns>A new instance of <see cref="RankList"/> with ranking based on the feature ID.</returns>
	public RankList GetRanking(int fid)
	{
		var score = new double[_dataPoints.Length];
		for (var i = 0; i < _dataPoints.Length; i++)
			score[i] = _dataPoints[i].GetFeatureValue(fid);

		var idx = Sorter.Sort(score, false);
		return new RankList(this, idx);
	}

	/// <inheritdoc />
	public override string ToString() => $"RankList ({Count}, {FeatureCount})";

	/// <inheritdoc />
	public IEnumerator<DataPoint> GetEnumerator() => ((IEnumerable<DataPoint>)_dataPoints).GetEnumerator();

	/// <inheritdoc />
	IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
