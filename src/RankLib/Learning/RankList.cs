using System.Collections;
using System.Runtime.Intrinsics.Arm;
using RankLib.Utilities;

namespace RankLib.Learning;

public class RankList : IEnumerable<DataPoint>
{
	private readonly DataPoint[] _dataPoints;

	// Constructor that initializes the rank list from a list of DataPoint
	public RankList(List<DataPoint> dataPoints)
	{
		_dataPoints = dataPoints.ToArray();
		Init();
	}

	// Copy constructor that creates a new RankList from an existing one
	public RankList(RankList rankList)
	{
		_dataPoints = new DataPoint[rankList.Count];
		Array.Copy(rankList._dataPoints, _dataPoints, rankList.Count);
		Init();
	}

	// Constructor that creates a RankList from selected indices
	public RankList(RankList rankList, int[] idx)
	{
		_dataPoints = new DataPoint[rankList.Count];
		for (var i = 0; i < idx.Length; i++)
			_dataPoints[i] = rankList[idx[i]];

		Init();
	}

	// Constructor that creates a RankList with offset
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

	// Get the ID of the first DataPoint in the list
	public string Id => this[0].Id;

	// Get the size of the rank list
	public int Count => _dataPoints.Length;

	// Gets the feature count
	public int FeatureCount { get; private set; }

	public DataPoint this[int index]
	{
		get => _dataPoints[index];
		set => _dataPoints[index] = value;
	}

	// Get the correct ranking by label
	public RankList GetCorrectRanking()
	{
		var score = new double[_dataPoints.Length];
		for (var i = 0; i < _dataPoints.Length; i++)
			score[i] = _dataPoints[i].Label;

		var idx = Sorter.Sort(score, false);
		return new RankList(this, idx);
	}

	// Get the ranking based on a specific feature ID
	public RankList GetRanking(int fid)
	{
		var score = new double[_dataPoints.Length];
		for (var i = 0; i < _dataPoints.Length; i++)
			score[i] = _dataPoints[i].GetFeatureValue(fid);

		var idx = Sorter.Sort(score, false);
		return new RankList(this, idx);
	}

	// Override the ToString method
	public override string ToString() => $"RankList ({Count}, {FeatureCount})";

	public IEnumerator<DataPoint> GetEnumerator() => ((IEnumerable<DataPoint>)_dataPoints).GetEnumerator();

	IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
