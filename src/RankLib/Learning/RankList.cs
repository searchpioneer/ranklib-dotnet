using System.Collections;
using System.Runtime.Intrinsics.Arm;
using RankLib.Utilities;

namespace RankLib.Learning;

public class RankList
{
	// Protected members with prefixed underscores
	protected readonly DataPoint[] DataPoints;

	// Constructor that initializes the rank list from a list of DataPoint
	public RankList(List<DataPoint> dataPoints)
	{
		DataPoints = dataPoints.ToArray();
		Init();
	}

	// Copy constructor that creates a new RankList from an existing one
	public RankList(RankList rankList)
	{
		DataPoints = new DataPoint[rankList.Count];
		Array.Copy(rankList.DataPoints, DataPoints, rankList.Count);
		Init();
	}

	// Constructor that creates a RankList from selected indices
	public RankList(RankList rankList, int[] idx)
	{
		DataPoints = new DataPoint[rankList.Count];
		for (var i = 0; i < idx.Length; i++)
		{
			var k = idx[i];
			DataPoints[i] = rankList[k];
		}
		Init();
	}

	// Constructor that creates a RankList with offset
	public RankList(RankList rankList, int[] idx, int offset)
	{
		DataPoints = new DataPoint[rankList.Count];
		for (var i = 0; i < idx.Length; i++)
		{
			var k = idx[i] - offset;
			DataPoints[i] = rankList[k];
		}
		Init();
	}

	// Initialize feature count
	protected void Init()
	{
		foreach (var dp in DataPoints)
		{
			var count = dp.FeatureCount;
			if (count > FeatureCount)
			{
				FeatureCount = count;
			}
		}
	}

	// Get the ID of the first DataPoint in the list
	public string Id => this[0].Id;

	// Get the size of the rank list
	public int Count => DataPoints.Length;

	// Gets the feature count
	public int FeatureCount { get; protected set; }

	public DataPoint this[int index]
	{
		get => DataPoints[index];
		set => DataPoints[index] = value;
	}

	// Get the correct ranking by label
	public RankList GetCorrectRanking()
	{
		var score = new double[DataPoints.Length];
		for (var i = 0; i < DataPoints.Length; i++)
		{
			score[i] = DataPoints[i].Label;
		}

		var idx = Sorter.Sort(score, false);
		return new RankList(this, idx);
	}

	// Get the ranking based on a specific feature ID
	public RankList GetRanking(short fid)
	{
		var score = new double[DataPoints.Length];
		for (var i = 0; i < DataPoints.Length; i++)
		{
			score[i] = DataPoints[i].GetFeatureValue(fid);
		}

		var idx = Sorter.Sort(score, false);
		return new RankList(this, idx);
	}

	// Override the ToString method
	public override string ToString() => $"RankList ({Count}, {FeatureCount})";
}
