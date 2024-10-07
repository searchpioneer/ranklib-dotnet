using System.Collections;
using System.Runtime.Intrinsics.Arm;
using RankLib.Utilities;

namespace RankLib.Learning;

public class RankList
{
	// Protected members with prefixed underscores
	protected readonly DataPoint[] _rl;
	protected int _featureCount = 0;

	// Constructor that initializes the rank list from a list of DataPoint
	public RankList(List<DataPoint> rl)
	{
		_rl = new DataPoint[rl.Count];
		for (var i = 0; i < rl.Count; i++)
		{
			_rl[i] = rl[i];
		}
		Init();
	}

	// Copy constructor that creates a new RankList from an existing one
	public RankList(RankList rl)
	{
		_rl = new DataPoint[rl.Count];
		for (var i = 0; i < rl.Count; i++)
		{
			_rl[i] = rl[i];
		}
		Init();
	}

	// Constructor that creates a RankList from selected indices
	public RankList(RankList rl, int[] idx)
	{
		_rl = new DataPoint[rl.Count];
		for (var i = 0; i < idx.Length; i++)
		{
			var k = idx[i];
			_rl[i] = rl[k];
		}
		Init();
	}

	// Constructor that creates a RankList with offset
	public RankList(RankList rl, int[] idx, int offset)
	{
		_rl = new DataPoint[rl.Count];
		for (var i = 0; i < idx.Length; i++)
		{
			var k = idx[i] - offset;
			_rl[i] = rl[k];
		}
		Init();
	}

	// Initialize feature count
	protected void Init()
	{
		foreach (var dp in _rl)
		{
			var count = dp.GetFeatureCount();
			if (count > _featureCount)
			{
				_featureCount = count;
			}
		}
	}

	// Get the ID of the first DataPoint in the list
	public string Id => this[0].Id;

	// Get the size of the rank list
	public int Count => _rl.Length;

	public DataPoint this[int index]
	{
		get => _rl[index];
		set => _rl[index] = value;
	}

	// Get the correct ranking by label
	public RankList GetCorrectRanking()
	{
		var score = new double[_rl.Length];
		for (var i = 0; i < _rl.Length; i++)
		{
			score[i] = _rl[i].Label;
		}

		var idx = Sorter.Sort(score, false);
		return new RankList(this, idx);
	}

	// Get the ranking based on a specific feature ID
	public RankList GetRanking(short fid)
	{
		var score = new double[_rl.Length];
		for (var i = 0; i < _rl.Length; i++)
		{
			score[i] = _rl[i].GetFeatureValue(fid);
		}

		var idx = Sorter.Sort(score, false);
		return new RankList(this, idx);
	}

	// Get the feature count
	public int GetFeatureCount() => _featureCount;

	// Override the ToString method
	public override string ToString()
	{
		var size = _rl?.Length ?? 0;
		return $"RankList ({size}, {_featureCount})";
	}
}
