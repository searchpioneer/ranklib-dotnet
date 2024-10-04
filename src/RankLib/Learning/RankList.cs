using System.Collections;
using RankLib.Utilities;

namespace RankLib.Learning;

public class RankList
{
    // Protected members with prefixed underscores
    protected DataPoint[]? _rl = null;
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
        _rl = new DataPoint[rl.Size()];
        for (var i = 0; i < rl.Size(); i++)
        {
            _rl[i] = rl.Get(i);
        }
        Init();
    }

    // Constructor that creates a RankList from selected indices
    public RankList(RankList rl, int[] idx)
    {
        _rl = new DataPoint[rl.Size()];
        for (var i = 0; i < idx.Length; i++)
        {
            _rl[i] = rl.Get(idx[i]);
        }
        Init();
    }

    // Constructor that creates a RankList with offset
    public RankList(RankList rl, int[] idx, int offset)
    {
        _rl = new DataPoint[rl.Size()];
        for (var i = 0; i < idx.Length; i++)
        {
            _rl[i] = rl.Get(idx[i] - offset);
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
    public string GetID()
    {
        return Get(0).GetID();
    }

    // Get the size of the rank list
    public int Size()
    {
        return _rl.Length;
    }

    // Get the DataPoint at a specific index
    public DataPoint Get(int k)
    {
        return _rl[k];
    }

    // Set the DataPoint at a specific index
    public void Set(int k, DataPoint p)
    {
        _rl[k] = p;
    }

    // Get the correct ranking by label
    public RankList GetCorrectRanking()
    {
        var score = new double[_rl.Length];
        for (var i = 0; i < _rl.Length; i++)
        {
            score[i] = _rl[i].GetLabel();
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
    public int GetFeatureCount()
    {
        return _featureCount;
    }

    // Override the ToString method
    public override string ToString()
    {
        var size = _rl?.Length ?? 0;
        return $"RankList ({size}, {_featureCount})";
    }
}