namespace RankLib.Learning;

public abstract class DataPoint
{
    public static bool MissingZero = false;
    protected static readonly int FeatureIncrease = 10;
    protected int _featureCount = 0;
    
    protected static readonly float Unknown = float.NaN;

    // attributes
    protected float _label = 0.0f; // [ground truth] the real label of the data point (e.g., degree of relevance)
    protected string _id = ""; // id of this data point (e.g., query-id)
    protected string _description = "";
    protected float[] _fVals = null; // _fVals[0] is unused. Feature id MUST start from 1

    // helper attributes
    protected int _knownFeatures; // number of known feature values

    // internal to learning procedures
    protected double _cached = -1.0; // the latest evaluation score of the learned model on this data point

    protected static bool IsUnknown(float fVal)
    {
        return float.IsNaN(fVal);
    }

    protected static string GetKey(string pair)
    {
        return pair.Substring(0, pair.IndexOf(':'));
    }

    protected static string GetValue(string pair)
    {
        return pair.Substring(pair.LastIndexOf(':') + 1);
    }

    /// <summary>
    /// Parse the given line of text to construct a dense array of feature values and reset metadata.
    /// </summary>
    /// <param name="text">The text to parse</param>
    /// <returns>Dense array of feature values</returns>
    protected float[] Parse(string text)
    {
        int maxFeature = 51;
        float[] fval = new float[maxFeature];
        Array.Fill(fval, Unknown);
        int lastFeature = -1;

        try
        {
            int idx = text.IndexOf('#');
            if (idx != -1)
            {
                _description = text.Substring(idx);
                text = text.Substring(0, idx).Trim(); // remove the comment part at the end of the line
            }

            string[] fs = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            _label = float.Parse(fs[0]);

            if (_label < 0)
            {
                throw new InvalidOperationException("Relevance label cannot be negative. System will now exit.");
            }

            _id = GetValue(fs[1]);

            for (int i = 2; i < fs.Length; i++)
            {
                _knownFeatures++;
                string key = GetKey(fs[i]);
                string val = GetValue(fs[i]);
                int f = int.Parse(key);

                if (f <= 0)
                {
                    throw new InvalidOperationException("Cannot use feature numbering less than or equal to zero. Start your features at 1.");
                }

                if (f >= maxFeature)
                {
                    while (f >= maxFeature)
                    {
                        maxFeature += FeatureIncrease;
                    }

                    float[] tmp = new float[maxFeature];
                    Array.Copy(fval, tmp, fval.Length);
                    Array.Fill(tmp, Unknown, fval.Length, maxFeature - fval.Length);
                    fval = tmp;
                }

                fval[f] = float.Parse(val);

                if (f > _featureCount)
                {
                    _featureCount = f;
                }

                if (f > lastFeature)
                {
                    lastFeature = f;
                }
            }

            // shrink fVals
            float[] shrinkFVals = new float[lastFeature + 1];
            Array.Copy(fval, shrinkFVals, lastFeature + 1);
            fval = shrinkFVals;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException("Error in DataPoint::Parse", ex);
        }

        return fval;
    }

    // Abstract methods for feature value operations
    public abstract float GetFeatureValue(int fid);
    public abstract void SetFeatureValue(int fid, float fval);
    public abstract void SetFeatureVector(float[] dfVals);
    public abstract float[] GetFeatureVector();

    // Default constructor
    protected DataPoint() { }

    // Constructor to initialize DataPoint from text
    protected DataPoint(string text)
    {
        SetFeatureVector(Parse(text));
    }

    public string GetID()
    {
        return _id;
    }

    public void SetID(string id)
    {
        _id = id;
    }

    public float GetLabel()
    {
        return _label;
    }

    public void SetLabel(float label)
    {
        _label = label;
    }

    public string GetDescription()
    {
        return _description;
    }

    public void SetDescription(string description)
    {
        _description = description;
    }

    public void SetCached(double c)
    {
        _cached = c;
    }

    public double GetCached()
    {
        return _cached;
    }

    public void ResetCached()
    {
        _cached = -100000000.0f;
    }

    public int GetFeatureCount()
    {
        return _featureCount;
    }

    // Override ToString method
    public override string ToString()
    {
        float[] fval = GetFeatureVector();
        var output = new System.Text.StringBuilder();
        output.Append(((int)_label) + " qid:" + _id + " ");

        for (int i = 1; i < fval.Length; i++)
        {
            if (!IsUnknown(fval[i]))
            {
                output.Append(i + ":" + fval[i] + (i == fval.Length - 1 ? "" : " "));
            }
        }

        output.Append(" " + _description);
        return output.ToString();
    }
}