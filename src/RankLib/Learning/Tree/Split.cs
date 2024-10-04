namespace RankLib.Learning.Tree;

public class Split
{
    // Key attributes of a split (tree node)
    private int featureID = -1;
    private float threshold = 0F;
    private double avgLabel = 0.0F;

    // Intermediate variables (ONLY used during learning)
    //*DO NOT* attempt to access them once the training is done
    private bool isRoot = false;
    private double sumLabel = 0.0;
    private double sqSumLabel = 0.0;
    private Split left = null;
    private Split right = null;
    private double deviance = 0F; // Mean squared error "S"
    private int[][] sortedSampleIDs = null;
    public int[] samples = null;
    public FeatureHistogram hist = null;

    public Split() { }

    public Split(int featureID, float threshold, double deviance)
    {
        this.featureID = featureID;
        this.threshold = threshold;
        this.deviance = deviance;
    }

    public Split(int[][] sortedSampleIDs, double deviance, double sumLabel, double sqSumLabel)
    {
        this.sortedSampleIDs = sortedSampleIDs;
        this.deviance = deviance;
        this.sumLabel = sumLabel;
        this.sqSumLabel = sqSumLabel;
        avgLabel = sumLabel / sortedSampleIDs[0].Length;
    }

    public Split(int[] samples, FeatureHistogram hist, double deviance, double sumLabel)
    {
        this.samples = samples;
        this.hist = hist;
        this.deviance = deviance;
        this.sumLabel = sumLabel;
        avgLabel = sumLabel / samples.Length;
    }

    public void Set(int featureID, float threshold, double deviance)
    {
        this.featureID = featureID;
        this.threshold = threshold;
        this.deviance = deviance;
    }

    public void SetLeft(Split s)
    {
        left = s;
    }

    public void SetRight(Split s)
    {
        right = s;
    }

    public void SetOutput(float output)
    {
        avgLabel = output;
    }

    public Split GetLeft()
    {
        return left;
    }

    public Split GetRight()
    {
        return right;
    }

    public double GetDeviance()
    {
        return deviance;
    }

    public double GetOutput()
    {
        return avgLabel;
    }

    public List<Split> Leaves()
    {
        List<Split> list = new List<Split>();
        Leaves(list);
        return list;
    }

    private void Leaves(List<Split> leaves)
    {
        if (featureID == -1)
        {
            leaves.Add(this);
        }
        else
        {
            left.Leaves(leaves);
            right.Leaves(leaves);
        }
    }

    public double Eval(DataPoint dp)
    {
        Split n = this;
        while (n.featureID != -1)
        {
            if (dp.GetFeatureValue(n.featureID) <= n.threshold)
            {
                n = n.left;
            }
            else
            {
                n = n.right;
            }
        }
        return n.avgLabel;
    }

    public override string ToString()
    {
        return ToString("");
    }

    public string ToString(string indent)
    {
        var buf = new System.Text.StringBuilder(100);
        buf.Append(indent).Append("<split>\n");
        buf.Append(GetString(indent + "\t"));
        buf.Append(indent).Append("</split>\n");
        return buf.ToString();
    }

    public string GetString(string indent)
    {
        var buf = new System.Text.StringBuilder(100);
        if (featureID == -1)
        {
            buf.Append(indent).Append("<output>").Append(avgLabel).Append(" </output>\n");
        }
        else
        {
            buf.Append(indent).Append("<feature>").Append(featureID).Append(" </feature>\n");
            buf.Append(indent).Append("<threshold> ").Append(threshold).Append(" </threshold>\n");
            buf.Append(indent).Append("<split pos=\"left\">\n");
            buf.Append(left.GetString(indent + "\t"));
            buf.Append(indent).Append("</split>\n");
            buf.Append(indent).Append("<split pos=\"right\">\n");
            buf.Append(right.GetString(indent + "\t"));
            buf.Append(indent).Append("</split>\n");
        }
        return buf.ToString();
    }

    // Internal functions (ONLY used during learning)
    //*DO NOT* attempt to call them once the training is done
    public bool split(double[] trainingLabels, int minLeafSupport)
    {
        return hist.FindBestSplit(this, trainingLabels, minLeafSupport);
    }

    public int[] GetSamples()
    {
        if (sortedSampleIDs != null)
        {
            return sortedSampleIDs[0];
        }
        return samples;
    }

    public int[][] GetSampleSortedIndex()
    {
        return sortedSampleIDs;
    }

    public double GetSumLabel()
    {
        return sumLabel;
    }

    public double GetSqSumLabel()
    {
        return sqSumLabel;
    }

    public void ClearSamples()
    {
        sortedSampleIDs = null;
        samples = null;
        hist = null;
    }

    public void SetRoot(bool isRoot)
    {
        this.isRoot = isRoot;
    }

    public bool IsRoot()
    {
        return isRoot;
    }
}
