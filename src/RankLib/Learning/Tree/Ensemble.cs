using RankLib.Utilities;

namespace RankLib.Learning.Tree;

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Xml;

public class Ensemble
{
	private List<RegressionTree> trees = [];
	private List<float> weights = [];
	private int[] features = null;

	public Ensemble() { }

	public Ensemble(Ensemble e)
	{
		trees.AddRange(e.trees);
		weights.AddRange(e.weights);
	}

	public Ensemble(string xml)
	{
		try
		{
			using var inStream = new MemoryStream(Encoding.UTF8.GetBytes(xml));
			var doc = new XmlDocument();
			doc.Load(inStream);
			var treeNodes = doc.GetElementsByTagName("tree");
			var fids = new Dictionary<int, int>();
			foreach (XmlNode n in treeNodes)
			{
				// Create a regression tree from this node
				var root = Create(n.FirstChild, fids);
				// Get the weight for this tree
				var weight = float.Parse(n.Attributes["weight"].Value);
				// Add it to the ensemble
				trees.Add(new RegressionTree(root));
				weights.Add(weight);
			}

			features = new int[fids.Keys.Count];
			var i = 0;
			foreach (var fid in fids.Keys)
			{
				features[i++] = fid;
			}
		}
		catch (Exception ex)
		{
			throw RankLibException.Create("Error reading ensemble from xml", ex);
		}
	}

	public void Add(RegressionTree tree, float weight)
	{
		trees.Add(tree);
		weights.Add(weight);
	}

	public RegressionTree GetTree(int k) => trees[k];

	public float GetWeight(int k) => weights[k];

	public double Variance()
	{
		double var = 0;
		foreach (var tree in trees)
		{
			var += tree.Variance();
		}
		return var;
	}

	public void Remove(int k)
	{
		trees.RemoveAt(k);
		weights.RemoveAt(k);
	}

	public int TreeCount => trees.Count;

	public int[] Features => features;

	public int LeafCount
	{
		get
		{
			var count = 0;
			foreach (var tree in trees)
				count += tree.Leaves.Count;

			return count;
		}
	}

	public float Eval(DataPoint dp)
	{
		float s = 0;
		for (var i = 0; i < trees.Count; i++)
		{
			s += Convert.ToSingle(trees[i].Eval(dp) * weights[i]);
		}
		return s;
	}

	public override string ToString()
	{
		var builder = new StringBuilder();
		builder.Append("<ensemble>\n");
		for (var i = 0; i < trees.Count; i++)
		{
			builder.Append("\t<tree id=\"").Append(i + 1).Append("\" weight=\"").Append(weights[i].ToString()).Append("\">\n");
			builder.Append(trees[i].ToString("\t\t"));
			builder.Append("\t</tree>\n");
		}
		builder.Append("</ensemble>\n");
		return builder.ToString();
	}

	private Split Create(XmlNode n, Dictionary<int, int> fids)
	{
		Split s;
		if (n.FirstChild.Name.Equals("feature", StringComparison.OrdinalIgnoreCase)) // this is a split
		{
			var nl = n.ChildNodes;
			var fid = int.Parse(nl[0].FirstChild.Value.Trim()); // <feature>
			fids[fid] = 0;
			var threshold = float.Parse(nl[1].FirstChild.Value.Trim()); // <threshold>
			s = new Split(fid, threshold, 0);
			s.SetLeft(Create(nl[2], fids));
			s.SetRight(Create(nl[3], fids));
		}
		else // this is a stump
		{
			var output = float.Parse(n.FirstChild.FirstChild.Value.Trim());
			s = new Split();
			s.SetOutput(output);
		}
		return s;
	}
}
