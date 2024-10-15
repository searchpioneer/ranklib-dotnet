using System.Globalization;
using RankLib.Utilities;

namespace RankLib.Learning.Tree;

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Xml;

public class Ensemble
{
	private readonly List<RegressionTree> _trees = [];
	private readonly List<float> _weights = [];
	private readonly int[] _features = [];

	public Ensemble() { }

	public Ensemble(Ensemble e)
	{
		_trees.AddRange(e._trees);
		_weights.AddRange(e._weights);
	}

	public Ensemble(string xml)
	{
		try
		{
			using var stream = new MemoryStream(Encoding.UTF8.GetBytes(xml));
			var doc = new XmlDocument();
			doc.Load(stream);
			var treeNodes = doc.GetElementsByTagName("tree");
			var fids = new Dictionary<int, int>();
			foreach (XmlNode n in treeNodes)
			{
				// Create a regression tree from this node
				var root = Create(n.FirstChild, fids);
				// Get the weight for this tree
				var weight = float.Parse(n.Attributes["weight"].Value);
				// Add it to the ensemble
				_trees.Add(new RegressionTree(root));
				_weights.Add(weight);
			}

			_features = new int[fids.Keys.Count];
			var i = 0;
			foreach (var fid in fids.Keys)
			{
				_features[i++] = fid;
			}
		}
		catch (Exception ex)
		{
			throw RankLibException.Create("Error reading ensemble from xml", ex);
		}
	}

	public void Add(RegressionTree tree, float weight)
	{
		_trees.Add(tree);
		_weights.Add(weight);
	}

	public IReadOnlyList<RegressionTree> Trees => _trees;

	public IReadOnlyList<float> Weights => _weights;

	public double Variance()
	{
		double variance = 0;
		foreach (var tree in _trees)
		{
			variance += tree.Variance();
		}
		return variance;
	}

	public void Remove(int k)
	{
		_trees.RemoveAt(k);
		_weights.RemoveAt(k);
	}

	public int TreeCount => _trees.Count;

	public int[] Features => _features;

	public int LeafCount
	{
		get
		{
			var count = 0;
			foreach (var tree in _trees)
				count += tree.Leaves.Count;

			return count;
		}
	}

	public float Eval(DataPoint dp)
	{
		float s = 0;
		for (var i = 0; i < _trees.Count; i++)
		{
			s += Convert.ToSingle(_trees[i].Eval(dp) * _weights[i]);
		}
		return s;
	}

	public override string ToString()
	{
		var builder = new StringBuilder();
		builder.Append("<ensemble>\n");
		for (var i = 0; i < _trees.Count; i++)
		{
			builder.Append("\t<tree id=\"").Append(i + 1).Append("\" weight=\"").Append(_weights[i].ToString(CultureInfo.InvariantCulture)).Append("\">\n");
			builder.Append(_trees[i].ToString("\t\t"));
			builder.Append("\t</tree>\n");
		}
		builder.Append("</ensemble>\n");
		return builder.ToString();
	}

	private Split Create(XmlNode node, Dictionary<int, int> fids)
	{
		Split s;
		if (node.FirstChild is null)
		{
			throw new InvalidOperationException("Node does not have a first child.");
		}

		if (node.FirstChild.Name.Equals("feature", StringComparison.OrdinalIgnoreCase)) // this is a split
		{
			var childNodes = node.ChildNodes;

			if (childNodes.Count != 4)
			{
				throw new InvalidDataException("Invalid feature");
			}

			var fid = int.Parse(childNodes[0].FirstChild.Value.Trim()); // <feature>
			fids[fid] = 0;
			var threshold = float.Parse(childNodes[1].FirstChild.Value.Trim()); // <threshold>
			s = new Split(fid, threshold, 0);
			s.SetLeft(Create(childNodes[2], fids));
			s.SetRight(Create(childNodes[3], fids));
		}
		else // this is a stump
		{
			var output = float.Parse(node.FirstChild.FirstChild.Value.Trim());
			s = new Split();
			s.SetOutput(output);
		}
		return s;
	}
}
