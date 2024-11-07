using System.Globalization;
using System.Text;
using System.Xml;
using RankLib.Utilities;

namespace RankLib.Learning.Tree;

public class Ensemble
{
	private readonly List<RegressionTree> _trees = [];
	private readonly List<float> _weights = [];
	private int[] _features = [];

	/// <summary>
	/// Parses a new instance of <see cref="Ensemble"/> from an XML string.
	/// </summary>
	/// <param name="xml">The XML to parse.</param>
	/// <returns>A new instance of <see cref="Ensemble"/></returns>
	public static Ensemble Parse(string xml)
	{
		try
		{
			var ensemble = new Ensemble();
			using var stream = new MemoryStream(Encoding.UTF8.GetBytes(xml));
			var doc = new XmlDocument();
			doc.Load(stream);
			var treeNodes = doc.GetElementsByTagName("tree");
			var fids = new Dictionary<int, int>();
			foreach (XmlNode node in treeNodes)
			{
				// Create a regression tree from this node
				var root = Create(node.FirstChild, fids);
				// Get the weight for this tree
				var weight = float.Parse(node.Attributes["weight"].Value);
				// Add it to the ensemble
				ensemble.Add(new RegressionTree(root), weight);
			}

			ensemble._features = new int[fids.Keys.Count];
			var i = 0;
			foreach (var fid in fids.Keys)
				ensemble._features[i++] = fid;

			return ensemble;
		}
		catch (Exception ex)
		{
			throw RankLibException.Create("Error reading ensemble from xml", ex);
		}
	}

	public IReadOnlyList<RegressionTree> Trees => _trees;

	public IReadOnlyList<float> Weights => _weights;

	public double Variance()
	{
		double variance = 0;
		foreach (var tree in _trees)
			variance += tree.Variance();

		return variance;
	}

	public void Add(RegressionTree tree, float weight)
	{
		_trees.Add(tree);
		_weights.Add(weight);
	}

	public void RemoveAt(int index)
	{
		_trees.RemoveAt(index);
		_weights.RemoveAt(index);
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

	public float Eval(DataPoint dataPoint)
	{
		float s = 0;
		for (var i = 0; i < _trees.Count; i++)
			s = (float)(s + _trees[i].Eval(dataPoint) * _weights[i]);

		return s;
	}

	public override string ToString()
	{
		var builder = new StringBuilder();
		builder.Append("<ensemble>\n");
		for (var i = 0; i < _trees.Count; i++)
		{
			builder.Append("\t<tree id=\"").Append(i + 1)
				.Append("\" weight=\"")
				.Append(_weights[i].ToString(CultureInfo.InvariantCulture))
				.Append("\">\n");
			builder.Append(_trees[i].ToString("\t\t"));
			builder.Append("\t</tree>\n");
		}
		builder.Append("</ensemble>\n");
		return builder.ToString();
	}

	private static Split Create(XmlNode node, Dictionary<int, int> fids)
	{
		if (node.FirstChild is null)
			throw new InvalidOperationException("Node does not have a first child.");

		Split split;
		if (node.FirstChild.Name.Equals("feature", StringComparison.OrdinalIgnoreCase)) // this is a split
		{
			var childNodes = node.ChildNodes;

			if (childNodes.Count != 4)
				throw new ArgumentException("Invalid feature");

			var fid = int.Parse(childNodes[0]!.FirstChild.Value.Trim()); // <feature>
			fids[fid] = 0;
			var threshold = float.Parse(childNodes[1].FirstChild.Value.Trim()); // <threshold>
			split = new Split(fid, threshold, 0)
			{
				Left = Create(childNodes[2], fids),
				Right = Create(childNodes[3], fids)
			};
		}
		else // this is a stump
		{
			var output = float.Parse(node.FirstChild.FirstChild.Value.Trim());
			split = new Split { Output = output };
		}

		return split;
	}
}
