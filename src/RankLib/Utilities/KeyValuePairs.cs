namespace RankLib.Utilities;

public class KeyValuePairs
{
	private readonly List<string> _keys = new();
	private readonly List<string> _values = new();

	public KeyValuePairs(string text)
	{
		if (string.IsNullOrWhiteSpace(text))
		{
			throw new ArgumentException("Input text cannot be null or empty.", nameof(text));
		}

		// Remove the comment part at the end of the line if it exists
		var idx = text.LastIndexOf('#');
		if (idx != -1)
		{
			text = text[..idx].Trim();
		}

		var pairs = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);

		foreach (var pair in pairs)
		{
			var separatorIdx = pair.IndexOf(':');
			if (separatorIdx == -1)
			{
				throw new InvalidOperationException($"Invalid key-value pair: '{pair}'");
			}

			_keys.Add(pair[..separatorIdx].Trim());
			_values.Add(pair[(separatorIdx + 1)..].Trim());
		}
	}

	public IReadOnlyList<string> Keys => _keys;

	public IReadOnlyList<string> Values => _values;
}
