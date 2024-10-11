using System.Collections;

namespace RankLib.Utilities;

public class KeyValuePairs : IReadOnlyList<KeyValuePair<string, string>>
{
	private readonly List<KeyValuePair<string, string>> _pairs = [];

	public KeyValuePairs(string text)
	{
		if (string.IsNullOrWhiteSpace(text))
			throw new ArgumentException("Input text cannot be null or empty.", nameof(text));

		var spanText = text.AsSpan();

		// Remove the comment part at the end of the line if it exists
		var idx = spanText.LastIndexOf('#');
		if (idx != -1)
			spanText = spanText[..idx].Trim();

		while (!spanText.IsEmpty)
		{
			// Find the next space to get the key-value pair
			var spaceIndex = spanText.IndexOf(' ');
			ReadOnlySpan<char> pair;

			if (spaceIndex == -1)
			{
				pair = spanText;
				spanText = ReadOnlySpan<char>.Empty;
			}
			else
			{
				pair = spanText[..spaceIndex];
				spanText = spanText[(spaceIndex + 1)..].TrimStart();
			}

			// Find the separator between key and value
			var separatorIdx = pair.IndexOf(':');
			if (separatorIdx == -1)
			{
				throw new InvalidOperationException($"Invalid key-value pair: '{pair.ToString()}'");
			}

			_pairs.Add(KeyValuePair.Create(
				pair[..separatorIdx].Trim().ToString(),
				pair[(separatorIdx + 1)..].Trim().ToString()));
		}
	}

	public IEnumerator<KeyValuePair<string, string>> GetEnumerator() => _pairs.GetEnumerator();

	IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

	public int Count => _pairs.Count;

	public KeyValuePair<string, string> this[int index] => _pairs[index];
}
