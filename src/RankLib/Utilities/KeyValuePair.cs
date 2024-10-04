namespace RankLib.Utilities;

public class KeyValuePair
{
	protected List<string> _keys = new();
	protected List<string> _values = new();

	public KeyValuePair(string text)
	{
		try
		{
			// Remove the comment part at the end of the line if it exists
			var idx = text.LastIndexOf('#');
			if (idx != -1)
			{
				text = text.Substring(0, idx).Trim();
			}

			var fs = text.Split(' ');

			foreach (var item in fs)
			{
				var trimmed = item.Trim();
				if (string.IsNullOrEmpty(trimmed))
				{
					continue;
				}

				_keys.Add(GetKey(trimmed));
				_values.Add(GetValue(trimmed));
			}
		}
		catch (Exception ex)
		{
			throw new InvalidOperationException("Error in KeyValuePair constructor", ex);
		}
	}

	public List<string> Keys() => _keys;

	public List<string> Values() => _values;

	private string GetKey(string pair) => pair.Substring(0, pair.IndexOf(':'));

	private string GetValue(string pair) => pair.Substring(pair.LastIndexOf(':') + 1);
}
