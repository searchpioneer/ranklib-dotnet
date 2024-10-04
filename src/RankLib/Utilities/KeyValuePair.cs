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
            int idx = text.LastIndexOf('#');
            if (idx != -1)
            {
                text = text.Substring(0, idx).Trim();
            }

            string[] fs = text.Split(' ');

            foreach (var item in fs)
            {
                string trimmed = item.Trim();
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

    public List<string> Keys()
    {
        return _keys;
    }

    public List<string> Values()
    {
        return _values;
    }

    private string GetKey(string pair)
    {
        return pair.Substring(0, pair.IndexOf(':'));
    }

    private string GetValue(string pair)
    {
        return pair.Substring(pair.LastIndexOf(':') + 1);
    }
}
