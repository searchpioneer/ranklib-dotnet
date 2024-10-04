using System.Text;
using RankLib.Utilities;

namespace RankLib.Parsing;

public class ModelLineProducer
{
	private const int CarriageReturn = 13;
	private const int LineFeed = 10;

	private readonly StringBuilder _model = new(1000);

	public delegate void LineConsumer(StringBuilder model, bool maybeEndEns);

	public StringBuilder GetModel() => _model;

	private bool ReadUntil(char[] fullTextChar, int beginOfLineCursor, int endOfLineCursor, StringBuilder model)
	{
		var isEnsembleEnd = true;

		// Append lines that don't start with '#'
		if (fullTextChar[beginOfLineCursor] != '#')
		{
			for (var j = beginOfLineCursor; j <= endOfLineCursor; j++)
			{
				model.Append(fullTextChar[j]);
			}
		}

		// Check for ensemble tag
		if (endOfLineCursor > 3)
		{
			isEnsembleEnd = (fullTextChar[endOfLineCursor - 9] == '/' && fullTextChar[endOfLineCursor - 2] == 'l'
				&& fullTextChar[endOfLineCursor - 1] == 'e' && fullTextChar[endOfLineCursor] == '>');
		}

		return isEnsembleEnd;
	}

	public void Parse(string fullText, LineConsumer modelConsumer)
	{
		try
		{
			var fullTextChar = fullText.ToCharArray();
			var beginOfLineCursor = 0;

			for (var i = 0; i < fullTextChar.Length; i++)
			{
				int charNum = fullTextChar[i];

				if (charNum == CarriageReturn || charNum == LineFeed)
				{
					// Read current line from beginOfLineCursor -> i
					if (fullTextChar[beginOfLineCursor] != '#')
					{
						var eolCursor = i;

						while (eolCursor > beginOfLineCursor && fullTextChar[eolCursor] <= 32)
						{
							eolCursor--;
						}

						modelConsumer(_model, ReadUntil(fullTextChar, beginOfLineCursor, eolCursor, _model));
					}

					// Move to the next non-whitespace character
					while (charNum <= 32 && i < fullTextChar.Length)
					{
						charNum = fullTextChar[i];
						beginOfLineCursor = i;
						i++;
					}
				}
			}

			// Process remaining content after the final newline
			modelConsumer(_model, ReadUntil(fullTextChar, beginOfLineCursor, fullTextChar.Length - 1, _model));
		}
		catch (Exception ex)
		{
			throw RankLibError.Create("Error in model loading", ex);
		}
	}
}
