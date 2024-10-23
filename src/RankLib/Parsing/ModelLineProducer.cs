using System.Text;
using RankLib.Utilities;

namespace RankLib.Parsing;

public class ModelLineProducer
{
	private const int CarriageReturn = '\r';
	private const int LineFeed = '\n';

	public delegate void LineConsumer(StringBuilder model, bool maybeEndEns);

	public StringBuilder Model { get; } = new();

	private bool ReadUntil(string fullTextChar, int beginOfLineCursor, int endOfLineCursor, StringBuilder model)
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
			var beginOfLineCursor = 0;
			for (var i = 0; i < fullText.Length; i++)
			{
				int charNum = fullText[i];
				if (charNum is CarriageReturn or LineFeed)
				{
					// Read current line from beginOfLineCursor -> i
					if (fullText[beginOfLineCursor] != '#')
					{
						var eolCursor = i;

						while (eolCursor > beginOfLineCursor && fullText[eolCursor] <= 32)
							eolCursor--;

						modelConsumer(Model, ReadUntil(fullText, beginOfLineCursor, eolCursor, Model));
					}

					// Move to the next non-whitespace character
					while (charNum <= 32 && i < fullText.Length)
					{
						charNum = fullText[i];
						beginOfLineCursor = i;
						i++;
					}
				}
			}

			// Process remaining content after the final newline
			modelConsumer(Model, ReadUntil(fullText, beginOfLineCursor, fullText.Length - 1, Model));
		}
		catch (Exception ex)
		{
			throw RankLibException.Create("Error in model loading", ex);
		}
	}
}
