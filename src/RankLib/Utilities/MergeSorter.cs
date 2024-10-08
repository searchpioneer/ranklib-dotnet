namespace RankLib.Utilities;

public static class MergeSorter
{
	public static int[] Sort(float[] list, bool asc) => Sort(list, 0, list.Length - 1, asc);

	public static int[] Sort(float[] list, int begin, int end, bool asc)
	{
		var len = end - begin + 1;
		var idx = new int[len];
		var tmp = new int[len];

		for (var ii = begin; ii <= end; ii++)
		{
			idx[ii - begin] = ii;
		}

		// Identify natural runs and merge them (first iteration)
		int i = 1, j = 0, k = 0, start = 0;
		var ph = new int[len / 2 + 3];
		ph[0] = 0;
		var p = 1;

		do
		{
			start = i - 1;
			while (i < idx.Length && ((asc && list[begin + i] >= list[begin + i - 1]) || (!asc && list[begin + i] <= list[begin + i - 1])))
			{
				i++;
			}

			if (i == idx.Length)
			{
				Array.Copy(idx, start, tmp, k, i - start);
				k = i;
			}
			else
			{
				j = i + 1;
				while (j < idx.Length && ((asc && list[begin + j] >= list[begin + j - 1]) || (!asc && list[begin + j] <= list[begin + j - 1])))
				{
					j++;
				}
				Merge(list, idx, start, i - 1, i, j - 1, tmp, k, asc);
				i = j + 1;
				k = j;
			}
			ph[p++] = k;
		} while (k < idx.Length);

		Array.Copy(tmp, 0, idx, 0, idx.Length);

		// Subsequent iterations
		while (p > 2)
		{
			if (p % 2 == 0)
			{
				ph[p++] = idx.Length;
			}
			k = 0;
			var np = 1;

			for (var w = 0; w < p - 1; w += 2)
			{
				Merge(list, idx, ph[w], ph[w + 1] - 1, ph[w + 1], ph[w + 2] - 1, tmp, k, asc);
				k = ph[w + 2];
				ph[np++] = k;
			}
			p = np;
			Array.Copy(tmp, 0, idx, 0, idx.Length);
		}

		return idx;
	}

	private static void Merge(float[] list, int[] idx, int s1, int e1, int s2, int e2, int[] tmp, int l, bool asc)
	{
		int i = s1, j = s2, k = l;

		while (i <= e1 && j <= e2)
		{
			if (asc)
			{
				if (list[idx[i]] <= list[idx[j]])
				{
					tmp[k++] = idx[i++];
				}
				else
				{
					tmp[k++] = idx[j++];
				}
			}
			else
			{
				if (list[idx[i]] >= list[idx[j]])
				{
					tmp[k++] = idx[i++];
				}
				else
				{
					tmp[k++] = idx[j++];
				}
			}
		}

		while (i <= e1)
		{
			tmp[k++] = idx[i++];
		}

		while (j <= e2)
		{
			tmp[k++] = idx[j++];
		}
	}

	public static int[] Sort(double[] list, bool asc) => Sort(list, 0, list.Length - 1, asc);

	public static int[] Sort(double[] list, int begin, int end, bool asc)
	{
		var len = end - begin + 1;
		var idx = new int[len];
		var tmp = new int[len];

		for (var ii = begin; ii <= end; ii++)
		{
			idx[ii - begin] = ii;
		}

		// Identify natural runs and merge them (first iteration)
		int i = 1, j = 0, k = 0, start = 0;
		var ph = new int[len / 2 + 3];
		ph[0] = 0;
		var p = 1;

		do
		{
			start = i - 1;
			while (i < idx.Length && ((asc && list[begin + i] >= list[begin + i - 1]) || (!asc && list[begin + i] <= list[begin + i - 1])))
			{
				i++;
			}

			if (i == idx.Length)
			{
				Array.Copy(idx, start, tmp, k, i - start);
				k = i;
			}
			else
			{
				j = i + 1;
				while (j < idx.Length && ((asc && list[begin + j] >= list[begin + j - 1]) || (!asc && list[begin + j] <= list[begin + j - 1])))
				{
					j++;
				}
				Merge(list, idx, start, i - 1, i, j - 1, tmp, k, asc);
				i = j + 1;
				k = j;
			}
			ph[p++] = k;
		} while (k < idx.Length);

		Array.Copy(tmp, 0, idx, 0, idx.Length);

		// Subsequent iterations
		while (p > 2)
		{
			if (p % 2 == 0)
			{
				ph[p++] = idx.Length;
			}
			k = 0;
			var np = 1;

			for (var w = 0; w < p - 1; w += 2)
			{
				Merge(list, idx, ph[w], ph[w + 1] - 1, ph[w + 1], ph[w + 2] - 1, tmp, k, asc);
				k = ph[w + 2];
				ph[np++] = k;
			}
			p = np;
			Array.Copy(tmp, 0, idx, 0, idx.Length);
		}

		return idx;
	}

	private static void Merge(double[] list, int[] idx, int s1, int e1, int s2, int e2, int[] tmp, int l, bool asc)
	{
		int i = s1, j = s2, k = l;

		while (i <= e1 && j <= e2)
		{
			if (asc)
			{
				if (list[idx[i]] <= list[idx[j]])
				{
					tmp[k++] = idx[i++];
				}
				else
				{
					tmp[k++] = idx[j++];
				}
			}
			else
			{
				if (list[idx[i]] >= list[idx[j]])
				{
					tmp[k++] = idx[i++];
				}
				else
				{
					tmp[k++] = idx[j++];
				}
			}
		}

		while (i <= e1)
		{
			tmp[k++] = idx[i++];
		}

		while (j <= e2)
		{
			tmp[k++] = idx[j++];
		}
	}
}
