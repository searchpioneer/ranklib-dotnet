using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace RankLib.Utilities;

public class MergeSorter
{
    // TODO: logging
    private static readonly ILogger<MergeSorter> _logger = NullLogger<MergeSorter>.Instance;

    public static void Main(string[] args)
    {
        float[][] f = new float[1000][];
        Random rd = new Random();

        for (int r = 0; r < f.Length; r++)
        {
            f[r] = new float[500];
            for (int i = 0; i < f[r].Length; i++)
            {
                float x = rd.Next(10);
                f[r][i] = x;
            }
        }

        double start = DateTime.UtcNow.Ticks / TimeSpan.TicksPerMillisecond;
        foreach (float[] element in f)
        {
            Sort(element, false);
        }
        double end = DateTime.UtcNow.Ticks / TimeSpan.TicksPerMillisecond;
        _logger.LogInformation($"# {(end - start) / 1000.0} seconds");
    }

    public static int[] Sort(float[] list, bool asc)
    {
        return Sort(list, 0, list.Length - 1, asc);
    }

    public static int[] Sort(float[] list, int begin, int end, bool asc)
    {
        int len = end - begin + 1;
        int[] idx = new int[len];
        int[] tmp = new int[len];

        for (int ii = begin; ii <= end; ii++)
        {
            idx[ii - begin] = ii;
        }

        // Identify natural runs and merge them (first iteration)
        int i = 1, j = 0, k = 0, start = 0;
        int[] ph = new int[len / 2 + 3];
        ph[0] = 0;
        int p = 1;

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
            int np = 1;

            for (int w = 0; w < p - 1; w += 2)
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

    public static int[] Sort(double[] list, bool asc)
    {
        return Sort(list, 0, list.Length - 1, asc);
    }

    public static int[] Sort(double[] list, int begin, int end, bool asc)
    {
        int len = end - begin + 1;
        int[] idx = new int[len];
        int[] tmp = new int[len];

        for (int ii = begin; ii <= end; ii++)
        {
            idx[ii - begin] = ii;
        }

        // Identify natural runs and merge them (first iteration)
        int i = 1, j = 0, k = 0, start = 0;
        int[] ph = new int[len / 2 + 3];
        ph[0] = 0;
        int p = 1;

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
            int np = 1;

            for (int w = 0; w < p - 1; w += 2)
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