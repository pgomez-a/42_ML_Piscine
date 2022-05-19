import math

class TinyStatistician(object):
    """
    Introduction to statistic operations
    """
    @staticmethod
    def mean(x):
        """
        Computes the mean of a given non-empty list or array x.
        """
        if not type(x) == list or len(x) == 0:
            return None
        return sum(x) / len(x)

    @staticmethod
    def median(x):
        """
        Computes the median, which is also the 50th percentile, of a given
        non-empty list or array x.
        """
        return TinyStatistician().percentile(x, 50)

    @staticmethod
    def quartile(x):
        """
        Computes the 1st and 3rd quartiles, alse called 25th percentile and
        75th percentile, of a given non-empty list or array x.
        """
        return [TinyStatistician().percentile(x, 25), TinyStatistician().percentile(x, 75)]

    @staticmethod
    def percentile(x, p):
        """
        Computes the expected percentile of a given non-empty list or array x.
        """
        if not type(x) == list or len(x) == 0:
            return None
        sorted_list = x.copy()
        sorted_list.sort()
        pos = (p/100) * len(sorted_list)
        if pos - int(pos) == 0:
            return (sorted_list[int(pos) - 1] + sorted_list[int(pos)]) / 2
        return float(sorted_list[int(pos)])

    @staticmethod
    def var(x):
        """
        Computes the variance of a given non-empty list or array x.
        """
        if not type(x) == list or len(x) == 0:
            return None
        result = 0.0
        mean = TinyStatistician().mean(x)
        for item in x:
            result += (mean - item)**2 
        return result / len(x)

    @staticmethod
    def std(x):
        """
        Computes the standard deviation of a given non-empty list or array x.
        """
        if not type(x) == list or len(x) == 0:
            return None
        return math.sqrt(TinyStatistician().var(x))
