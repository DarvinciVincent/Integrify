import math
import numpy as np
from scipy.stats import t, norm, ttest_1samp, ttest_ind, ttest_rel, f_oneway, f_twoway
import warnings


class ZTest:
    """Perform a Z-test for a single sample"""
    def __init__(self, sample, std, alpha, pop_mean):
        """
        Parameters:
        -----------
        sample: list
            A list of numeric values representing the sample data
        std: float
            The population standard deviation
        alpha: float
            The significance level for the hypothesis test
        pop_mean: float
            The population mean to compare the sample mean against
        """
        assert isinstance(sample, list), "Sample must be a list of numeric values."
        assert std > 0, "Standard deviation must be positive."
        assert 0 < alpha < 1, "Alpha must be between 0 and 1."

        self.sample = sample
        self.std = std
        self.alpha = alpha
        self.pop_mean = pop_mean

    def test(self):
        """Perform the Z-test"""
        sample_mean = np.array(self.sample).mean()
        sample_size = len(self.sample)
        z = (sample_mean - self.pop_mean) / (self.std / math.sqrt(sample_size))
        p_value = 2 * (1 - norm.cdf(abs(z)))

        if p_value > self.alpha:
            print('Fail to reject H0 (Accept H0). The mean of the population is equal to', self.pop_mean)
        else:
            print('Reject H0 (Accept H1). The mean of the population is NOT equal to', self.pop_mean)

    def degree_of_freedom(self):
        """Calculate the degrees of freedom for the Z-test"""
        df = len(self.sample) - 1
        return df

    def critical_value(self):
        """Calculate the critical value for the Z-test"""
        df = self.degree_of_freedom()
        cv = t.ppf(1 - self.alpha / 2, df)  # critical value
        print(f"The critical value given the degree of freedom and the alpha = (+/-) {cv:.3f}")


class TTest:
    """Perform a t-test"""
    def __init__(self, sample, alpha, std=0, pop_mean=0):
        """
        Parameters:
        -----------
        sample: list
            A list of numeric values representing the first sample data
        alpha: float
            The significance level for the hypothesis test
        std: float, optional
            The population standard deviation (only for one-sample t-test)
        pop_mean: float, optional
            The population mean to compare the sample mean against (only for one-sample t-test)
        """
        assert isinstance(sample, list), "Sample must be a list of numeric values."
        assert std > 0, "Standard deviation must be positive."
        assert 0 < alpha < 1, "Alpha must be between 0 and 1."

        self.sample = sample
        self.alpha = alpha
        self.std = std
        self.pop_mean = pop_mean

    def one_sample_test(self):
        t_stat, p_value = ttest_1samp(self.sample, self.pop_mean)

        print(f"t-statistic: {t_stat:.2f}")
        print(f"p-value:: {p_value:.2f}")

        if p_value > self.alpha:
            print('Fail to reject H0 (Accept H0). The mean of the population is equal to', self.pop_mean)
        else:
            print('Reject H0 (Accept H1). The mean of the population is NOT equal to', self.pop_mean)

    def two_sample_test(self, sample2):
        assert isinstance(sample2, list), "Sample 2 must be a list of numeric values."

        t_statistic, p_value = ttest_ind(self.sample, sample2, equal_var=False)

        print(f"t-statistic: {t_statistic:.2f}")
        print(f"p-value: {p_value:.2f}")

        if p_value > self.alpha:
            print('Fail to reject H0 (Accept H0). The mean of the population is equal to', self.pop_mean)
        else:
            print('Reject H0 (Accept H1). The mean of the population is NOT equal to', self.pop_mean)

    def paired_t_test(self, sample2):
        assert isinstance(sample2, list), "Sample 2 must be a list of numeric values."

        t_statistic, p_value = ttest_rel(self.sample, sample2)

        print(f"t-statistic: {t_statistic:.2f}")
        print(f"p-value: {p_value:.2f}")

        if p_value > self.alpha:
            print('Fail to reject H0 (Accept H0). The mean of the population is equal to', self.pop_mean)
        else:
            print('Reject H0 (Accept H1). The mean of the population is NOT equal to', self.pop_mean)

    def one_way_anova(self, sample2=None, sample3=None, sample4=None):
        all_samples = [self.sample, sample2, sample3, sample4]

        f_statistic, p_value = f_oneway(*all_samples)

        if p_value < self.alpha:
            print("Reject the null hypothesis. There is a significant difference in salaries between the departments.")
        else:
            print("Fail to reject the null hypothesis. There is no significant difference in salaries between "
                  "the departments.")

