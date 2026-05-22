import numpy as np


# -------------------------------------------------
# Question 1: Joint Gaussian PDF and Marginals
# -------------------------------------------------

def joint_gaussian_pdf(x, y, mu_x=1, mu_y=-2, sigma_x=2, sigma_y=3, rho=0.6):
    """
    Return the bivariate Gaussian PDF f_XY(x,y).

    Use the formula:

    f_XY(x,y) =
    1 / (2*pi*sigma_x*sigma_y*sqrt(1-rho^2))
    *
    exp( -Q / (2*(1-rho^2)) )
    """

    coefficient = 1 / (
        2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2)
    )

    q = (
        ((x - mu_x) / sigma_x) ** 2
        - 2 * rho * ((x - mu_x) / sigma_x) * ((y - mu_y) / sigma_y)
        + ((y - mu_y) / sigma_y) ** 2
    )

    exponent = np.exp(-q / (2 * (1 - rho**2)))

    return coefficient * exponent


def marginal_pdf_x(x, mu_x=1, sigma_x=2):
    """
    Return marginal Gaussian PDF of X.
    """

    coefficient = 1 / (sigma_x * np.sqrt(2 * np.pi))

    exponent = np.exp(
        -((x - mu_x) ** 2) / (2 * sigma_x**2)
    )

    return coefficient * exponent


def marginal_pdf_y(y, mu_y=-2, sigma_y=3):
    """
    Return marginal Gaussian PDF of Y.
    """

    coefficient = 1 / (sigma_y * np.sqrt(2 * np.pi))

    exponent = np.exp(
        -((y - mu_y) ** 2) / (2 * sigma_y**2)
    )

    return coefficient * exponent


def covariance_matrix(sigma_x=2, sigma_y=3, rho=0.6):
    """
    Return covariance matrix:

    [[sigma_x^2, rho*sigma_x*sigma_y],
     [rho*sigma_x*sigma_y, sigma_y^2]]
    """

    return np.array([
        [sigma_x**2, rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y, sigma_y**2]
    ])


def joint_pdf_grid_integral(
    mu_x=1,
    mu_y=-2,
    sigma_x=2,
    sigma_y=3,
    rho=0.6,
    n=250
):
    """
    Numerically approximate integral of joint Gaussian PDF
    over the rectangle:

    [mu_x - 4*sigma_x, mu_x + 4*sigma_x]
    x
    [mu_y - 4*sigma_y, mu_y + 4*sigma_y]

    Use a rectangular grid or trapezoidal numerical integration.
    """

    x_vals = np.linspace(
        mu_x - 4 * sigma_x,
        mu_x + 4 * sigma_x,
        n
    )

    y_vals = np.linspace(
        mu_y - 4 * sigma_y,
        mu_y + 4 * sigma_y,
        n
    )

    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]

    X, Y = np.meshgrid(x_vals, y_vals)

    Z = joint_gaussian_pdf(
        X, Y,
        mu_x, mu_y,
        sigma_x, sigma_y,
        rho
    )

    integral = np.sum(Z) * dx * dy

    return integral


# -------------------------------------------------
# Question 2: Simulation and Independence
# -------------------------------------------------

def generate_joint_gaussian_samples(
    n=100000,
    mu_x=1,
    mu_y=-2,
    sigma_x=2,
    sigma_y=3,
    rho=0.6,
    seed=0
):
    """
    Generate n samples from a jointly Gaussian distribution.

    Return two arrays:
    x_samples, y_samples

    Hint:
    Use np.random.multivariate_normal.
    """

    np.random.seed(seed)

    mean = [mu_x, mu_y]

    cov = covariance_matrix(
        sigma_x,
        sigma_y,
        rho
    )

    samples = np.random.multivariate_normal(
        mean,
        cov,
        size=n
    )

    x_samples = samples[:, 0]
    y_samples = samples[:, 1]

    return x_samples, y_samples


def sample_means(x_samples, y_samples):
    """
    Return sample means of X and Y.
    """

    mean_x = np.mean(x_samples)
    mean_y = np.mean(y_samples)

    return mean_x, mean_y


def sample_covariance_matrix(x_samples, y_samples):
    """
    Return 2 by 2 sample covariance matrix.

    Use denominator n-1.
    """

    return np.cov(x_samples, y_samples, ddof=1)


def sample_correlation(x_samples, y_samples):
    """
    Return sample correlation coefficient.
    """

    return np.corrcoef(x_samples, y_samples)[0, 1]


def gaussian_independence_check(rho):
    """
    For jointly Gaussian variables:
    return True if rho is zero, otherwise False.
    """

    return rho == 0


def zero_rho_covariance_check(n=100000):

    x_samples, y_samples = generate_joint_gaussian_samples(
        n=n,
        rho=0
    )

    cov_matrix = sample_covariance_matrix(
        x_samples,
        y_samples
    )

    sample_cov = cov_matrix[0, 1]

    return bool(np.abs(sample_cov) < 0.05)


def nonzero_rho_covariance_check(n=100000):

    sigma_x = 2
    sigma_y = 3
    rho = 0.6

    theoretical_cov = rho * sigma_x * sigma_y

    x_samples, y_samples = generate_joint_gaussian_samples(
        n=n,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        rho=rho
    )

    cov_matrix = sample_covariance_matrix(
        x_samples,
        y_samples
    )

    sample_cov = cov_matrix[0, 1]

    return bool(
        np.abs(sample_cov - theoretical_cov) < 0.1
    )