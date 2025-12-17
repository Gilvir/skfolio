r"""
==============================
Cross-Sectional Factor Model.
==============================

This tutorial shows how to use the :class:`~skfolio.prior.CrossSectionalFactorModel`
estimator in the :class:`~skfolio.optimization.MeanRisk` optimization.

The :class:`~skfolio.prior.CrossSectionalFactorModel` implements a Barra-style
cross-sectional factor model where factor exposures (betas) are pre-computed
characteristics (e.g., sector membership, style factors like value, momentum, size),
and factor returns are estimated via cross-sectional regression at each time step.

This differs from time-series factor models (like :class:`~skfolio.prior.FactorModel`)
where factor returns are observed and betas are estimated via time-series regression.

**When to use Cross-Sectional vs Time-Series Factor Models:**

Use **Cross-Sectional Factor Model** when:
    * You have pre-computed factor exposures (characteristics)
    * Factors represent cross-sectional attributes (sectors, style scores)
    * You want to estimate factor returns from asset returns
    * Following Barra/Axioma methodology

Use **Time-Series Factor Model** (:class:`~skfolio.prior.FactorModel`) when:
    * You have observable factor returns (Fama-French factors, ETFs)
    * You want to estimate factor exposures (betas) from historical data
    * Following traditional factor analysis methodology

The mathematical model is:

.. math:: r_{i,t} = \\sum_{k=1}^{K} \\beta_{i,k,t} \\times f_{k,t} + \\epsilon_{i,t}

Where:
    * :math:`r_{i,t}` is the return of asset i at time t
    * :math:`\\beta_{i,k,t}` is the **known** exposure of asset i to factor k at time t
    * :math:`f_{k,t}` is the **unknown** return of factor k at time t (estimated via cross-sectional regression)
    * :math:`\\epsilon_{i,t}` is the asset-specific residual return

In this tutorial we will build a Maximum Sharpe Ratio portfolio using the
`CrossSectionalFactorModel` estimator with sector and style factors.
"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>` and create synthetic factor exposures:
import numpy as np
import sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from skfolio import Population, RiskMeasure
from skfolio.datasets import load_sp500_dataset
from skfolio.moments import LedoitWolf
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import CrossSectionalFactorModel, CrossSectionalRegression, EmpiricalPrior

# Enable metadata routing for sklearn>=1.4
sklearn.set_config(enable_metadata_routing=True)

prices = load_sp500_dataset()
X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

# %%
# Factor Exposures
# =================
# For this example, we create synthetic factor exposures representing:
#
# 1. **Sector factors**: Binary sector membership (one-hot encoded)
# 2. **Style factors**: Value, momentum, and size scores (standardized)
#
# In practice, these would come from:
#     * Sector data: GICS classifications
#     * Style factors: Calculated from fundamentals (P/B, trailing returns, market cap)
#     * Risk factors: Volatility, beta, etc.
#
# Note: The model automatically adds a market factor (column of ones), so you only
# need to provide sector and style exposures separately.

n_observations, n_assets = X_train.shape

# Create synthetic exposures that vary over time
np.random.seed(42)

# Sector exposures (binary, constant over time for each asset)
# Use N-1 sectors for sum-to-zero constraint (Barra standard)
n_sectors = 5
# Ensure balanced sector assignment (each sector gets roughly equal number of assets)
sector_assignment = np.tile(np.arange(n_sectors - 1), n_assets // (n_sectors - 1) + 1)[:n_assets]
np.random.shuffle(sector_assignment)
sector_exposures_train = np.zeros((n_observations, n_assets, n_sectors - 1))
for i in range(n_assets):
    sector_exposures_train[:, i, sector_assignment[i]] = 1.0

# Style exposures (continuous, varying over time)
# Value factor: simulate from fundamentals
value_exposures = np.random.randn(n_observations, n_assets, 1)
value_exposures = (value_exposures - value_exposures.mean(axis=1, keepdims=True)) / (
    value_exposures.std(axis=1, keepdims=True) + 1e-8
)

# Momentum factor: simulate from trailing returns
momentum_exposures = np.random.randn(n_observations, n_assets, 1)
momentum_exposures = (
    momentum_exposures - momentum_exposures.mean(axis=1, keepdims=True)
) / (momentum_exposures.std(axis=1, keepdims=True) + 1e-8)

# Size factor: simulate from market cap
size_exposures = np.random.randn(n_observations, n_assets, 1)
size_exposures = (size_exposures - size_exposures.mean(axis=1, keepdims=True)) / (
    size_exposures.std(axis=1, keepdims=True) + 1e-8
)

# Combine style exposures: shape (n_observations, n_assets, n_style_factors)
style_exposures_train = np.concatenate(
    [value_exposures, momentum_exposures, size_exposures], axis=2
)

print(f"Exposures shape:")
print(f"  - Sector: {sector_exposures_train.shape} ({n_sectors-1} sectors for sum-to-zero)")
print(f"  - Style: {style_exposures_train.shape} (value, momentum, size)")
print(f"  - Total factors: 1 market + {n_sectors-1} sectors + 3 styles = {1 + (n_sectors-1) + 3}")

# %%
# Market Capitalization Weights
# ==============================
# Barra-style models use market-cap weighting in the cross-sectional regression.
# Larger firms have more influence on factor return estimates.
#
# We simulate market caps that vary over time:
market_caps_train = np.abs(np.random.lognormal(mean=10, sigma=1, size=X_train.shape))
market_caps_train = market_caps_train / market_caps_train.sum(
    axis=1, keepdims=True
)  # Normalize to sum to 1

print(f"Market caps shape: {market_caps_train.shape}")

# %%
# Cross-Sectional Factor Model
# =============================
# We create a Maximum Sharpe Ratio model using the Cross-Sectional Factor Model:
model_cross_sectional = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    prior_estimator=CrossSectionalFactorModel(
        # Use LedoitWolf for more robust factor covariance estimation
        factor_prior_estimator=EmpiricalPrior(covariance_estimator=LedoitWolf())
    ).set_fit_request(
        sector_exposures=True,
        style_exposures=True,
        sample_weight=True,
    ),
    portfolio_params=dict(name="Cross-Sectional Factor Model"),
)

# Fit with sector exposures, style exposures, and market-cap weights
model_cross_sectional.fit(
    X=X_train,
    sector_exposures=sector_exposures_train,
    style_exposures=style_exposures_train,
    sample_weight=market_caps_train,
)


print("Optimal weights:")
print(model_cross_sectional.weights_)

# %%
# Custom Regression Method
# ========================
# By default, the cross-sectional regression uses :class:`~sklearn.linear_model.LinearRegression`.
# You can use any scikit-learn regressor, for example Ridge regression for regularization:

model_ridge = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    prior_estimator=CrossSectionalFactorModel(
        cross_sectional_regressor=CrossSectionalRegression(
            linear_regressor=Ridge(alpha=0.1)
        ),
        # Use LedoitWolf for more robust factor covariance estimation
        factor_prior_estimator=EmpiricalPrior(covariance_estimator=LedoitWolf()),
    ).set_fit_request(sector_exposures=True, style_exposures=True, sample_weight=True),
    portfolio_params=dict(name="Cross-Sectional (Ridge)"),
)

model_ridge.fit(
    X=X_train,
    sector_exposures=sector_exposures_train,
    style_exposures=style_exposures_train,
    sample_weight=market_caps_train,
)

# %%
# Factor Analysis
# ===============
# We can access the estimated factor returns and residuals:
prior_estimator = model_cross_sectional.prior_estimator_

# Factor returns: shape (n_observations, n_factors)
factor_returns = prior_estimator.cross_sectional_regressor_.factor_returns_
print(f"\nFactor returns shape: {factor_returns.shape}")
print(f"Mean factor returns:\n{factor_returns.mean(axis=0)}")

# Residuals: shape (n_observations, n_assets)
residuals = prior_estimator.cross_sectional_regressor_.residuals_
print(f"\nResiduals shape: {residuals.shape}")
print(f"Mean absolute residual: {np.abs(residuals).mean():.4f}")

# %%
# Return Distribution
# ===================
# The fitted prior estimator contains the :class:`~skfolio.prior.ReturnDistribution`:
return_distribution = prior_estimator.return_distribution_

print(f"\nExpected returns (mu) shape: {return_distribution.mu.shape}")
print(f"Covariance matrix shape: {return_distribution.covariance.shape}")
print(f"Mean expected return: {return_distribution.mu.mean():.4%}")
print(f"Mean volatility: {np.sqrt(np.diag(return_distribution.covariance)).mean():.4%}")

# %%
# Empirical Model Comparison
# ===========================
# For comparison, we create a Maximum Sharpe Ratio model using the default
# Empirical estimator (no factor structure):
model_empirical = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    portfolio_params=dict(name="Empirical (No Factors)"),
)
model_empirical.fit(X_train)

# %%
# Test Set Prediction
# ===================
# To predict on the test set, we need test set exposures and market caps:
n_test_obs = X_test.shape[0]

# Create test exposures (in practice, these would be calculated from actual data)
# Sector exposures (same sector assignment as training)
sector_exposures_test = np.zeros((n_test_obs, n_assets, n_sectors - 1))
for i in range(n_assets):
    sector_exposures_test[:, i, sector_assignment[i]] = 1.0

# Style exposures
value_exposures_test = np.random.randn(n_test_obs, n_assets, 1)
value_exposures_test = (
    value_exposures_test - value_exposures_test.mean(axis=1, keepdims=True)
) / (value_exposures_test.std(axis=1, keepdims=True) + 1e-8)

momentum_exposures_test = np.random.randn(n_test_obs, n_assets, 1)
momentum_exposures_test = (
    momentum_exposures_test - momentum_exposures_test.mean(axis=1, keepdims=True)
) / (momentum_exposures_test.std(axis=1, keepdims=True) + 1e-8)

size_exposures_test = np.random.randn(n_test_obs, n_assets, 1)
size_exposures_test = (
    size_exposures_test - size_exposures_test.mean(axis=1, keepdims=True)
) / (size_exposures_test.std(axis=1, keepdims=True) + 1e-8)

style_exposures_test = np.concatenate(
    [value_exposures_test, momentum_exposures_test, size_exposures_test], axis=2
)

market_caps_test = np.abs(np.random.lognormal(mean=10, sigma=1, size=X_test.shape))
market_caps_test = market_caps_test / market_caps_test.sum(axis=1, keepdims=True)

# Predict on test set
ptf_cross_sectional_test = model_cross_sectional.predict(X_test)
ptf_ridge_test = model_ridge.predict(X_test)
ptf_empirical_test = model_empirical.predict(X_test)

population = Population([ptf_cross_sectional_test, ptf_ridge_test, ptf_empirical_test])

# %%
# Performance Comparison
# ======================
# Let's compare the cumulative returns:
print("\n" + "=" * 60)
print("PERFORMANCE COMPARISON")
print("=" * 60)

# Uncomment to display plot in interactive environment:
# fig = population.plot_cumulative_returns()
# show(fig)

# %%
# |
#
# Summary statistics:
print("\n" + population.summary().to_string())

# %%
# |
#
# Portfolio composition:
# Uncomment to display plot in interactive environment:
# population.plot_composition()
