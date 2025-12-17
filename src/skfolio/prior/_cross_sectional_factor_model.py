"""Cross-Sectional Factor Model estimator."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import numpy as np
import numpy.typing as npt
import sklearn.base as skb
import sklearn.linear_model as skl
import sklearn.utils.metadata_routing as skm
import sklearn.utils.validation as skv

import skfolio.measures as sm
from skfolio.prior._base import BasePrior, ReturnDistribution
from skfolio.prior._empirical import EmpiricalPrior
from skfolio.utils.stats import cov_nearest
from skfolio.utils.tools import check_estimator


class CrossSectionalRegression(skb.BaseEstimator):
    r"""Cross-Sectional Regression estimator for Barra-style factor models.

    Estimates factor returns by fitting a linear regressor at each time point,
    regressing asset returns on factor exposures. Uses two-stage regression
    where style factors are estimated on residuals from sector factors
    (standard Barra/Axioma methodology).

    At each time point t, this estimator solves:

    **Stage 1 (Sector factors):**

    .. math:: r_{i,t} = \sum_{k=1}^{K_s} \beta^{sector}_{i,k,t} f^{sector}_{k,t} + \epsilon^{(1)}_{i,t}

    **Stage 2 (Style factors on residuals):**

    .. math:: \epsilon^{(1)}_{i,t} = \sum_{k=1}^{K_{st}} \beta^{style}_{i,k,t} f^{style}_{k,t} + \epsilon_{i,t}

    Parameters
    ----------
    linear_regressor : BaseEstimator, optional
        Linear regressor used to fit exposures on returns at each time point.
        The regressor should support `sample_weight` in its `fit` method for
        weighted least squares (e.g., market-cap weighting).
        The default (`None`) is to use `LinearRegression(fit_intercept=False)`.

    sqrt_weights : bool, default=True
        If True, apply square root to sample weights before regression.
        This is the standard Barra/Axioma convention where the variance of
        specific returns is assumed to be inversely proportional to
        sqrt(market_cap).

        From the Barra USE4 methodology: "Factor returns are estimated using
        weighted least-squares regression, assuming that the variance of specific
        returns is inversely proportional to the square root of total market
        capitalization."

        Set to False to use raw sample weights directly.

    include_market : bool, default=True
        If True, automatically prepend a market factor (column of ones) to
        the sector exposures. This is standard in Barra-style models.

    Attributes
    ----------
    factor_returns_ : ndarray of shape (n_observations, n_factors)
        Estimated factor returns time series.
        Columns are ordered as: [market (if include_market), sectors, styles]

    sector_factor_returns_ : ndarray of shape (n_observations, n_sector_factors)
        Estimated sector factor returns (includes market if include_market=True).

    style_factor_returns_ : ndarray of shape (n_observations, n_style_factors)
        Estimated style factor returns.

    residuals_ : ndarray of shape (n_observations, n_assets)
        Final regression residuals (idiosyncratic returns).

    n_observations_ : int
        Number of time observations.

    n_assets_ : int
        Number of assets.

    n_sector_factors_ : int
        Number of sector factors (includes market if include_market=True).

    n_style_factors_ : int
        Number of style factors.

    Notes
    -----
    **Sector Sum-to-Zero Constraint:**

    If you want sector factor returns to sum to zero (Barra/Axioma standard),
    provide N-1 sectors in `sector_exposures` instead of N. The dropped sector
    is implicitly -sum(other sectors).

    Examples
    --------
    >>> import numpy as np
    >>> from skfolio.prior import CrossSectionalRegression
    >>> # Generate sample data
    >>> n_obs, n_assets = 100, 50
    >>> returns = np.random.randn(n_obs, n_assets) * 0.01
    >>> # Sector exposures: 9 sectors (10th is implicit for sum-to-zero)
    >>> sector_exposures = np.zeros((n_obs, n_assets, 9))
    >>> for i in range(n_assets):
    ...     sector_exposures[:, i, i % 9] = 1.0
    >>> # Style exposures: 3 style factors
    >>> style_exposures = np.random.randn(n_obs, n_assets, 3)
    >>> market_caps = np.random.uniform(1e6, 1e9, (n_obs, n_assets))
    >>> # Fit model
    >>> cs_reg = CrossSectionalRegression()
    >>> cs_reg.fit(returns, sector_exposures, style_exposures, sample_weight=market_caps)
    CrossSectionalRegression()
    >>> print(cs_reg.factor_returns_.shape)  # 1 market + 9 sectors + 3 styles = 13
    (100, 13)
    """

    n_observations_: int
    n_assets_: int
    n_sector_factors_: int
    n_style_factors_: int
    factor_returns_: np.ndarray
    sector_factor_returns_: np.ndarray
    style_factor_returns_: np.ndarray
    residuals_: np.ndarray

    def __init__(
        self,
        linear_regressor: skb.BaseEstimator | None = None,
        sqrt_weights: bool = True,
        include_market: bool = True,
    ):
        self.linear_regressor = linear_regressor
        self.sqrt_weights = sqrt_weights
        self.include_market = include_market

    def get_metadata_routing(self):
        """Get metadata routing for this estimator."""
        router = skm.MetadataRouter(owner=self.__class__.__name__).add(
            linear_regressor=self.linear_regressor,
            method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
        )
        return router

    def fit(
        self,
        X: npt.ArrayLike,
        sector_exposures: npt.ArrayLike,
        style_exposures: npt.ArrayLike,
        sample_weight: npt.ArrayLike | None = None,
        **fit_params,
    ):
        """Fit the cross-sectional regression estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Asset returns.

        sector_exposures : array-like of shape (n_observations, n_assets, n_sectors)
            Sector factor exposures for each asset at each time point.
            Typically binary (one-hot encoded sector membership).
            Use N-1 sectors if you want sum-to-zero constraint.

        style_exposures : array-like of shape (n_observations, n_assets, n_styles)
            Style factor exposures for each asset at each time point.
            Typically continuous values (e.g., value, momentum, size scores).

        sample_weight : array-like of shape (n_observations, n_assets), optional
            Sample weights (e.g., market capitalizations) for weighted regression.
            If sqrt_weights=True (default), sqrt is applied internally.

        **fit_params : dict
            Parameters to pass to the underlying linear regressor.

        Returns
        -------
        self : CrossSectionalRegression
            Fitted estimator.
        """
        routed_params = skm.process_routing(self, "fit", **fit_params)

        # Validate inputs
        X = skv.check_array(X, dtype=float, ensure_2d=True, allow_nd=False)
        sector_exposures = np.asarray(sector_exposures, dtype=float)
        style_exposures = np.asarray(style_exposures, dtype=float)

        n_observations, n_assets = X.shape

        # Validate sector exposures
        if sector_exposures.ndim != 3:
            raise ValueError(
                f"sector_exposures must be 3D array of shape "
                f"(n_observations, n_assets, n_sectors), got shape {sector_exposures.shape}"
            )
        if sector_exposures.shape[:2] != (n_observations, n_assets):
            raise ValueError(
                f"sector_exposures shape {sector_exposures.shape} doesn't match "
                f"X shape {X.shape} in first two dimensions"
            )

        # Validate style exposures
        if style_exposures.ndim != 3:
            raise ValueError(
                f"style_exposures must be 3D array of shape "
                f"(n_observations, n_assets, n_styles), got shape {style_exposures.shape}"
            )
        if style_exposures.shape[:2] != (n_observations, n_assets):
            raise ValueError(
                f"style_exposures shape {style_exposures.shape} doesn't match "
                f"X shape {X.shape} in first two dimensions"
            )

        n_sectors = sector_exposures.shape[2]
        n_styles = style_exposures.shape[2]

        # Add market factor if requested
        if self.include_market:
            market_factor = np.ones((n_observations, n_assets, 1))
            sector_exposures_full = np.concatenate([market_factor, sector_exposures], axis=2)
            n_sector_factors = 1 + n_sectors
        else:
            sector_exposures_full = sector_exposures
            n_sector_factors = n_sectors

        self.n_observations_ = n_observations
        self.n_assets_ = n_assets
        self.n_sector_factors_ = n_sector_factors
        self.n_style_factors_ = n_styles

        # Validate and transform sample weights
        if sample_weight is not None:
            sample_weight = skv.check_array(
                sample_weight, dtype=float, ensure_2d=True, allow_nd=False
            )
            if sample_weight.shape != X.shape:
                raise ValueError(
                    f"sample_weight must have same shape as X: "
                    f"expected {X.shape}, got {sample_weight.shape}"
                )
            if self.sqrt_weights:
                sample_weight = np.sqrt(sample_weight)

        # Initialize regressor
        _linear_regressor = check_estimator(
            self.linear_regressor,
            default=skl.LinearRegression(fit_intercept=False),
            check_type=skb.BaseEstimator,
        )

        # Initialize output arrays
        self.sector_factor_returns_ = np.zeros((n_observations, n_sector_factors))
        self.style_factor_returns_ = np.zeros((n_observations, n_styles))
        self.residuals_ = np.zeros_like(X)

        # Run two-stage cross-sectional regression at each time point
        for t in range(n_observations):
            returns_t = X[t]
            sector_exp_t = sector_exposures_full[t]
            style_exp_t = style_exposures[t]
            weights_t = sample_weight[t] if sample_weight is not None else None

            # Handle NaN values
            if weights_t is not None:
                valid_mask = ~(
                    np.isnan(returns_t)
                    | np.isnan(weights_t)
                    | np.any(np.isnan(sector_exp_t), axis=1)
                    | np.any(np.isnan(style_exp_t), axis=1)
                )
            else:
                valid_mask = ~(
                    np.isnan(returns_t)
                    | np.any(np.isnan(sector_exp_t), axis=1)
                    | np.any(np.isnan(style_exp_t), axis=1)
                )

            if not valid_mask.any():
                self.sector_factor_returns_[t] = np.nan
                self.style_factor_returns_[t] = np.nan
                self.residuals_[t] = np.nan
                continue

            returns_valid = returns_t[valid_mask]
            sector_exp_valid = sector_exp_t[valid_mask]
            style_exp_valid = style_exp_t[valid_mask]
            weights_valid = weights_t[valid_mask] if weights_t is not None else None

            try:
                # Stage 1: Estimate sector factor returns (includes market)
                regressor_sector = skb.clone(_linear_regressor)
                if weights_valid is not None:
                    regressor_sector.fit(
                        sector_exp_valid,
                        returns_valid,
                        sample_weight=weights_valid,
                        **routed_params.linear_regressor.fit,
                    )
                else:
                    regressor_sector.fit(
                        sector_exp_valid,
                        returns_valid,
                        **routed_params.linear_regressor.fit,
                    )
                self.sector_factor_returns_[t] = regressor_sector.coef_

                # Compute residuals from sector factors
                sector_fitted = sector_exp_t @ self.sector_factor_returns_[t]
                sector_residuals = returns_t - sector_fitted
                sector_residuals_valid = sector_residuals[valid_mask]

                # Stage 2: Estimate style factor returns on sector residuals
                regressor_style = skb.clone(_linear_regressor)
                if weights_valid is not None:
                    regressor_style.fit(
                        style_exp_valid,
                        sector_residuals_valid,
                        sample_weight=weights_valid,
                        **routed_params.linear_regressor.fit,
                    )
                else:
                    regressor_style.fit(
                        style_exp_valid,
                        sector_residuals_valid,
                        **routed_params.linear_regressor.fit,
                    )
                self.style_factor_returns_[t] = regressor_style.coef_

                # Compute final residuals
                style_fitted = style_exp_t @ self.style_factor_returns_[t]
                self.residuals_[t] = sector_residuals - style_fitted

            except Exception:
                self.sector_factor_returns_[t] = np.nan
                self.style_factor_returns_[t] = np.nan
                self.residuals_[t] = np.nan

        # Combine factor returns: [sector (incl. market), styles]
        self.factor_returns_ = np.hstack([
            self.sector_factor_returns_,
            self.style_factor_returns_
        ])

        return self


class CrossSectionalFactorModel(BasePrior):
    r"""Cross-Sectional Factor Model (Barra-style) prior estimator.

    This model estimates asset return distribution using characteristic-based factors
    through cross-sectional regression. Uses two-stage regression where style factors
    are estimated on residuals from sector factors (standard Barra/Axioma methodology).

    **Mathematical Framework:**

    At each time t, the two-stage cross-sectional regression is:

    **Stage 1:** :math:`r_{i,t} = \sum_{k} \beta^{sector}_{i,k,t} f^{sector}_{k,t} + \epsilon^{(1)}_{i,t}`

    **Stage 2:** :math:`\epsilon^{(1)}_{i,t} = \sum_{k} \beta^{style}_{i,k,t} f^{style}_{k,t} + \epsilon_{i,t}`

    Asset moments are constructed as:

    .. math::
        \mu_i = \sum_{k} \bar{\beta}_{i,k} \mu_{f,k}

    .. math::
        \Sigma_{i,j} = \sum_{k,l} \bar{\beta}_{i,k} \Sigma_{f,k,l} \bar{\beta}_{j,l} + D_i \delta_{i,j}

    Parameters
    ----------
    cross_sectional_regressor : CrossSectionalRegression, optional
        Estimator for cross-sectional regression to extract factor returns.
        The default (`None`) uses `CrossSectionalRegression()`.

    factor_prior_estimator : BasePrior, optional
        Prior estimator for factor returns distribution.
        The default (`None`) uses `EmpiricalPrior()`.

    residual_variance : bool, default=True
        If True, add diagonal specific risk (residual variance) to the factor covariance.

    sqrt_weights : bool, default=True
        If True, apply square root to sample weights before regression.
        This is the standard Barra/Axioma convention.

    include_market : bool, default=True
        If True, automatically prepend a market factor (column of ones) to
        the sector exposures.

    higham : bool, default=False
        If True, use Higham (2002) algorithm for nearest PSD covariance matrix.

    max_iteration : int, default=100
        Maximum iterations for Higham algorithm.

    Attributes
    ----------
    return_distribution_ : ReturnDistribution
        Fitted return distribution for assets based on factor model.

    factor_returns_ : ndarray of shape (n_observations, n_factors)
        Estimated factor returns time series.

    cross_sectional_regressor_ : CrossSectionalRegression
        Fitted cross-sectional regression estimator.

    factor_prior_estimator_ : BasePrior
        Fitted factor prior estimator.

    n_features_in_ : int
        Number of assets seen during fit.

    Notes
    -----
    **Sector Sum-to-Zero Constraint:**

    If you want sector factor returns to sum to zero (Barra/Axioma standard),
    provide N-1 sectors in `sector_exposures` instead of N. The dropped sector
    is implicitly -sum(other sectors).

    Examples
    --------
    >>> from skfolio.prior import CrossSectionalFactorModel
    >>> from skfolio.optimization import MeanRisk
    >>> import numpy as np
    >>>
    >>> # Generate sample data
    >>> n_obs, n_assets = 100, 50
    >>> returns = np.random.randn(n_obs, n_assets) * 0.01
    >>> # Sector exposures (9 sectors for sum-to-zero with 10 actual sectors)
    >>> sector_exposures = np.zeros((n_obs, n_assets, 9))
    >>> for i in range(n_assets):
    ...     sector_exposures[:, i, i % 9] = 1.0
    >>> # Style exposures
    >>> style_exposures = np.random.randn(n_obs, n_assets, 3)
    >>> market_caps = np.random.uniform(1e6, 1e9, (n_obs, n_assets))
    >>>
    >>> # Create and fit model
    >>> model = CrossSectionalFactorModel()
    >>> model.fit(returns, sector_exposures=sector_exposures,
    ...           style_exposures=style_exposures, sample_weight=market_caps)
    CrossSectionalFactorModel()

    References
    ----------
    .. [1] "Barra Risk Model Handbook", MSCI (2015)

    See Also
    --------
    FactorModel : Time-series factor model
    CrossSectionalRegression : Cross-sectional regression estimator
    """

    factor_returns_: np.ndarray
    cross_sectional_regressor_: CrossSectionalRegression
    factor_prior_estimator_: BasePrior
    n_features_in_: int
    feature_names_in_: np.ndarray

    def __init__(
        self,
        cross_sectional_regressor: CrossSectionalRegression | None = None,
        factor_prior_estimator: BasePrior | None = None,
        residual_variance: bool = True,
        sqrt_weights: bool = True,
        include_market: bool = True,
        higham: bool = False,
        max_iteration: int = 100,
    ):
        self.cross_sectional_regressor = cross_sectional_regressor
        self.factor_prior_estimator = factor_prior_estimator
        self.residual_variance = residual_variance
        self.sqrt_weights = sqrt_weights
        self.include_market = include_market
        self.higham = higham
        self.max_iteration = max_iteration

    def get_metadata_routing(self):
        """Get metadata routing for this estimator."""
        router = (
            skm.MetadataRouter(owner=self.__class__.__name__)
            .add_self_request(self)
            .add(
                factor_prior_estimator=self.factor_prior_estimator,
                method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
            )
            .add(
                cross_sectional_regressor=self.cross_sectional_regressor,
                method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
            )
        )
        return router

    def set_fit_request(
        self,
        *,
        sector_exposures: bool | None = None,
        style_exposures: bool | None = None,
        sample_weight: bool | None = None,
    ):
        """Request metadata to be passed to the fit method.

        Parameters
        ----------
        sector_exposures : bool, default=None
            Whether to request sector_exposures for fit.

        style_exposures : bool, default=None
            Whether to request style_exposures for fit.

        sample_weight : bool, default=None
            Whether to request sample_weight for fit.

        Returns
        -------
        self
            Updated estimator.
        """
        if not hasattr(self, "_metadata_request"):
            self._metadata_request = skm.MetadataRequest(owner=self.__class__.__name__)

        if sector_exposures is not None:
            if sector_exposures:
                self._metadata_request.fit.add_request(
                    param="sector_exposures", alias="sector_exposures"
                )
            else:
                self._metadata_request.fit.remove_request(param="sector_exposures")

        if style_exposures is not None:
            if style_exposures:
                self._metadata_request.fit.add_request(
                    param="style_exposures", alias="style_exposures"
                )
            else:
                self._metadata_request.fit.remove_request(param="style_exposures")

        if sample_weight is not None:
            if sample_weight:
                self._metadata_request.fit.add_request(
                    param="sample_weight", alias="sample_weight"
                )
            else:
                self._metadata_request.fit.remove_request(param="sample_weight")

        return self

    def fit(
        self,
        X: npt.ArrayLike,
        y: Any = None,
        sector_exposures: npt.ArrayLike | None = None,
        style_exposures: npt.ArrayLike | None = None,
        sample_weight: npt.ArrayLike | None = None,
        **fit_params,
    ):
        """Fit the Cross-Sectional Factor Model.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Asset returns.

        y : array-like, optional
            Ignored, present for API consistency.

        sector_exposures : array-like of shape (n_observations, n_assets, n_sectors)
            Sector factor exposures. Typically binary (one-hot encoded).
            Use N-1 sectors if you want sum-to-zero constraint.

        style_exposures : array-like of shape (n_observations, n_assets, n_styles)
            Style factor exposures. Typically continuous values.

        sample_weight : array-like of shape (n_observations, n_assets), optional
            Sample weights for cross-sectional regression (e.g., market caps).

        **fit_params : dict
            Parameters to pass to nested estimators via metadata routing.

        Returns
        -------
        self : CrossSectionalFactorModel
            Fitted estimator.
        """
        if sector_exposures is None:
            raise ValueError("sector_exposures parameter is required")
        if style_exposures is None:
            raise ValueError("style_exposures parameter is required")

        routed_params = skm.process_routing(self, "fit", **fit_params)

        # Initialize estimators
        self.cross_sectional_regressor_ = check_estimator(
            self.cross_sectional_regressor,
            default=CrossSectionalRegression(
                sqrt_weights=self.sqrt_weights,
                include_market=self.include_market,
            ),
            check_type=CrossSectionalRegression,
        )

        self.factor_prior_estimator_ = check_estimator(
            self.factor_prior_estimator,
            default=EmpiricalPrior(),
            check_type=BasePrior,
        )

        # Validate inputs
        sector_exposures = np.asarray(sector_exposures, dtype=float)
        style_exposures = np.asarray(style_exposures, dtype=float)

        if sector_exposures.ndim != 3:
            raise ValueError(
                f"sector_exposures must be 3D array, got shape {sector_exposures.shape}"
            )
        if style_exposures.ndim != 3:
            raise ValueError(
                f"style_exposures must be 3D array, got shape {style_exposures.shape}"
            )

        # Validate and convert returns
        X = skv.validate_data(self, X, dtype=float)
        n_observations, n_assets = X.shape

        if sector_exposures.shape[:2] != (n_observations, n_assets):
            raise ValueError(
                f"sector_exposures shape {sector_exposures.shape} doesn't match "
                f"X shape {X.shape} in first two dimensions"
            )
        if style_exposures.shape[:2] != (n_observations, n_assets):
            raise ValueError(
                f"style_exposures shape {style_exposures.shape} doesn't match "
                f"X shape {X.shape} in first two dimensions"
            )

        # Step 1: Run cross-sectional regression to estimate factor returns
        self.cross_sectional_regressor_.fit(
            X=X,
            sector_exposures=sector_exposures,
            style_exposures=style_exposures,
            sample_weight=sample_weight,
            **routed_params.cross_sectional_regressor.fit,
        )
        self.factor_returns_ = self.cross_sectional_regressor_.factor_returns_

        # Step 2: Estimate factor return distribution
        valid_rows = ~np.any(np.isnan(self.factor_returns_), axis=1)
        factor_returns_valid = self.factor_returns_[valid_rows]

        if factor_returns_valid.shape[0] == 0:
            raise ValueError(
                "All factor returns are NaN. Check that exposures and returns "
                "have valid (non-NaN) data."
            )

        self.factor_prior_estimator_.fit(
            X=factor_returns_valid, **routed_params.factor_prior_estimator.fit
        )
        factor_dist = self.factor_prior_estimator_.return_distribution_

        # Step 3: Construct asset return distribution from factor model
        # Combine exposures for last observation
        n_sectors = sector_exposures.shape[2]
        if self.cross_sectional_regressor_.include_market:
            market_factor = np.ones((n_assets, 1))
            last_sector_exp = np.hstack([market_factor, sector_exposures[-1]])
        else:
            last_sector_exp = sector_exposures[-1]
        last_style_exp = style_exposures[-1]
        last_exposures = np.hstack([last_sector_exp, last_style_exp])

        # Asset expected returns: mu = exposures @ factor_mu
        mu = last_exposures @ factor_dist.mu

        # Asset covariance: Sigma = B @ Sigma_f @ B.T
        covariance = last_exposures @ factor_dist.covariance @ last_exposures.T

        # Reconstruct returns from factor model
        returns = factor_returns_valid @ last_exposures.T

        # Cholesky factor (low-rank representation)
        cholesky = last_exposures @ np.linalg.cholesky(factor_dist.covariance)

        # Step 4: Add residual variance (specific risk) if requested
        if self.residual_variance:
            residuals = self.cross_sectional_regressor_.residuals_
            residuals_valid = residuals[valid_rows]
            residual_var = np.diag(sm.variance(residuals_valid))
            covariance += residual_var
            cholesky = np.hstack((cholesky, np.sqrt(residual_var)))

        # Ensure covariance is positive semi-definite
        covariance = cov_nearest(
            covariance, higham=self.higham, higham_max_iteration=self.max_iteration
        )

        # Create return distribution
        self.return_distribution_ = ReturnDistribution(
            mu=mu,
            covariance=covariance,
            returns=returns,
            cholesky=cholesky,
            sample_weight=factor_dist.sample_weight,
        )

        return self
