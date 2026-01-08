# stdlib
from typing import List
# thirdpartylib
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
# projectlib
from pv_inverter_modeling.visualization.timeseries import use_dark_theme

class KaplanMeierModel(object):
    """
    Kaplan-Meier survival analysis model for detecting productivity
    degradation events in daily time-series data.

    This class converts daily aggregated productivity sequences into a
    binary failure process based on a relative threshold, computes
    time-to-failure durations, and fits a Kaplan-Meier survival model
    to estimate the probability of remaining above the productivity
    threshold over time.
    """
    def __init__(self, data: np.ndarray, water_mark: float = 30.0) -> None:
        """
        Initialize the Kaplan-Meier model from daily productivity data.

        Parameters
        ----------
        data : numpy.ndarray
            Array of shape (num_days, num_timesteps) containing daily
            productivity values (e.g., power measurements per day).
            Values are expected to be non-negative.
        water_mark : float, default 30.0
            Percentage of the median daily productivity used as the
            failure threshold. Days with total productivity below
            (water_mark / 100) x median are treated as failure events.

        Notes
        -----
        - Daily productivity is computed as the sum across timesteps.
        - A binary event sequence is created where 1 indicates failure
          and 0 indicates normal operation.
        - Durations and censoring indicators are derived automatically.
        """
        self.kmf = None
        # Percentage threshold for defining a failure event
        self.water_mark = water_mark
        # Aggregate daily productivity
        daily_productivity = data.sum(axis=1)
        # Define failure threshold as a fraction of the median
        median_productivity = np.median(daily_productivity)
        threshold = (water_mark / 100) * median_productivity
        # Binary event indicator: 1 = failure, 0 = normal
        self.events = (daily_productivity < threshold).astype(int)
        # Time-to-event (days) and censoring flags
        self.durations: List[int] = []
        self.observed: List[int] = []
        # Compute durations and event observations
        self.duration_to_failure()

    def duration_to_failure(self) -> None:
        """
        Compute time-to-failure durations and censoring indicators.

        For each day index, this method determines how many days elapse
        until the first failure event occurs. If no failure is observed
        after a given day, the observation is treated as right-censored.

        Populates
        ----------
        self.durations : list[int]
            Number of days until failure (or censoring).
        self.observed : list[int]
            Event indicator where 1 denotes an observed failure and
            0 denotes right-censoring.
        """
        for i in range(len(self.events)):
            # Look ahead for the first failure after day i
            failure_days = np.where(self.events[i:] == 1)[0]

            if len(failure_days) > 0:
                self.durations.append(failure_days[0])
                self.observed.append(1)
            else:
                self.durations.append(len(self.events) - i)
                self.observed.append(0)
    
    def kaplan_meier_analysis(self) -> None:
        """
        Fit and plot the Kaplan-Meier survival function.

        This method fits a Kaplan-Meier estimator using the computed
        durations and event indicators, then visualizes the survival
        probability over time using the configured plotting theme.
        """
        self.kmf = KaplanMeierFitter()
        self.kmf.fit(  # pyright: ignore[reportUnknownMemberType]
            self.durations,
            event_observed=self.observed
        )
        # Plot survival curve
        use_dark_theme()
        self.kmf.plot_survival_function()  # pyright: ignore
        plt.title(  # pyright: ignore[reportUnknownMemberType]
            f"Survival: Days Until < {self.water_mark}% Productivity"
        )
        plt.xlabel("Days")  # pyright: ignore[reportUnknownMemberType]
        plt.ylabel( # pyright: ignore[reportUnknownMemberType]
            "Survival Probability"
        )
        plt.show() # pyright: ignore[reportUnknownMemberType]
    
    def early_warning(self, threshold: float = 0.10) -> None:
        """
        Run survival analysis and report an early-warning horizon.

        This method fits the Kaplan-Meier model (if not already fit)
        and identifies the earliest time at which the survival
        probability drops below ``threshold``, providing a conservative 
        estimate of impending productivity degradation.

        Raises
        ------
        ValueError
            If the Kaplan-Meier model has not been successfully fit.
        """
        self.kaplan_meier_analysis()
        if self.kmf is None:
            raise ValueError("Kaplan-Meier model has not been fit.")
        surv_func = self.kmf.survival_function_
        # Time when survival probability falls below threshold
        time_at_threshold = (
            surv_func[surv_func["KM_estimate"] <= threshold]
            .index
            .min()
        )
        print(
            f"Time when survival drops below {threshold*100:.2f}%: "
            f"{time_at_threshold} days"
        )