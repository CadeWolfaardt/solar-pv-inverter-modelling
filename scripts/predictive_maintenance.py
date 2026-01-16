# projectlib
from pv_inverter_modeling.config.env import DATA_ROOT, PDM_SAMPLE_DEVICE
from pv_inverter_modeling.models.predictive_maintenance import (
    PredictiveMaintenance
)

def main() -> None:
    """
    Execute the end-to-end predictive maintenance workflow.

    This entry point orchestrates the full modeling pipeline for
    inverter-level power generation forecasting and monitoring. It:

    - Initializes the predictive maintenance system with configuration
      and logging settings
    - Loads and preprocesses historical inverter time-series data
    - Trains a sequence-to-sequence LSTM forecasting model
    - Generates and visualizes a sample multi-day forecast
    - Monitors recent predicted power generation against a configurable
      performance threshold

    The workflow is intended for exploratory analysis, model validation,
    and qualitative monitoring rather than automated deployment.

    Notes
    -----
    - The inverter identifier is provided via ``PDM_SAMPLE_DEVICE``.
    - Forecasts are generated at 5-minute resolution.
    - Monitoring thresholds may be interpreted as percentages or
      absolute values depending on configuration.
    """
    path = DATA_ROOT
    filename = "pdm_data.parquet"
    pm = PredictiveMaintenance(
        path=path,
        file=filename,
        inverter=PDM_SAMPLE_DEVICE,
        verbosity=0,
        write_log=False
    )
    _model, _history = pm.train_model()
    _prediction = pm.testout(index=1000)
    pm.monitor_power(
        numbers_of_days=30, 
        threshold=13.0, 
        threshold_mode="pct"
    )

if __name__=="__main__":
    main()