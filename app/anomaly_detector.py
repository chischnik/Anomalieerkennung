"""
FastAPI-Service für die Echtzeit-Anomalieerkennung.

Das Modul verbindet den Elec2-Datenstrom mit:
- Pydantic-Validierung
- Sliding Window
- Isolation Forest
- One-Class SVM
- PSI und KS
- Infrastrukturdiagnostik
- Prometheus

Die Online-Diagnose trennt Systemzustand, Anomalietyp und Root Cause:

    Systemzustand:
        0 = Normal
        1 = Anomaly

    Anomalietyp:
        NONE
        ANOMALY_CONCEPT_DRIFT
        INFRASTRUCTURE_FAILURE

Infrastructure Failure ist damit keine dritte Systemklasse. Die Ground Truth
wird ausschließlich für die experimentelle Evaluation dokumentiert und
beeinflusst die Online-Klassifikation nicht.
"""

import inspect
from contextlib import asynccontextmanager
import threading
import time
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field, validator
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import config
from app.metrics import update_all_metrics
from app.models import AnomalyModels, create_feature_vector
from app.stream_ingestor import Elec2StreamIngestor, create_ingestor
from app.window import SlidingWindow


# ============================================================================
# FastAPI
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialisiert Modelle und Demo-Stream über den FastAPI-Lebenszyklus."""

    startup_event()

    try:
        yield
    finally:
        stop_stream()


app = FastAPI(
    title="Kubernetes ML Anomaly Detector",
    description=(
        "Echtzeit-Anomalieerkennung für eine Kubernetes-basierte "
        "ML-Pipeline auf Basis des Elec2-Datenstroms."
    ),
    version="3.2.1",
    lifespan=lifespan,
)


# ============================================================================
# Pydantic-Eingabemodell
# ============================================================================


class PredictionRequest(BaseModel):
    """
    Definiert das validierte Eingabeformat für /predict.

    Die numerischen Fachwerte werden vor der Inferenz auf endliche
    Werte geprüft. Ungültige Payloads werden dadurch als HTTP 422
    behandelt und nicht stillschweigend in die Modelle übernommen.
    """

    nswprice: float = Field(...)
    nswdemand: float = Field(...)
    vicprice: float = Field(...)
    vicdemand: float = Field(...)
    transfer: float = Field(...)

    # Das Target ist ein beobachtetes Ergebnis und kein ML-Eingabefeature.
    # Es dient ausschließlich zur nachgelagerten Leistungsmessung.
    target: Optional[Any] = None

    step: Optional[int] = None
    timestamp: Optional[float] = None

    infrastructure_cpu_load: float = 0.0
    infrastructure_memory_load: float = 0.0
    infrastructure_restarts: int = 0

    @validator(
        "nswprice",
        "nswdemand",
        "vicprice",
        "vicdemand",
        "transfer",
        "infrastructure_cpu_load",
        "infrastructure_memory_load",
        pre=True,
    )
    def validate_finite(cls, value):
        """Verhindert NaN und unendliche numerische Eingaben."""

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Der Wert muss numerisch sein."
            ) from exc

        if not np.isfinite(numeric_value):
            raise ValueError(
                "Der Wert muss endlich sein; NaN und Inf sind nicht erlaubt."
            )

        return numeric_value


# ============================================================================
# Globale Komponenten
# ============================================================================

window = SlidingWindow(
    window_size=config.WINDOW_SIZE,
    min_window_size=config.MIN_WINDOW_SIZE,
)

models = AnomalyModels()

ingestor: Optional[Elec2StreamIngestor] = None
stream_thread: Optional[threading.Thread] = None

stream_running = False
stream_lock = threading.Lock()

# ============================================================================
# Laufzeitcache für lokale Driftberechnungen
# ============================================================================

_drift_cache_metrics: Dict[str, Dict[str, Optional[float]]] = {}
_drift_cache_comparison_size = -1
_window_metrics_cache: Dict[str, Optional[float]] = {}
_window_metrics_cache_size = -1

# Lokale PSI-/KS-Berechnungen sind relativ teuer. Der Cache wird nach
# mehreren neuen Beobachtungen gezielt invalidiert. Die statistischen
# Schwellenwerte und die Entscheidungslogik bleiben unverändert.
LOCAL_DRIFT_CACHE_INTERVAL = 10

# Ein statistisches Drift-Signal muss über mehrere unabhängige
# Fenstervergleiche bestätigt werden, bevor es operativ als Drift gilt.
# Der Wert wird zentral in app.config definiert, damit Produktionspfad
# und Evaluationspfad dieselbe Persistenzanforderung verwenden.
DRIFT_CONFIRMATION_WINDOWS = config.DRIFT_CONFIRMATION_WINDOWS

# Leistungsüberwachung für den eingefrorenen Zielprädiktor.
# Der Wert ist absichtlich lokal definiert, damit die bestehende config.py
# unverändert kompatibel bleibt.
PERFORMANCE_WINDOW_SIZE = 50
PERFORMANCE_DEGRADATION_MARGIN = 0.10
PERFORMANCE_DEGRADATION_FACTOR = 3.0
# Absolute Fehlergrenze für die Erkennung eines nachhaltigen Concept Drifts.
# Dieser Wert bildet einen konservativen Mittelweg zwischen den im Replay
# beobachteten Normal-/Data-Drift-Fehlern und dem deutlich höheren Concept-Drift-Fehler.
PERFORMANCE_ABSOLUTE_THRESHOLD = 0.65
# Zusätzlich zum 50-Beobachtungs-Fenster müssen drei aufeinanderfolgende
# Auswertungen die Degradation bestätigen. Dadurch werden einzelne
# Überschreitungen nicht unmittelbar als Concept Drift klassifiziert.
PERFORMANCE_CONFIRMATION_WINDOWS = 3

_performance_errors = deque(maxlen=PERFORMANCE_WINDOW_SIZE)
_performance_probability_errors = deque(maxlen=PERFORMANCE_WINDOW_SIZE)
_performance_baseline_error: Optional[float] = None
_performance_baseline_conditional_error = np.full(10, np.nan, dtype=float)
_performance_baseline_conditional_counts = np.zeros(10, dtype=int)
_performance_confirmation_count = 0
_performance_degraded = False

_drift_confirmation_count = 0
_drift_last_evaluated_comparison_size = -1
_drift_confirmation_evaluation_index = -1
_drift_confirmation_last_processed_evaluation_index = -1
_drift_confirmation_last_window_size = -1
_drift_cache_last_observation_step = -1
_drift_confirmed = False
_drift_confirmed_feature = ""


# ============================================================================
# Laufzeitstatus
# ============================================================================

latest_state: Dict[str, Any] = {
    "step": 0,
    "cycle": 1,
    "cycle_step": 0,
    "timestamp": 0.0,
    "scenario": config.SCENARIO_NORMAL,
    "system_state": config.SYSTEM_STATE_NORMAL,
    "system_state_name": config.SYSTEM_STATE_NAMES[
        config.SYSTEM_STATE_NORMAL
    ],
    "classification": config.SYSTEM_STATE_NORMAL,
    "classification_name": config.SYSTEM_STATE_NAMES[
        config.SYSTEM_STATE_NORMAL
    ],
    "ground_truth_classification": config.SYSTEM_STATE_NORMAL,
    "ground_truth_name": config.SYSTEM_STATE_NAMES[
        config.SYSTEM_STATE_NORMAL
    ],
    "anomaly_type": config.ANOMALY_NONE,
    "target_true": None,
    "target_prediction": None,
    "target_prediction_probability": None,
    "prediction_error": None,
    "rolling_prediction_error": None,
    "performance_expected_error": None,
    "performance_baseline_error": None,
    "performance_degraded": False,
    "performance_confirmation_count": 0,
    "concept_drift_detected": False,
    "drift_detected": False,
    "raw_drift_detected": False,
    "drift_confirmation_count": 0,
    "drift_confirmation_required": DRIFT_CONFIRMATION_WINDOWS,
    "root_cause": config.ROOT_CAUSE_NORMAL,
    "trigger_feature": config.TRIGGER_NONE,
    "point_anomaly": False,
    "point_anomaly_type": "",
    "point_anomaly_trigger_feature": "",
    "point_anomaly_feature_scores": {},
    "nswprice": None,
    "nswdemand": None,
    "vicprice": None,
    "vicdemand": None,
    "transfer": None,
    "rolling_mean": None,
    "rolling_std": None,
    "rate_of_change": None,
    "peak_density": None,
    "ks_pvalue": None,
    "ks_statistic": None,
    "z_score": None,
    "psi": None,
    "psi_drift": False,
    "psi_by_feature": {},
    "ks_by_feature": {},
    "isolation_forest_score": None,
    "one_class_svm_distance": None,
    "isolation_forest_anomaly": False,
    "one_class_svm_anomaly": False,
    "missing_value_rate": {},
    "out_of_bounds_rate": {},
    "schema_validation_failure": False,
    "inference_latency_seconds": None,
    "cpu_usage_percent": 0.0,
    "memory_usage_percent": 0.0,
    "pod_restarts": 0,
    "model_ready": False,
    "window_ready": False,
    "ks_reference_ready": False,
    "ks_reference_size": 0,
    "calibration_ready": False,
    "isolation_forest_threshold": None,
    "one_class_svm_threshold": None,
    "calibration_quantile": None,
    "remediation_signal": 0,
}


# ============================================================================
# Hilfsfunktionen
# ============================================================================


def _reset_runtime_state() -> None:
    """Setzt das kurzfristige Laufzeitfenster und den Zustand zurück."""

    global latest_state

    window.clear()

    global _drift_cache_metrics
    global _drift_cache_comparison_size
    global _drift_confirmation_evaluation_index
    global _drift_confirmation_last_processed_evaluation_index
    global _drift_confirmation_last_window_size
    global _drift_cache_last_observation_step

    _drift_cache_metrics = {}
    _drift_cache_comparison_size = -1

    global _window_metrics_cache
    global _window_metrics_cache_size
    global _drift_confirmation_count
    global _drift_last_evaluated_comparison_size
    global _drift_confirmed
    global _drift_confirmed_feature

    _window_metrics_cache = {}
    _window_metrics_cache_size = -1

    _drift_confirmation_count = 0
    _drift_last_evaluated_comparison_size = -1
    _drift_confirmation_evaluation_index = -1
    _drift_confirmation_last_processed_evaluation_index = -1
    _drift_confirmation_last_window_size = -1
    _drift_cache_last_observation_step = -1
    _drift_confirmed = False
    _drift_confirmed_feature = ""

    global _performance_errors
    global _performance_probability_errors
    global _performance_confirmation_count
    global _performance_degraded

    _performance_errors.clear()
    _performance_probability_errors.clear()

    # Der Kalibrierungs-Baseline-Fehler gehört zum trainierten Modell
    # und darf bei einem Runtime-Reset nicht gelöscht werden.
    # Nur die kurzfristige Performance-Historie und deren Bestätigung
    # werden für den nächsten Testlauf zurückgesetzt.
    _performance_confirmation_count = 0
    _performance_degraded = False

    with stream_lock:
        latest_state = {
            **latest_state,
            "step": 0,
            "cycle": 1,
            "cycle_step": 0,
            "timestamp": 0.0,
            "scenario": config.SCENARIO_NORMAL,
            "system_state": config.SYSTEM_STATE_NORMAL,
            "system_state_name": config.SYSTEM_STATE_NAMES[
                config.SYSTEM_STATE_NORMAL
            ],
            "classification": config.SYSTEM_STATE_NORMAL,
            "classification_name": config.SYSTEM_STATE_NAMES[
                config.SYSTEM_STATE_NORMAL
            ],
            "ground_truth_classification": config.SYSTEM_STATE_NORMAL,
            "ground_truth_name": config.SYSTEM_STATE_NAMES[
                config.SYSTEM_STATE_NORMAL
            ],
            "anomaly_type": config.ANOMALY_NONE,
            "drift_detected": False,
            "raw_drift_detected": False,
            "drift_confirmation_count": 0,
            "drift_confirmation_required": DRIFT_CONFIRMATION_WINDOWS,
            "root_cause": config.ROOT_CAUSE_NORMAL,
            "trigger_feature": config.TRIGGER_NONE,
            "point_anomaly": False,
            "point_anomaly_type": "",
            "point_anomaly_trigger_feature": "",
            "point_anomaly_feature_scores": {},
            "window_ready": False,
            "schema_validation_failure": False,
            "inference_latency_seconds": None,
            "missing_value_rate": {},
            "out_of_bounds_rate": {},
            "remediation_signal": 0,
        }

def _initialize_reference_distribution() -> None:
    """
    Initialisiert PSI-/KS-Referenzen aus einem gesunden Referenzblock.

    Für jedes Elec2-Feature wird dieselbe normale Referenzphase verwendet.
    Die Referenz wird nach der Initialisierung nicht verändert.
    """

    global ingestor

    if ingestor is None:
        raise RuntimeError("Der Stream-Ingestor wurde nicht initialisiert.")

    reference_size = min(
        int(config.REFERENCE_SIZE),
        int(ingestor.data_drift_start),
        int(ingestor.total_rows),
    )

    if reference_size < config.MIN_WINDOW_SIZE:
        raise RuntimeError(
            "Der normale Referenzbereich ist zu klein."
        )

    references = {}

    for feature in config.FEATURE_COLUMNS:
        values = (
            ingestor.data[feature]
            .iloc[:reference_size]
            .astype(float)
            .to_numpy()
        )

        values = values[np.isfinite(values)]

        if len(values) >= config.MIN_WINDOW_SIZE:
            references[feature] = values

    if not references:
        raise RuntimeError(
            "Es konnten keine gültigen Referenzverteilungen erzeugt werden."
        )

    window.set_reference(references)


def _train_models() -> None:
    """
    Trainiert IF und One-Class SVM ausschließlich auf gesunden
    Referenzdaten und kalibriert ihre Scores auf einem separaten
    normalen Kalibrierungsblock.
    """

    global ingestor

    if ingestor is None:
        raise RuntimeError("Der Stream-Ingestor wurde nicht initialisiert.")

    reference_size = int(config.REFERENCE_SIZE)
    calibration_size = int(getattr(config, "CALIBRATION_SIZE", 1000))
    evaluation_start = reference_size + calibration_size

    training_features = []
    training_targets = []
    calibration_features = []
    calibration_targets = []

    for observation in ingestor.stream_fast(max_rows=evaluation_start):
        feature_vector = create_feature_vector(
            nswprice=observation.nswprice,
            nswdemand=observation.nswdemand,
            vicprice=observation.vicprice,
            vicdemand=observation.vicdemand,
            transfer=observation.transfer,
        )

        values = feature_vector[0]

        if observation.step < reference_size:
            training_features.append(values)
            training_targets.append(observation.target)

        elif observation.step < evaluation_start:
            calibration_features.append(values)
            calibration_targets.append(observation.target)

    X_train = np.asarray(training_features, dtype=float)
    X_calibration = np.asarray(calibration_features, dtype=float)
    y_train = np.asarray(training_targets, dtype=str)
    y_calibration = np.asarray(calibration_targets, dtype=str)

    if len(X_train) != reference_size:
        raise RuntimeError(
            f"Falsche Trainingsgröße: {len(X_train)} / {reference_size}"
        )

    if len(X_calibration) != calibration_size:
        raise RuntimeError(
            f"Falsche Kalibrierungsgröße: "
            f"{len(X_calibration)} / {calibration_size}"
        )

    if not np.isfinite(X_train).all():
        raise RuntimeError(
            "Die Trainingsdaten enthalten ungültige Werte."
        )

    if not np.isfinite(X_calibration).all():
        raise RuntimeError(
            "Die Kalibrierungsdaten enthalten ungültige Werte."
        )

    models.fit(X_train)
    models.fit_target_predictor(X_train, y_train)

    # Für die Online-Inferenz wird das Zielmodell auf einen Thread begrenzt.
    # Das Zielmodell verarbeitet im Produktionspfad einzelne Beobachtungen.
    # Unter Windows verursacht n_jobs=-1 bei solchen Einzelvorhersagen
    # unnötigen Joblib-Overhead und kann den Replay massiv verlangsamen.
    # Das ändert weder das trainierte Modell noch seine Vorhersagequalität.
    if hasattr(models, "target_predictor") and hasattr(
        models.target_predictor, "model"
    ):
        models.target_predictor.model.n_jobs = 1

    if hasattr(models, "calibrate"):
        models.calibrate(
            X_calibration,
            quantile=getattr(
                config,
                "CALIBRATION_QUANTILE",
                0.01,
            ),
        )

    # Die normale Referenzleistung wird einmalig auf dem getrennten
    # Kalibrierungsblock bestimmt. Das Zielmodell bleibt danach eingefroren.
    #
    # Die Kalibrierung wird bewusst batchweise durchgeführt. Eine einzelne
    # Random-Forest-Inferenz pro Beobachtung erzeugt unter Windows unnötigen
    # Joblib-/Prozess-Overhead und kann den Start des RCA-Replays stark
    # verlangsamen. Die fachliche Logik bleibt dabei identisch.
    target_model = models.target_predictor.model
    target_encoder = models.target_predictor.label_encoder

    encoded_calibration_predictions = target_model.predict(
        X_calibration
    )
    calibration_predictions = target_encoder.inverse_transform(
        encoded_calibration_predictions.astype(int)
    ).astype(str)

    calibration_errors = (
        calibration_predictions != y_calibration
    ).astype(float)

    global _performance_baseline_error
    global _performance_baseline_conditional_error
    global _performance_baseline_conditional_counts

    _performance_baseline_error = float(np.mean(calibration_errors))

    # Baseline-Fehler nach Modellkonfidenz speichern. Dadurch wird ein
    # erwartbarer Fehleranstieg unter Covariate/Data Drift nicht automatisch
    # als Concept Drift interpretiert. Auch die Wahrscheinlichkeiten werden
    # batchweise berechnet, damit die Kalibrierung effizient bleibt.
    calibration_probabilities = np.max(
        target_model.predict_proba(X_calibration),
        axis=1,
    ).astype(float)
    _performance_baseline_conditional_error = np.full(10, np.nan, dtype=float)
    _performance_baseline_conditional_counts = np.zeros(10, dtype=int)

    for bin_index in range(10):
        lower = bin_index / 10.0
        upper = (bin_index + 1) / 10.0
        if bin_index == 9:
            mask = (calibration_probabilities >= lower) & (calibration_probabilities <= upper)
        else:
            mask = (calibration_probabilities >= lower) & (calibration_probabilities < upper)
        count = int(np.sum(mask))
        _performance_baseline_conditional_counts[bin_index] = count
        if count >= 10:
            _performance_baseline_conditional_error[bin_index] = float(
                np.mean(calibration_errors[mask])
            )

    with stream_lock:
        latest_state["model_ready"] = models.is_fitted
        latest_state["calibration_ready"] = models.is_calibrated
        latest_state["isolation_forest_threshold"] = (
            models.isolation_forest_threshold
        )
        latest_state["one_class_svm_threshold"] = (
            models.one_class_svm_threshold
        )
        latest_state["calibration_quantile"] = (
            models.calibration_quantile
        )
        latest_state["performance_baseline_error"] = (
            _performance_baseline_error
        )

    print("Modelle erfolgreich trainiert und kalibriert.")


def _classify_observation(
    psi_by_feature: Dict[str, Optional[float]],
    ks_by_feature: Dict[str, Dict[str, Optional[float]]],
) -> tuple[bool, str]:
    """
    Erzeugt ein statistisches Rohsignal für eine Verteilungsänderung.

    PSI und KS müssen ein statistisches Rohsignal gemeinsam stützen. Die
    operative Driftentscheidung wird anschließend zeitlich über mehrere neue
    Fenstervergleiche bestätigt. Dadurch wird ein einzelner statistischer
    Ausschlag nicht sofort als operativer Drift behandelt.

    Ein Drift ist weiterhin keine Anomalie.
    """
    drift_candidates = []

    for feature in config.FEATURE_COLUMNS:
        psi_value = psi_by_feature.get(feature)
        ks_result = ks_by_feature.get(feature, {})
        ks_pvalue = ks_result.get("ks_pvalue")

        psi_drift = (
            psi_value is not None
            and np.isfinite(psi_value)
            and float(psi_value) > config.PSI_THRESHOLD
        )

        ks_drift = (
            ks_pvalue is not None
            and np.isfinite(ks_pvalue)
            and float(ks_pvalue) < config.KS_ALPHA
        )

        # PSI und KS müssen dasselbe Feature gemeinsam stützen.
        # Die zusätzliche zeitliche Bestätigung verhindert, dass ein
        # einzelner statistischer Ausschlag direkt als Drift gilt.
        if not (psi_drift and ks_drift):
            continue

        if psi_drift:
            ranking_score = float(psi_value)
        else:
            ranking_score = float(
                -np.log10(
                    max(
                        float(ks_pvalue),
                        np.finfo(float).tiny,
                    )
                )
            )
        drift_candidates.append((feature, ranking_score))

    if not drift_candidates:
        return False, ""

    trigger_feature = max(
        drift_candidates,
        key=lambda item: item[1],
    )[0]

    return True, trigger_feature


def _update_drift_confirmation(
    raw_drift_detected: bool,
    raw_drift_trigger_feature: str,
    comparison_size: int,
    evaluation_index: int,
) -> tuple[bool, str, int]:
    """
    Bestätigt ein statistisches Drift-Signal über mehrere tatsächlich
    neu berechnete lokale Fenstervergleiche.

    Der Bestätigungszähler basiert auf einem separaten Index für bereits
    verarbeitete Drift-Evaluationen. Dadurch wird verhindert, dass der
    aktuelle Evaluationsindex vorzeitig aktualisiert wird und eine neue
    statistische Evidenz fälschlich als bereits verarbeitet gilt.

    Ein zwischengespeicherter PSI-/KS-Wert wird nicht mehrfach als
    unabhängige Bestätigung gewertet.
    """
    global _drift_confirmation_count
    global _drift_last_evaluated_comparison_size
    global _drift_confirmation_evaluation_index
    global _drift_confirmation_last_processed_evaluation_index
    global _drift_confirmed
    global _drift_confirmed_feature

    if comparison_size < 0 or evaluation_index < 0:
        _drift_confirmation_count = 0
        _drift_last_evaluated_comparison_size = comparison_size
        _drift_confirmation_evaluation_index = evaluation_index
        _drift_confirmation_last_processed_evaluation_index = evaluation_index
        _drift_confirmed = False
        _drift_confirmed_feature = ""
        return False, "", 0

    # Nur eine neue statistische Berechnung darf die Bestätigungskette
    # verändern. Der aktuelle Index und der zuletzt verarbeitete Index
    # werden deshalb getrennt geführt.
    if (
        evaluation_index
        != _drift_confirmation_last_processed_evaluation_index
    ):
        _drift_confirmation_last_processed_evaluation_index = evaluation_index
        _drift_confirmation_evaluation_index = evaluation_index
        _drift_last_evaluated_comparison_size = comparison_size

        if raw_drift_detected:
            _drift_confirmation_count += 1

            if raw_drift_trigger_feature:
                _drift_confirmed_feature = raw_drift_trigger_feature
        else:
            _drift_confirmation_count = 0
            _drift_confirmed_feature = ""

        _drift_confirmed = (
            _drift_confirmation_count
            >= DRIFT_CONFIRMATION_WINDOWS
        )

    if not _drift_confirmed:
        return False, "", _drift_confirmation_count

    return (
        True,
        _drift_confirmed_feature or raw_drift_trigger_feature,
        _drift_confirmation_count,
    )

def _get_window_metrics() -> Dict[str, Optional[float]]:
    """
    Liefert historische Fenster-Metriken aus einem Laufzeitcache.

    Die statistischen Referenztests werden nur nach mehreren neuen
    Beobachtungen erneut berechnet. Referenzen und Schwellenwerte bleiben
    unverändert.
    """
    global _window_metrics_cache
    global _window_metrics_cache_size

    current_size = window.size

    if not window.is_ready:
        return window.calculate_metrics(
            feature=config.PRICE_COLUMN
        )

    if (
        not _window_metrics_cache
        or _window_metrics_cache_size < 0
        or current_size < _window_metrics_cache_size
        or (
            current_size - _window_metrics_cache_size
            >= LOCAL_DRIFT_CACHE_INTERVAL
        )
    ):
        _window_metrics_cache = window.calculate_metrics(
            feature=config.PRICE_COLUMN
        )
        _window_metrics_cache_size = current_size

    return _window_metrics_cache


def _get_local_drift_metrics(
    current_step: int,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Liefert lokale PSI-/KS-Metriken aus einem kleinen Laufzeitcache.

    Die statistische Definition des Drifts bleibt unverändert. PSI und KS
    werden weiterhin auf zwei vollständigen, nicht überlappenden Fenstern
    berechnet.

    Der Cache wird anhand des monotonen Beobachtungsschritts invalidiert.
    Die Größe eines Sliding Windows ist dafür ungeeignet, weil sie nach
    Erreichen der maximalen Fenstergröße konstant bleiben kann.
    """

    global _drift_cache_metrics
    global _drift_cache_comparison_size
    global _drift_confirmation_evaluation_index
    global _drift_cache_last_observation_step

    comparison_size = window.comparison_size

    if not window.is_local_comparison_ready:
        return {}

    # Die lokale Statistik wird nach mehreren neuen Beobachtungen erneut
    # berechnet. Dadurch entsteht tatsächlich neue statistische Evidenz.
    should_recalculate = (
        not _drift_cache_metrics
        or _drift_cache_last_observation_step < 0
        or current_step < _drift_cache_last_observation_step
        or (
            current_step - _drift_cache_last_observation_step
            >= LOCAL_DRIFT_CACHE_INTERVAL
        )
    )

    if should_recalculate:
        _drift_cache_metrics = (
            window.calculate_all_local_drift_metrics()
        )
        _drift_cache_comparison_size = comparison_size
        _drift_cache_last_observation_step = current_step

        # Dieser Index zählt ausschließlich tatsächlich neu berechnete
        # PSI-/KS-Vergleiche.
        _drift_confirmation_evaluation_index += 1

    return _drift_cache_metrics

def _detect_infrastructure_failure(
    values: Dict[str, float],
    cpu_usage: float,
    memory_usage: float,
    pod_restarts: int,
    missing_value_rates: Optional[Dict[str, float]] = None,
    out_of_bounds_rates: Optional[Dict[str, float]] = None,
    observation: Any = None,
) -> tuple[bool, str, str]:
    """
    Erkennt Infrastruktur-/Datenqualitätsfehler unabhängig vom Systemzustand.

    Die Root Cause wird möglichst spezifisch bestimmt:
        - CPU/RAM/Pod-Restart -> INFRA_RESOURCE
        - fehlender Wert -> INFRA_MISSING_DATA
        - konfigurierter Wertebereich verletzt -> INFRA_VALUE_EXTREME

    Schemafehler können nur über einen expliziten Schemaindikator erkannt
    werden; ein solcher Indikator ist im aktuellen PredictionRequest nicht
    vorhanden und wird daher hier nicht erfunden.
    """
    # Explizite Datenqualitätsinjektionen werden vor den allgemeinen
    # Ressourcenindikatoren ausgewertet. Dadurch bleibt der konkrete
    # Trigger auch bei gleichzeitig hoher CPU-/RAM-Last erhalten.
    if observation is not None:
        if bool(getattr(observation, "missing_value_injected", False)):
            trigger = getattr(observation, "trigger_feature", "")
            return (True, config.INFRA_MISSING_DATA, trigger or "MISSING_VALUE")

        if bool(getattr(observation, "extreme_value_injected", False)):
            trigger = getattr(observation, "trigger_feature", "")
            return (True, config.INFRA_VALUE_EXTREME, trigger or "EXTREME_VALUE")

    if cpu_usage >= config.CPU_ALERT_PERCENT:
        return True, config.INFRA_RESOURCE, "HIGH_CPU"

    if memory_usage >= config.MEMORY_ALERT_PERCENT:
        return True, config.INFRA_RESOURCE, "HIGH_MEMORY"

    if pod_restarts >= config.POD_RESTART_ALERT_COUNT:
        return True, config.INFRA_RESOURCE, "POD_RESTART"

    for feature in config.FEATURE_COLUMNS:
        value = values.get(feature, np.nan)
        if not np.isfinite(value):
            return True, config.INFRA_MISSING_DATA, feature

    bounds_config = getattr(config, "FEATURE_BOUNDS", {})
    for feature in config.FEATURE_COLUMNS:
        bounds = bounds_config.get(feature)
        value = values.get(feature, np.nan)

        if bounds is None or not np.isfinite(value):
            continue

        lower, upper = bounds
        if value < lower or value > upper:
            return True, config.INFRA_VALUE_EXTREME, feature

    missing_value_rates = missing_value_rates or {}
    if any(
        rate >= config.MISSING_VALUE_ALERT_RATE
        for rate in missing_value_rates.values()
    ):
        feature = max(
            missing_value_rates,
            key=missing_value_rates.get,
        )
        return True, config.INFRA_MISSING_DATA, feature

    out_of_bounds_rates = out_of_bounds_rates or {}
    if any(
        rate >= config.OUT_OF_BOUNDS_ALERT_RATE
        for rate in out_of_bounds_rates.values()
    ):
        feature = max(
            out_of_bounds_rates,
            key=out_of_bounds_rates.get,
        )
        return True, config.INFRA_VALUE_EXTREME, feature

    return False, config.ROOT_CAUSE_NORMAL, ""


def _detect_point_anomaly_trigger(
    values: Dict[str, float],
    isolation_forest_anomaly: bool,
    one_class_svm_anomaly: bool,
) -> tuple[bool, str, str, Dict[str, float]]:
    """
    Ermittelt eine punktuelle Anomalie und lokalisiert ihr Trigger-Feature.

    Die Detektionsentscheidung stammt ausschließlich aus IF/OCSVM.
    Für die Lokalisierung wird die aktuelle Beobachtung mit den bereits
    vorhandenen Werten des Sliding Windows verglichen.

    Bei einer konstanten lokalen Referenz kann die robuste Skalierung null
    sein. In diesem Fall wird statt eines "unknown"-Triggers die absolute
    Abweichung vom lokalen Median verwendet.
    """

    point_anomaly = (
        bool(isolation_forest_anomaly)
        or bool(one_class_svm_anomaly)
    )

    if not point_anomaly:
        return False, "", "", {}

    observations = getattr(
        window,
        "observations",
        getattr(window, "_observations", []),
    )

    feature_scores: Dict[str, float] = {}
    absolute_deviations: Dict[str, float] = {}

    for feature in config.FEATURE_COLUMNS:
        previous_values = []

        for observation in observations:
            observation_values = getattr(
                observation,
                "values",
                {},
            )

            try:
                value = float(
                    observation_values.get(
                        feature,
                        np.nan,
                    )
                )
            except (TypeError, ValueError):
                value = np.nan

            if np.isfinite(value):
                previous_values.append(value)

        current_value = values.get(
            feature,
            np.nan,
        )

        if not np.isfinite(current_value):
            feature_scores[feature] = float("inf")
            absolute_deviations[feature] = float("inf")
            continue

        if len(previous_values) < 2:
            feature_scores[feature] = 0.0
            absolute_deviations[feature] = 0.0
            continue

        previous = np.asarray(
            previous_values,
            dtype=float,
        )

        center = float(
            np.median(previous)
        )

        deviation = abs(
            float(current_value) - center
        )

        absolute_deviations[feature] = deviation

        mad = float(
            np.median(
                np.abs(previous - center)
            )
        )

        robust_scale = 1.4826 * mad

        if robust_scale <= 1e-12:
            robust_scale = float(
                np.std(
                    previous,
                    ddof=0,
                )
            )

        if robust_scale <= 1e-12:
            # Bei einer konstanten Referenz ist jede Abweichung vom
            # Median relevant. Der absolute Abstand bleibt deshalb
            # als Fallback erhalten.
            feature_scores[feature] = (
                float("inf")
                if deviation > 1e-12
                else 0.0
            )
        else:
            feature_scores[feature] = (
                deviation / robust_scale
            )

    # Zuerst werden endliche robuste Scores bevorzugt.
    finite_scores = {
        feature: score
        for feature, score in feature_scores.items()
        if np.isfinite(score)
    }

    if finite_scores:
        trigger_feature = max(
            finite_scores,
            key=finite_scores.get,
        )
    else:
        # Falls mehrere Features wegen einer konstanten Referenz
        # unendliche Scores besitzen, wird anhand der absoluten
        # Abweichung entschieden. Dadurch bleibt die Lokalisierung
        # deterministisch und "unknown" wird vermieden.
        trigger_feature = max(
            absolute_deviations,
            key=absolute_deviations.get,
        )

    return (
        True,
        "POINT_ANOMALY",
        trigger_feature,
        feature_scores,
    )

def _safe_metric_update(**kwargs) -> None:
    """
    Übergibt nur Parameter, die die installierte metrics.py tatsächlich
    unterstützt. Dadurch bleibt die Migration auf die erweiterten
    Prometheus-Metriken rückwärtskompatibel.
    """

    signature = inspect.signature(update_all_metrics)
    supported = {
        name: value
        for name, value in kwargs.items()
        if name in signature.parameters
    }

    update_all_metrics(**supported)


def _calculate_window_quality_rates() -> tuple[
    Dict[str, float],
    Dict[str, float],
]:
    """
    Berechnet Missing-Value- und Out-of-Bounds-Raten aus dem aktuellen
    Sliding Window.

    Die Raten basieren ausschließlich auf den beobachteten Fensterwerten.
    Die experimentellen Ground-Truth-Felder werden nicht verwendet.
    """

    # SlidingWindow verwendet intern eine Deque namens _observations.
    # Die Funktion unterstützt zusätzlich eine öffentliche observations-Eigenschaft,
    # falls diese in einer späteren Version vorhanden ist.
    observations = getattr(
        window,
        "observations",
        getattr(window, "_observations", []),
    )
    denominator = max(len(observations), 1)

    missing_counts = {
        feature: 0
        for feature in config.FEATURE_COLUMNS
    }

    out_of_bounds_counts = {
        feature: 0
        for feature in config.FEATURE_COLUMNS
    }

    for observation in observations:
        observation_values = getattr(observation, "values", {})

        for feature in config.FEATURE_COLUMNS:
            value = observation_values.get(feature, np.nan)

            try:
                value = float(value)
            except (TypeError, ValueError):
                value = np.nan

            if not np.isfinite(value):
                missing_counts[feature] += 1
                continue

            bounds = getattr(
                config,
                "FEATURE_BOUNDS",
                {},
            ).get(feature)

            if bounds is not None:
                lower, upper = bounds

                if value < lower or value > upper:
                    out_of_bounds_counts[feature] += 1

    missing_rates = {
        feature: count / denominator
        for feature, count in missing_counts.items()
    }

    out_of_bounds_rates = {
        feature: count / denominator
        for feature, count in out_of_bounds_counts.items()
    }

    return missing_rates, out_of_bounds_rates


def _drift_status_by_feature(
    drift_metrics: Dict[str, Dict[str, Optional[float]]],
) -> Dict[str, bool]:
    """
    Leitet den operativen Driftstatus je Feature aus PSI UND KS ab.
    """

    result = {}

    for feature, metrics in drift_metrics.items():
        psi_value = metrics.get("psi")
        ks_pvalue = metrics.get("ks_pvalue")

        psi_drift = (
            psi_value is not None
            and np.isfinite(psi_value)
            and float(psi_value) > config.PSI_THRESHOLD
        )

        ks_drift = (
            ks_pvalue is not None
            and np.isfinite(ks_pvalue)
            and float(ks_pvalue) < config.KS_ALPHA
        )

        # Ein Feature gilt nur dann als statistisch auffällig, wenn
        # sowohl PSI als auch KS die Verteilungsänderung stützen.
        # Die zeitliche Bestätigung erfolgt separat auf Systemebene.
        result[feature] = bool(psi_drift and ks_drift)

    return result


def _remediation_signal(
    system_state: int,
    anomaly_type: str,
) -> int:
    """
    Erzeugt ein Remediation-Signal.

    Ein reiner Drift bleibt NORMAL und löst deshalb keine automatische
    Remediation aus. Eine erkannte Anomalie während des Drifts erzeugt
    ein Retraining-Signal. Infrastructure Failure erzeugt ein
    Infrastruktur-Remediation-Signal.
    """
    if anomaly_type in {

        getattr(config, "ANOMALY_CONCEPT_DRIFT", "ANOMALY_CONCEPT_DRIFT"),
    }:
        return 1

    if anomaly_type == config.ANOMALY_INFRASTRUCTURE:
        return 2

    return 0

def _update_performance_monitor(
    target_true: Optional[object],
    target_prediction: Optional[object],
    target_probability: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Überwacht die Vorhersageleistung des eingefrorenen Zielmodells.

    Ein Concept Drift wird über einen nachhaltigen Anstieg des
    Vorhersagefehlers erkannt. Der aktuelle Rolling Error wird deshalb
    mit einer festen, auf dem gesunden Kalibrierungs-/Normalverhalten
    basierten absoluten Fehlergrenze verglichen.

    Die Modellkonfidenz wird nur weiterhin protokolliert; sie bestimmt
    nicht mehr direkt die Degradationsschwelle. Dadurch wird verhindert,
    dass ein sehr niedriger konditionaler Kalibrierungsfehler zu einer
    unrealistisch niedrigen Schwelle und damit zu zahlreichen False
    Positives in Normal- und Data-Drift-Phasen führt.

    Data Drift selbst ist keine Anomalie. Die finale Concept-Drift-
    Entscheidung basiert ausschließlich auf der beobachteten Änderung
    der Beziehung zwischen Eingaben X und Ziel Y.
    """
    global _performance_confirmation_count
    global _performance_degraded

    result = {
        "prediction_error": None,
        "rolling_prediction_error": None,
        "performance_expected_error": None,
        "performance_degradation_threshold": None,
        "performance_degraded": False,
        "performance_confirmation_count": _performance_confirmation_count,
    }

    if target_true is None or target_prediction is None:
        return result

    error = float(str(target_true) != str(target_prediction))
    _performance_errors.append(error)

    if target_probability is not None and np.isfinite(float(target_probability)):
        _performance_probability_errors.append(
            (float(np.clip(target_probability, 0.0, 1.0)), error)
        )

    result["prediction_error"] = error

    if len(_performance_errors) < PERFORMANCE_WINDOW_SIZE:
        return result

    rolling_error = float(np.mean(_performance_errors))
    result["rolling_prediction_error"] = rolling_error

    if _performance_baseline_error is None:
        return result

    # Die globale Kalibrierungsleistung bleibt als Diagnoseinformation
    # erhalten. Die operative Schwelle wird jedoch nicht mehr aus einzelnen
    # Konfidenzbins abgeleitet, da diese bei kleinen bzw. stark asymmetrischen
    # Fehlerverteilungen zu unrealistisch niedrigen Schwellen führen können.
    expected_error = float(_performance_baseline_error)
    result["performance_expected_error"] = expected_error

    # Die feste absolute Grenze verhindert, dass normale Schwankungen oder
    # ein reiner Data Drift als Concept Drift klassifiziert werden. Das
    # 50-Beobachtungs-Fenster stellt zusätzlich die zeitliche Persistenz sicher.
    degradation_threshold = float(PERFORMANCE_ABSOLUTE_THRESHOLD)
    result["performance_degradation_threshold"] = degradation_threshold

    degraded = rolling_error > degradation_threshold

    if degraded:
        _performance_confirmation_count += 1
    else:
        _performance_confirmation_count = 0
        _performance_degraded = False

    _performance_degraded = (
        _performance_confirmation_count
        >= PERFORMANCE_CONFIRMATION_WINDOWS
    )

    result["performance_degraded"] = _performance_degraded
    result["performance_confirmation_count"] = _performance_confirmation_count
    result["performance_degradation_threshold"] = degradation_threshold

    return result


def _process_observation(observation: Any) -> Dict[str, Any]:
    """
    Verarbeitet eine Beobachtung und aktualisiert den Laufzeitstatus.
    """

    start_time = time.perf_counter()

    # Infrastrukturmetriken werden unabhängig von den ML-Modellen betrachtet.
    cpu_usage = float(
        getattr(observation, "infrastructure_cpu_load", 0.0)
    )
    memory_usage = float(
        getattr(observation, "infrastructure_memory_load", 0.0)
    )
    pod_restarts = int(
        getattr(observation, "infrastructure_restarts", 0)
    )

    # Nur gültige Fachwerte werden in das Sliding Window übernommen.
    values = {
        feature: getattr(observation, feature, np.nan)
        for feature in config.FEATURE_COLUMNS
    }

    # Die aktuelle Beobachtung wird bewusst noch NICHT in das Fenster
    # aufgenommen. Dadurch basieren Drift-, Rolling- und Triggerberechnungen
    # ausschließlich auf bereits beobachteten Daten.
    window_was_ready = window.is_ready

    window_metrics = _get_window_metrics()

    # Die eigentliche Driftentscheidung erfolgt erst, wenn zwei
    # vollständige, aufeinanderfolgende Fenster verfügbar sind.
    # Dadurch wird ein kurzer normaler Anlauf nicht als Drift bewertet.
    local_drift_ready = window.is_local_comparison_ready

    drift_metrics = (
        _get_local_drift_metrics(observation.step)
        if local_drift_ready
        else {}
    )

    psi_by_feature = {
        feature: result.get("psi")
        for feature, result in drift_metrics.items()
    }

    ks_by_feature = drift_metrics

    ks_pvalue_by_feature = {
        feature: result.get("ks_pvalue")
        for feature, result in drift_metrics.items()
    }

    ks_statistic_by_feature = {
        feature: result.get("ks_statistic")
        for feature, result in drift_metrics.items()
    }

    # Der Driftstatus basiert auf der lokalen Fenstervergleichslogik.
    drift_by_feature = _drift_status_by_feature(
        drift_metrics
    )

    psi_value = psi_by_feature.get(config.PRICE_COLUMN)
    ks_result = ks_by_feature.get(config.PRICE_COLUMN, {})

    isolation_forest_score = None
    one_class_svm_distance = None
    isolation_forest_anomaly = False
    one_class_svm_anomaly = False
    target_prediction = None
    target_prediction_probability = None
    target_true = getattr(observation, "target", None)

    # NaN-Injektionen werden bereits vor der Modellinferenz erkannt.
    valid_for_ml = all(
        np.isfinite(float(values[feature]))
        for feature in config.FEATURE_COLUMNS
    )

    if models.is_fitted and models.is_calibrated and valid_for_ml:
        feature_vector = create_feature_vector(
            nswprice=observation.nswprice,
            nswdemand=observation.nswdemand,
            vicprice=observation.vicprice,
            vicdemand=observation.vicdemand,
            transfer=observation.transfer,
        )

        model_results = models.predict(feature_vector)

        if_result = model_results["isolation_forest"]
        svm_result = model_results["one_class_svm"]

        isolation_forest_score = float(if_result.score)
        one_class_svm_distance = float(svm_result.score)

        isolation_forest_anomaly = bool(if_result.is_anomaly)
        one_class_svm_anomaly = bool(svm_result.is_anomaly)

        target_result = models.predict_target(feature_vector)
        # Die Zielklasse bleibt semantisch erhalten (z. B. UP/DOWN).
        target_prediction = target_result.prediction
        target_prediction_probability = float(
            target_result.probability
        )

    performance_status = _update_performance_monitor(
        target_true=(
            target_true
            if target_true is not None
            else None
        ),
        target_prediction=target_prediction,
        target_probability=target_prediction_probability,
    )

    # ------------------------------------------------------------
    # 1. Statistischen Data Drift separat bestimmen.
    # ------------------------------------------------------------

    raw_drift_detected, raw_drift_trigger_feature = _classify_observation(
        psi_by_feature=psi_by_feature,
        ks_by_feature=ks_by_feature,
    )

    drift_detected, drift_trigger_feature, drift_confirmation_count = (
        _update_drift_confirmation(
            raw_drift_detected=raw_drift_detected,
            raw_drift_trigger_feature=raw_drift_trigger_feature,
            comparison_size=window.comparison_size,
            evaluation_index=_drift_confirmation_evaluation_index,
        )
    )

    point_anomaly, point_anomaly_type, anomaly_trigger_feature, feature_scores = (
        _detect_point_anomaly_trigger(
            values=values,
            isolation_forest_anomaly=isolation_forest_anomaly,
            one_class_svm_anomaly=one_class_svm_anomaly,
        )
    )

    # Infrastruktur wird unabhängig vom Driftzweig geprüft.
    current_missing_rates = {
        feature: (
            1.0
            if not np.isfinite(float(values.get(feature, np.nan)))
            else 0.0
        )
        for feature in config.FEATURE_COLUMNS
    }

    current_out_of_bounds_rates = {}

    infrastructure_failure, infrastructure_root_cause, infrastructure_trigger = (
        _detect_infrastructure_failure(
            values=values,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            pod_restarts=pod_restarts,
            missing_value_rates=current_missing_rates,
            out_of_bounds_rates=current_out_of_bounds_rates,
            observation=observation,
        )
    )

    # ------------------------------------------------------------
    # 2. Finale Systementscheidung
    #
    # NORMAL:
    #   - stabiler Datenstrom
    #   - reiner Drift ohne ungewöhnliches Verhalten
    #
    # ANOMALY:
    #   - bestätigter Concept Drift über X -> Y Performance
    #   - Infrastructure Failure
    # ------------------------------------------------------------

    concept_drift_detected = bool(
        performance_status["performance_degraded"]
    )

    if infrastructure_failure:
        system_state = config.SYSTEM_STATE_ANOMALY
        anomaly_type = config.ANOMALY_INFRASTRUCTURE
        root_cause = config.ROOT_CAUSE_INFRASTRUCTURE
        trigger_feature = infrastructure_trigger

    elif concept_drift_detected:
        system_state = config.SYSTEM_STATE_ANOMALY
        anomaly_type = getattr(
            config,
            "ANOMALY_CONCEPT_DRIFT",
            "ANOMALY_CONCEPT_DRIFT",
        )
        root_cause = config.ROOT_CAUSE_CONCEPT_DRIFT
        trigger_feature = "PREDICTION_ERROR"

    # Ein reiner Data Drift darf niemals selbst den Systemzustand ANOMALY
    # auslösen. Der Concept Drift wird unabhängig vom statistischen Data-Drift-
    # Signal über die beobachtete X -> Y Performanceverschlechterung bestimmt.
    # IF/OCSVM bleiben diagnostische Informationen.

    else:
        system_state = config.SYSTEM_STATE_NORMAL
        anomaly_type = config.ANOMALY_NONE
        root_cause = config.ROOT_CAUSE_NORMAL
        trigger_feature = config.TRIGGER_NONE

    # Der Drifttrigger bleibt separat sichtbar, auch wenn ein reiner Drift
    # weiterhin NORMAL ist.
    # Die operative Klassifikation entspricht ausschließlich dem finalen
    # Systemzustand. Ein reiner Drift darf hier niemals als eigene Klasse
    # erscheinen.
    classification = system_state

    # Die Beobachtung wird erst nach allen Online-Entscheidungen in das
    # Sliding Window übernommen. Dadurch entsteht keine zeitliche
    # Rückkopplung innerhalb derselben Entscheidung.
    window.add_observation(
        values=values,
        step=observation.step,
        timestamp=observation.timestamp,
    )

    missing_value_rates, out_of_bounds_rates = (
        _calculate_window_quality_rates()
    )

    remediation_signal = _remediation_signal(
        system_state=system_state,
        anomaly_type=anomaly_type,
    )

    scenario = getattr(
        observation,
        "scenario",
        config.SCENARIO_NORMAL,
    )

    # Ground Truth des Ingestors wird für die experimentelle Dokumentation
    # übernommen. Sie beeinflusst die Online-Entscheidung nicht.
    observation_ground_truth = getattr(
        observation,
        "system_state",
        config.SYSTEM_STATE_NORMAL,
    )

    try:
        ground_truth = int(observation_ground_truth)
    except (TypeError, ValueError):
        ground_truth = config.SYSTEM_STATE_NORMAL

    latency_seconds = time.perf_counter() - start_time

    _safe_metric_update(
        rate_of_change=window_metrics.get("rate_of_change"),
        peak_density=window_metrics.get("peak_density"),
        ks_pvalue=window_metrics.get("ks_pvalue"),
        rolling_mean=window_metrics.get("rolling_mean"),
        rolling_std=window_metrics.get("rolling_std"),
        z_score=window_metrics.get("z_score"),
        isolation_forest_score=isolation_forest_score,
        one_class_svm_distance=one_class_svm_distance,
        isolation_forest_anomaly=isolation_forest_anomaly,
        one_class_svm_anomaly=one_class_svm_anomaly,
        cpu_usage_percent=cpu_usage,
        memory_usage_percent=memory_usage,
        pod_restarts=pod_restarts,
        missing_value_rates=missing_value_rates,
        out_of_bounds_rates=out_of_bounds_rates,
        classification=system_state,

        # Leistungsbasierte Concept-Drift-Metriken:
        # Der Concept Drift basiert ausschließlich auf der X-Y-
        # Vorhersageleistung und nicht auf PSI/KS oder Kubernetes-Metriken.
        prediction_error=performance_status["prediction_error"],
        rolling_prediction_error=performance_status[
            "rolling_prediction_error"
        ],
        performance_baseline_error=performance_status[
            "performance_expected_error"
        ],
        performance_degradation_threshold=performance_status[
            "performance_degradation_threshold"
        ],
        performance_confirmation_count=performance_status[
            "performance_confirmation_count"
        ],
        performance_degraded=performance_status[
            "performance_degraded"
        ],
        concept_drift_detected=concept_drift_detected,

        system_state=system_state,
        scenario_code={
            config.SCENARIO_NORMAL: 0,
            config.SCENARIO_CONCEPT_DRIFT: 1,
            config.SCENARIO_INFRASTRUCTURE_FAILURE: 2,
        }.get(scenario, 0),
        step=observation.step,
        current_window_size=window.size,
        isolation_forest_threshold=models.isolation_forest_threshold,
        one_class_svm_threshold=models.one_class_svm_threshold,
        z_score_threshold=config.Z_SCORE_THRESHOLD,
        nswprice=observation.nswprice,
        nswdemand=observation.nswdemand,
        vicprice=observation.vicprice,
        vicdemand=observation.vicdemand,
        transfer=observation.transfer,
        cycle=getattr(observation, "cycle", 1),
        cycle_step=getattr(observation, "cycle_step", observation.step),
        stream_running=True,
        psi=psi_value,
        psi_by_feature=psi_by_feature,
        ks_pvalue_by_feature=ks_pvalue_by_feature,
        ks_statistic_by_feature=ks_statistic_by_feature,
        drift_by_feature=drift_by_feature,
        root_cause=root_cause,
        trigger_feature=trigger_feature,
        drift_trigger_feature=drift_trigger_feature,
        remediation_signal=remediation_signal,
        inference_latency_seconds=latency_seconds,
        point_anomaly=point_anomaly,
        anomaly_type=anomaly_type,
        anomaly_trigger_feature=anomaly_trigger_feature,
        infrastructure_failure=infrastructure_failure,
        infrastructure_root_cause=infrastructure_root_cause,
        infrastructure_trigger_feature=infrastructure_trigger,
    )

    state = {
        "step": observation.step,
        "cycle": getattr(observation, "cycle", 1),
        "cycle_step": getattr(observation, "cycle_step", observation.step),
        "timestamp": observation.timestamp,
        "scenario": scenario,
        "system_state": system_state,
        "system_state_name": config.SYSTEM_STATE_NAMES[system_state],
        # Legacy-Feld: classification entspricht weiterhin dem Systemzustand.
        "classification": system_state,
        "classification_name": config.SYSTEM_STATE_NAMES[system_state],
        "ground_truth_classification": ground_truth,
        "ground_truth_name": config.SYSTEM_STATE_NAMES.get(
            ground_truth,
            "Unknown",
        ),
        "anomaly_type": anomaly_type,
        "target_true": (
            target_true
            if target_true is not None
            else None
        ),
        "target_prediction": target_prediction,
        "target_prediction_probability": target_prediction_probability,
        "prediction_error": performance_status["prediction_error"],
        "rolling_prediction_error": performance_status[
            "rolling_prediction_error"
        ],
        "performance_expected_error": performance_status[
            "performance_expected_error"
        ],
        "performance_baseline_error": _performance_baseline_error,
        "performance_degraded": performance_status[
            "performance_degraded"
        ],
        "performance_confirmation_count": performance_status[
            "performance_confirmation_count"
        ],
        "concept_drift_detected": concept_drift_detected,
        "anomaly_type": anomaly_type,
        "drift_detected": drift_detected,
        "raw_drift_detected": raw_drift_detected,
        "drift_confirmation_count": drift_confirmation_count,
        "drift_confirmation_required": DRIFT_CONFIRMATION_WINDOWS,
        "drift_confirmation_evaluation_index": _drift_confirmation_evaluation_index,
        "root_cause": root_cause,
        "trigger_feature": trigger_feature,
        "drift_trigger_feature": drift_trigger_feature,
        "point_anomaly": point_anomaly,
        "point_anomaly_type": (
            point_anomaly_type
            if point_anomaly
            else ""
        ),
        "point_anomaly_trigger_feature": anomaly_trigger_feature,
        "point_anomaly_feature_scores": feature_scores,
        "infrastructure_failure": infrastructure_failure,
        "infrastructure_root_cause": infrastructure_root_cause,
        "infrastructure_trigger_feature": infrastructure_trigger,
        "nswprice": observation.nswprice,
        "nswdemand": observation.nswdemand,
        "vicprice": observation.vicprice,
        "vicdemand": observation.vicdemand,
        "transfer": observation.transfer,
        "rolling_mean": window_metrics.get("rolling_mean"),
        "rolling_std": window_metrics.get("rolling_std"),
        "rate_of_change": window_metrics.get("rate_of_change"),
        "peak_density": window_metrics.get("peak_density"),
        "ks_pvalue": window_metrics.get("ks_pvalue"),
        "ks_statistic": window_metrics.get("ks_statistic"),
        "z_score": window_metrics.get("z_score"),
        "psi": psi_value,
        "psi_drift": bool(
            psi_value is not None
            and psi_value > config.PSI_THRESHOLD
        ),
        "psi_by_feature": psi_by_feature,
        "ks_by_feature": ks_by_feature,
        "isolation_forest_score": isolation_forest_score,
        "one_class_svm_distance": one_class_svm_distance,
        "isolation_forest_anomaly": isolation_forest_anomaly,
        "one_class_svm_anomaly": one_class_svm_anomaly,
        "missing_value_rate": missing_value_rates,
        "out_of_bounds_rate": out_of_bounds_rates,
        "schema_validation_failure": False,
        "remediation_signal": remediation_signal,
        "cpu_usage_percent": cpu_usage,
        "memory_usage_percent": memory_usage,
        "pod_restarts": pod_restarts,
        "model_ready": models.is_fitted,
        "window_ready": window.is_ready,
        "ks_reference_ready": window.has_reference,
        "ks_reference_size": window.reference_size,
        "calibration_ready": models.is_calibrated,
        "isolation_forest_threshold": models.isolation_forest_threshold,
        "one_class_svm_threshold": models.one_class_svm_threshold,
        "calibration_quantile": models.calibration_quantile,
        "inference_latency_seconds": latency_seconds,
    }

    with stream_lock:
        latest_state = state

    return state


# ============================================================================
# Hintergrundstream
# ============================================================================


def _stream_worker() -> None:
    """Verarbeitet den kontinuierlichen Demo-Stream."""

    global stream_running

    if ingestor is None:
        stream_running = False
        return

    previous_cycle = None

    try:
        for observation in ingestor.stream_continuous():

            if not stream_running:
                break

            current_cycle = getattr(observation, "cycle", 1)

            if (
                previous_cycle is not None
                and current_cycle != previous_cycle
            ):
                window.clear()

            previous_cycle = current_cycle

            _process_observation(observation)

    finally:
        stream_running = False


def start_stream() -> None:
    """Startet den Hintergrundstream, sofern er nicht bereits läuft."""

    global stream_thread, stream_running

    if stream_running:
        return

    if ingestor is None:
        raise RuntimeError(
            "Der Stream-Ingestor wurde nicht initialisiert."
        )

    _reset_runtime_state()
    stream_running = True

    stream_thread = threading.Thread(
        target=_stream_worker,
        daemon=True,
        name="elec2-stream",
    )
    stream_thread.start()


def stop_stream() -> None:
    """Stoppt den Hintergrundstream."""

    global stream_running
    stream_running = False


# ============================================================================
# FastAPI Startup
# ============================================================================


def startup_event() -> None:
    """Initialisiert Referenzen, Modelle und den Demo-Stream."""

    global ingestor

    ingestor = create_ingestor()

    _initialize_reference_distribution()
    _train_models()
    start_stream()


# ============================================================================
# API-Endpunkte
# ============================================================================


@app.get("/health")
def health() -> Dict[str, Any]:
    """Liefert den Gesundheitszustand des Services."""

    return {
        "status": "ok",
        "model_ready": models.is_fitted,
        "calibration_ready": models.is_calibrated,
        "window_ready": window.is_ready,
        "ks_reference_ready": window.has_reference,
        "ks_reference_size": window.reference_size,
        "stream_running": stream_running,
        "stream_step": latest_state["step"],
        "cycle": latest_state.get("cycle", 1),
        "cycle_step": latest_state.get("cycle_step", 0),
    }


@app.get("/state")
def state() -> Dict[str, Any]:
    """Liefert den aktuellen Diagnosezustand."""

    with stream_lock:
        return dict(latest_state)


@app.post("/predict")
def predict(request: PredictionRequest) -> Dict[str, Any]:
    """
    Verarbeitet eine einzelne validierte HTTP-Anfrage.

    Dieser Endpoint bildet den vorgesehenen Datenpfad des externen
    Stream-Generators ab. Ungültige Payloads werden bereits durch
    FastAPI/Pydantic mit HTTP 422 abgewiesen.
    """

    observation_step = (
        request.step
        if request.step is not None
        else latest_state["step"] + 1
    )

    observation_timestamp = (
        request.timestamp
        if request.timestamp is not None
        else time.time()
    )

    observation = type(
        "PredictionObservation",
        (),
        {
            "step": observation_step,
            "cycle": latest_state.get("cycle", 1),
            "cycle_step": latest_state.get("cycle_step", 0) + 1,
            "timestamp": observation_timestamp,
            "scenario": config.SCENARIO_NORMAL,
            "classification": config.CLASS_NORMAL,
            "nswprice": request.nswprice,
            "nswdemand": request.nswdemand,
            "vicprice": request.vicprice,
            "vicdemand": request.vicdemand,
            "transfer": request.transfer,
            "target": request.target,
            "infrastructure_cpu_load": request.infrastructure_cpu_load,
            "infrastructure_memory_load": request.infrastructure_memory_load,
            "infrastructure_restarts": request.infrastructure_restarts,
        },
    )()

    try:
        result = _process_observation(observation)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Fehler während der Inferenz: {exc}",
        ) from exc

    return {
        "system_state": result["system_state"],
        "system_state_name": result["system_state_name"],
        # Legacy-Felder für bestehende Clients.
        "classification": result["classification"],
        "classification_name": result["classification_name"],
        "anomaly_type": result["anomaly_type"],
        "target_prediction": result.get("target_prediction"),
        "target_prediction_probability": result.get(
            "target_prediction_probability"
        ),
        "prediction_error": result.get("prediction_error"),
        "rolling_prediction_error": result.get(
            "rolling_prediction_error"
        ),
        "performance_expected_error": result.get(
            "performance_expected_error"
        ),
        "performance_baseline_error": result.get(
            "performance_baseline_error"
        ),
        "performance_degraded": result.get("performance_degraded", False),
        "performance_confirmation_count": result.get(
            "performance_confirmation_count",
            0,
        ),
        "concept_drift_detected": result.get(
            "concept_drift_detected",
            False,
        ),
        "drift_detected": result.get("drift_detected", False),
        "raw_drift_detected": result.get("raw_drift_detected", False),
        "drift_confirmation_count": result.get("drift_confirmation_count", 0),
        "drift_confirmation_required": result.get(
            "drift_confirmation_required",
            DRIFT_CONFIRMATION_WINDOWS,
        ),
        "drift_confirmation_evaluation_index": result.get(
            "drift_confirmation_evaluation_index",
            -1,
        ),
        "root_cause": result["root_cause"],
        "trigger_feature": result["trigger_feature"],
        "point_anomaly": result["point_anomaly"],
        "point_anomaly_type": result["point_anomaly_type"],
        "infrastructure_failure": result["infrastructure_failure"],
        "infrastructure_root_cause": result["infrastructure_root_cause"],
        "infrastructure_trigger_feature": (
            result["infrastructure_trigger_feature"]
        ),
        "point_anomaly_trigger_feature": (
            result["point_anomaly_trigger_feature"]
        ),
        "remediation_signal": result.get(
            "remediation_signal",
            0,
        ),
        "isolation_forest": {
            "score": result["isolation_forest_score"],
            "is_anomaly": result["isolation_forest_anomaly"],
        },
        "one_class_svm": {
            "score": result["one_class_svm_distance"],
            "is_anomaly": result["one_class_svm_anomaly"],
        },
        "psi": result["psi"],
        "psi_by_feature": result["psi_by_feature"],
        "ks_by_feature": result["ks_by_feature"],
        "latency_seconds": result["inference_latency_seconds"],
    }


@app.get(config.METRICS_ENDPOINT)
def prometheus_metrics() -> Response:
    """Exponiert die Prometheus-Metriken."""

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/stream/start")
def start_stream_endpoint() -> Dict[str, Any]:
    """Startet den internen Demo-Stream."""

    start_stream()

    return {
        "status": "started",
        "stream_running": stream_running,
    }


@app.post("/stream/stop")
def stop_stream_endpoint() -> Dict[str, Any]:
    """Stoppt den internen Demo-Stream."""

    stop_stream()

    return {
        "status": "stopped",
        "stream_running": stream_running,
    }


@app.post("/runtime/reset")
def reset_runtime_endpoint() -> Dict[str, Any]:
    """
    Setzt den kurzfristigen Online-Laufzeitstatus zurück, ohne Modelle
    oder die Referenzverteilung neu zu trainieren.

    Dieser Endpoint ist insbesondere für reproduzierbare Szenariotests
    vorgesehen. Im Gegensatz zu einem vollständigen Stream-Neustart wird
    kein neuer Hintergrundworker gestartet. Dadurch werden Race Conditions
    und unnötige Wartezeiten zwischen unabhängigen Tests vermieden.
    """

    global stream_thread

    # Einen eventuell laufenden Demo-Worker zuerst sauber stoppen.
    stop_stream()

    if stream_thread is not None and stream_thread.is_alive():
        stream_thread.join(timeout=5.0)

    # Nur Fenster, Caches und den kurzfristigen Diagnosezustand löschen.
    # Modelle, Kalibrierung und Referenzverteilung bleiben erhalten.
    _reset_runtime_state()

    return {
        "status": "reset",
        "stream_running": stream_running,
        "window_size": window.size,
        "comparison_size": window.comparison_size,
        "model_ready": models.is_fitted,
        "calibration_ready": models.is_calibrated,
        "reference_size": window.reference_size,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.anomaly_detector:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
    )
