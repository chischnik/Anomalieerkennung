
"""
RCA-Evaluator – produktionsgetreues Replay

Dieses Skript evaluiert die Root-Cause-Analyse, indem der aktuelle
Produktionspfad aus app.anomaly_detector.py direkt wiederverwendet wird.

Wichtig:
    - Der Evaluator implementiert keine zweite Detektionslogik.
    - _process_observation() aus dem produktiven Service wird direkt
      auf jede Beobachtung angewendet.
    - Ground Truth wird ausschließlich nach der Vorhersage verwendet.
    - Ein reiner Drift bleibt NORMAL.
    - Concept Drift ist eine Anomalie, sobald die produktive Logik sie erkennt.
    - Infrastructure Failure wird unabhängig behandelt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
import sys

# ============================================================================
# Projektwurzel zum Python-Pfad hinzufügen
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from app import anomaly_detector as production
from app import models as model_module


# ============================================================================
# Pfade
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

RCA_RESULTS_FILE = RESULTS_DIR / "rca_results.json"
RCA_PREDICTIONS_FILE = RESULTS_DIR / "rca_predictions.csv"


# ============================================================================
# Konfiguration
# ============================================================================

# Der Replay-Evaluator verwendet exakt dieselben Referenz- und
# Kalibrierungsgrößen wie der produktive Detektor.
REFERENCE_SIZE = int(production.config.REFERENCE_SIZE)
CALIBRATION_SIZE = int(production.config.CALIBRATION_SIZE)

SYSTEM_STATE_NORMAL = production.config.SYSTEM_STATE_NORMAL
SYSTEM_STATE_ANOMALY = production.config.SYSTEM_STATE_ANOMALY

ANOMALY_NONE = production.config.ANOMALY_NONE
ANOMALY_CONCEPT_DRIFT = production.config.ANOMALY_CONCEPT_DRIFT
ANOMALY_INFRASTRUCTURE = production.config.ANOMALY_INFRASTRUCTURE

SCENARIO_NORMAL = production.config.SCENARIO_NORMAL
SCENARIO_DATA_DRIFT = production.config.SCENARIO_DATA_DRIFT
SCENARIO_CONCEPT_DRIFT = production.config.SCENARIO_CONCEPT_DRIFT
SCENARIO_INFRASTRUCTURE_FAILURE = (
    production.config.SCENARIO_INFRASTRUCTURE_FAILURE
)

FEATURES = list(production.config.FEATURE_COLUMNS)

INFRA_TRIGGER_LABELS = [
    "HIGH_CPU",
    "HIGH_MEMORY",
    "POD_RESTART",
    "MISSING_VALUE",
    "EXTREME_VALUE",
]


# ============================================================================
# Ground Truth
# ============================================================================

def true_system_state(observation) -> int:
    """Leitet den autoritativen Systemzustand aus der Ground Truth ab."""
    if observation.scenario in (
        SCENARIO_CONCEPT_DRIFT,
        SCENARIO_INFRASTRUCTURE_FAILURE,
    ):
        return SYSTEM_STATE_ANOMALY

    # Reiner Normalbetrieb und reiner Data Drift bleiben NORMAL.
    return SYSTEM_STATE_NORMAL


def true_anomaly_type(observation) -> str:
    """Leitet den autoritativen Anomalietyp aus der Ground Truth ab."""
    if observation.scenario == SCENARIO_INFRASTRUCTURE_FAILURE:
        return ANOMALY_INFRASTRUCTURE

    if observation.scenario == SCENARIO_CONCEPT_DRIFT:
        return ANOMALY_CONCEPT_DRIFT

    # Data Drift ist kein Anomaliezustand.
    return ANOMALY_NONE


def true_infrastructure_trigger(observation) -> Optional[str]:
    """
    Bestimmt den konkreten Infrastrukturtrigger ausschließlich für die
    spätere Evaluation.

    Das vom Stream-Ingestor gesetzte trigger_feature wird bevorzugt,
    sofern es einem definierten Infrastrukturtrigger entspricht.
    Die nachfolgende Rekonstruktion dient nur als Fallback.
    """

    # Das vom Stream-Ingestor gesetzte Trigger-Feature ist die
    # autoritative Ground Truth für die Evaluation.
    trigger = getattr(
        observation,
        "trigger_feature",
        None,
    )

    if trigger in INFRA_TRIGGER_LABELS:
        return trigger

    if bool(
        getattr(
            observation,
            "missing_value_injected",
            False,
        )
    ):
        return "MISSING_VALUE"

    if bool(
        getattr(
            observation,
            "extreme_value_injected",
            False,
        )
    ):
        return "EXTREME_VALUE"

    cpu = float(
        getattr(
            observation,
            "infrastructure_cpu_load",
            0.0,
        )
    )

    memory = float(
        getattr(
            observation,
            "infrastructure_memory_load",
            0.0,
        )
    )

    restarts = int(
        getattr(
            observation,
            "infrastructure_restarts",
            0,
        )
    )

    if cpu >= production.config.CPU_ALERT_PERCENT:
        return "HIGH_CPU"

    if memory >= production.config.MEMORY_ALERT_PERCENT:
        return "HIGH_MEMORY"

    if restarts >= production.config.POD_RESTART_ALERT_COUNT:
        return "POD_RESTART"

    return None


# ============================================================================
# Metriken
# ============================================================================

def classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> Dict:
    """Berechnet robuste Klassifikationsmetriken."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
        "support": int(len(y_true)),
        "true_label_distribution": (
            pd.Series(y_true)
            .value_counts()
            .to_dict()
        ),
        "predicted_label_distribution": (
            pd.Series(y_pred)
            .value_counts()
            .to_dict()
        ),
    }


def binary_metrics(
    y_true: List[bool],
    y_pred: List[bool],
) -> Dict:
    """Berechnet binäre Erkennungsmetriken."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[False, True],
        average="binary",
        zero_division=0,
    )

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support_positive": int(np.sum(y_true)),
        "support_negative": int(
            len(y_true) - np.sum(y_true)
        ),
    }


# ============================================================================
# Batch-Inferenz für den Offline-Replay
# ============================================================================

def _build_batch_model_cache(
    observations: List,
    detector,
) -> Dict[tuple, Dict[str, object]]:
    """
    Berechnet die Modellvorhersagen für den Replay einmalig im Batch.

    Der produktive Dienst verarbeitet einzelne Beobachtungen. Für einen
    Offline-Replay mit mehr als 40.000 Beobachtungen würde dieselbe
    Einzelvorhersage jedoch einen erheblichen Laufzeit-Overhead erzeugen.
    Deshalb werden die bereits trainierten Modelle hier einmal auf dem
    vollständigen Replay-Block ausgeführt. Die erzeugten Ergebnisse werden
    anschließend vom unveränderten Produktionspfad wiederverwendet.

    Die Modellparameter, Schwellenwerte und Entscheidungsfunktionen bleiben
    identisch. Es wird ausschließlich die technische Ausführungsform von
    Einzelinferenz auf Batch-Inferenz für die Evaluation geändert.
    """
    feature_rows = []
    valid_keys = []

    for observation in observations:
        values = np.asarray(
            [
                observation.nswprice,
                observation.nswdemand,
                observation.vicprice,
                observation.vicdemand,
                observation.transfer,
            ],
            dtype=float,
        )

        if not np.all(np.isfinite(values)):
            continue

        key = tuple(float(value) for value in values)
        feature_rows.append(values)
        valid_keys.append(key)

    if not feature_rows:
        return {}

    X = np.asarray(feature_rows, dtype=float)

    print(
        "Batch-Inferenz : berechne IF, One-Class SVM und "
        "Target Predictor ..."
    )

    # ------------------------------------------------------------------
    # Isolation Forest
    # ------------------------------------------------------------------
    isolation_detector = detector.models.isolation_forest
    isolation_scores = isolation_detector.model.decision_function(X)
    isolation_threshold = float(
        detector.models.isolation_forest_threshold
    )
    isolation_anomaly = isolation_scores < isolation_threshold

    # ------------------------------------------------------------------
    # One-Class SVM
    # ------------------------------------------------------------------
    svm_detector = detector.models.one_class_svm
    X_scaled = svm_detector.scaler.transform(X)
    svm_scores = svm_detector.model.decision_function(X_scaled)
    svm_threshold = float(
        detector.models.one_class_svm_threshold
    )
    svm_anomaly = svm_scores < svm_threshold

    # ------------------------------------------------------------------
    # Überwachtes Zielmodell für Concept Drift
    # ------------------------------------------------------------------
    target_predictor = detector.models.target_predictor
    encoded_predictions = target_predictor.model.predict(X)
    decoded_predictions = target_predictor.label_encoder.inverse_transform(
        encoded_predictions
    )
    probabilities = target_predictor.model.predict_proba(X)
    max_probabilities = np.max(probabilities, axis=1)

    cache: Dict[tuple, Dict[str, object]] = {}

    for index, key in enumerate(valid_keys):
        isolation_score = float(isolation_scores[index])
        svm_score = float(svm_scores[index])

        cache[key] = {
            "model": {
                "isolation_forest": model_module.DetectorResult(
                    score=isolation_score,
                    anomaly_score=-isolation_score,
                    prediction=(
                        -1
                        if bool(isolation_anomaly[index])
                        else 1
                    ),
                    is_anomaly=bool(isolation_anomaly[index]),
                ),
                "one_class_svm": model_module.DetectorResult(
                    score=svm_score,
                    anomaly_score=-svm_score,
                    prediction=(
                        -1
                        if bool(svm_anomaly[index])
                        else 1
                    ),
                    is_anomaly=bool(svm_anomaly[index]),
                ),
            },
            "target": model_module.TargetPredictionResult(
                prediction=str(decoded_predictions[index]),
                probability=float(max_probabilities[index]),
            ),
        }

    print(
        f"Batch-Inferenz : {len(cache)} gültige Feature-Vektoren vorbereitet."
    )

    return cache


def _install_batch_model_cache(
    detector,
    cache: Dict[tuple, Dict[str, object]],
):
    """
    Ersetzt die Einzelinferenz des Modellobjekts temporär durch Cache-Zugriff.

    Der Produktionscode ruft weiterhin models.predict() und
    models.predict_target() auf. Beide Funktionen liefern hier lediglich die
    zuvor exakt mit den trainierten Produktionsmodellen berechneten Ergebnisse.
    """
    original_predict = detector.models.predict
    original_predict_target = detector.models.predict_target

    def cached_predict(X):
        values = np.asarray(X, dtype=float).reshape(-1)
        key = tuple(float(value) for value in values)

        if key not in cache:
            return original_predict(X)

        return cache[key]["model"]

    def cached_predict_target(X):
        values = np.asarray(X, dtype=float).reshape(-1)
        key = tuple(float(value) for value in values)

        if key not in cache:
            return original_predict_target(X)

        return cache[key]["target"]

    detector.models.predict = cached_predict
    detector.models.predict_target = cached_predict_target

    return original_predict, original_predict_target


def _restore_model_prediction_functions(
    detector,
    originals,
) -> None:
    """Stellt die ursprünglichen Produktionsmethoden wieder her."""
    original_predict, original_predict_target = originals
    detector.models.predict = original_predict
    detector.models.predict_target = original_predict_target


# ============================================================================
# Produktionsgetreues Replay
# ============================================================================

def run_production_replay() -> pd.DataFrame:
    """
    Führt den gesamten Datenstrom durch den echten Onlinepfad.

    Es wird bewusst nicht versucht, die Funktionen aus
    anomaly_detector.py nachzuimplementieren. Dadurch ist die Evaluation
    an die tatsächlich eingesetzte Produktionslogik gekoppelt.
    """
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    detector = production

    print("=" * 72)
    print(
        "RCA EVALUATOR - "
        "PRODUKTIONSGETREUES REPLAY"
    )
    print("=" * 72)
    print(
        "Der produktive _process_observation()-Pfad "
        "wird direkt wiederverwendet."
    )
    print(
        "Data Drift = NORMAL; "
        "Concept Drift = ANOMALY_CONCEPT_DRIFT"
    )
    print(
        "Infrastructure Failure = INFRASTRUCTURE_FAILURE"
    )

    # ------------------------------------------------------------------
    # Produktionskomponenten exakt wie beim FastAPI-Startup initialisieren.
    # ------------------------------------------------------------------
    detector.ingestor = detector.create_ingestor()

    detector._initialize_reference_distribution()
    detector._train_models()
    detector._reset_runtime_state()

    ingestor = detector.ingestor

    evaluation_start = REFERENCE_SIZE + CALIBRATION_SIZE

    # Der Replay verwendet exakt dieselbe Szenariologik wie der
    # aktuelle kontinuierliche Experimentstrom.
    #
    # include_concept_drift=True aktiviert die eigenständige Concept-Drift-Phase.
    # live_demo=True aktiviert die deterministischen Infrastrukturindikatoren.
    #
    # Dadurch wird die Ground Truth direkt durch den Stream-Ingestor erzeugt.
    # Eine nachträgliche manuelle Concept-Drift-Injektion ist nicht erforderlich.
    observations = [
        ingestor._create_observation(
            step,
            include_concept_drift=True,
            live_demo=True,
        )
        for step in range(ingestor.total_rows)
    ]

    model_cache = _build_batch_model_cache(
        observations,
        detector,
    )
    original_model_functions = _install_batch_model_cache(
        detector,
        model_cache,
    )

    print()
    print(f"Dataset       : {ingestor.total_rows}")
    print(f"Reference     : {REFERENCE_SIZE}")
    print(f"Calibration   : {CALIBRATION_SIZE}")
    print(f"Evaluation    : ab Schritt {evaluation_start}")
    print(f"Features      : {', '.join(FEATURES)}")
    print(f"Window        : {detector.config.WINDOW_SIZE}")
    print(
        "Drift-Cache   : "
        f"{detector.LOCAL_DRIFT_CACHE_INTERVAL}"
    )

    total_rows = int(getattr(ingestor, "total_rows", 0))

    # Die Phasengrenzen stammen ausschließlich aus dem aktuellen
    # Stream-Ingestor. Der Evaluator definiert keine eigene Szenariologik.
    phase_summary = ingestor.get_phase_summary()

    print("Aktuelle Stream-Phasen:")
    print(phase_summary.to_string(index=False))
    print()

    rows = []
    previous_cycle = None

    # stream_fast erzeugt dieselben StreamObservation-Objekte wie der
    # produktive Datenstrom. Die Ground Truth bleibt dabei außerhalb
    # der Entscheidungslogik.
    try:
        for step, observation in enumerate(observations):
            current_cycle = getattr(
                observation,
                "cycle",
                1,
            )

            # Der Produktions-Worker setzt das kurzfristige Fenster bei einem
            # Zykluswechsel zurück. Dieses Verhalten wird exakt nachgebildet.
            if (
                previous_cycle is not None
                and current_cycle != previous_cycle
            ):
                detector.window.clear()

                # Auch die produktiven Drift-/Fenster-Caches werden beim
                # Zurücksetzen des Laufzeitfensters invalidiert.
                detector._drift_cache_metrics = {}
                detector._drift_cache_comparison_size = -1
                detector._window_metrics_cache = {}
                detector._window_metrics_cache_size = -1

            previous_cycle = current_cycle

            # ------------------------------------------------------------------
            # Concept Drift wird bereits durch den Stream-Ingestor erzeugt.
            # ------------------------------------------------------------------
            # Die aktuelle StreamObservation enthält die vollständige
            # Szenariologik bereits beim Erzeugen:
            #
            #   - Data Drift:     Änderung von P(X), bleibt NORMAL.
            #   - Concept Drift: Änderung von P(Y|X), wird als Anomalie
            #                    bewertet, sobald die Produktionslogik die
            #                    persistente Performance-Degradation bestätigt.
            #   - Infrastructure Failure: separate Infrastrukturphase.
            #
            # Eine nachträgliche Anwendung von apply_target_concept_drift()
            # würde die aktuelle Experimentdefinition erneut verändern und
            # damit die Produktions- und Ground-Truth-Logik voneinander
            # entkoppeln. Deshalb wird hier keine zweite Drift-Injektion
            # durchgeführt.
            #
            # Ab hier keine eigene Detektionslogik:
            # Der echte Produktionspfad entscheidet.
            predicted = detector._process_observation(
                observation
            )

            if observation.step < evaluation_start:
                continue

            predicted_state = int(
                predicted.get(
                    "system_state",
                    predicted.get(
                        "classification",
                        SYSTEM_STATE_NORMAL,
                    ),
                )
            )

            predicted_type = predicted.get(
                "anomaly_type",
                ANOMALY_NONE,
            )

            predicted_root_cause = predicted.get(
                "root_cause",
                production.config.ROOT_CAUSE_NORMAL,
            )

            predicted_trigger = (
                predicted.get(
                    "trigger_feature",
                    "",
                )
                or ""
            )

            infrastructure_trigger = (
                predicted.get(
                    "infrastructure_trigger_feature",
                    "",
                )
                or ""
            )

            infrastructure_failure = bool(
                predicted.get(
                    "infrastructure_failure",
                    False,
                )
            )

            drift_detected = bool(
                predicted.get(
                    "drift_detected",
                    False,
                )
            )

            true_trigger = None

            if observation.scenario == SCENARIO_INFRASTRUCTURE_FAILURE:
                true_trigger = true_infrastructure_trigger(observation)

            row = {
                "step": int(observation.step),
                "cycle": int(
                    getattr(
                        observation,
                        "cycle",
                        1,
                    )
                ),
                "scenario": observation.scenario,
                "classification": int(
                    observation.classification
                ),
                "true_system_state": true_system_state(
                    observation
                ),
                "true_anomaly_type": true_anomaly_type(
                    observation
                ),
                "true_root_cause": getattr(
                    observation,
                    "root_cause",
                    production.config.ROOT_CAUSE_NORMAL,
                ),
                "true_trigger_feature": true_trigger,
                "missing_value_injected": bool(
                    getattr(
                        observation,
                        "missing_value_injected",
                        False,
                    )
                ),
                "extreme_value_injected": bool(
                    getattr(
                        observation,
                        "extreme_value_injected",
                        False,
                    )
                ),
                "infrastructure_cpu_load": float(
                    getattr(
                        observation,
                        "infrastructure_cpu_load",
                        0.0,
                    )
                ),
                "infrastructure_memory_load": float(
                    getattr(
                        observation,
                        "infrastructure_memory_load",
                        0.0,
                    )
                ),
                "infrastructure_restarts": int(
                    getattr(
                        observation,
                        "infrastructure_restarts",
                        0,
                    )
                ),
                "predicted_system_state": predicted_state,
                "predicted_anomaly_type": predicted_type,
                "predicted_root_cause": predicted_root_cause,
                "predicted_trigger_feature": predicted_trigger,
                "data_drift_signal": drift_detected,
                "infrastructure_signal": infrastructure_failure,
                "infrastructure_trigger": infrastructure_trigger,
                "concept_drift_signal": bool(
                    predicted.get("concept_drift_detected", False)
                ),

                # ----------------------------------------------------------
                # Performance-Diagnostik für die Schwellenwertanalyse
                # ----------------------------------------------------------
                # Diese Werte beeinflussen die Evaluation nicht.
                # Sie machen lediglich sichtbar, warum eine Beobachtung
                # als Concept Drift bestätigt oder verworfen wurde.
                "target_true": predicted.get("target_true"),
                "target_prediction": predicted.get("target_prediction"),
                "target_prediction_probability": predicted.get(
                    "target_prediction_probability"
                ),
                "prediction_error": predicted.get("prediction_error"),
                "rolling_prediction_error": predicted.get(
                    "rolling_prediction_error"
                ),
                "performance_expected_error": predicted.get(
                    "performance_expected_error"
                ),
                "performance_baseline_error": predicted.get(
                    "performance_baseline_error"
                ),
                "performance_degraded": bool(
                    predicted.get("performance_degraded", False)
                ),
                "performance_confirmation_count": predicted.get(
                    "performance_confirmation_count"
                ),

                "isolation_forest_anomaly": bool(
                    predicted.get(
                        "isolation_forest_anomaly",
                        False,
                    )
                ),
                "one_class_svm_anomaly": bool(
                    predicted.get(
                        "one_class_svm_anomaly",
                        False,
                    )
                ),
                "isolation_forest_score": predicted.get(
                    "isolation_forest_score"
                ),
                "one_class_svm_distance": predicted.get(
                    "one_class_svm_distance"
                ),
                "inference_latency_seconds": predicted.get(
                    "inference_latency_seconds"
                ),
            }

            rows.append(row)

            if (
                step > 0
                and step % 5000 == 0
            ):
                print(
                    f"Progression : "
                    f"{step}/{ingestor.total_rows}"
                )

    finally:
        _restore_model_prediction_functions(
            detector,
            original_model_functions,
        )

    return pd.DataFrame(rows)


# ============================================================================
# Analyse temporelle des Phasen
# ============================================================================

def analyze_concept_drift_temporally(
    df: pd.DataFrame,
) -> Dict:
    """
    Analysiert den zeitlichen Verlauf des Concept Drifts.

    Diese Analyse verändert keine Produktionsentscheidung. Sie dient nur dazu,
    nachvollziehbar zu machen, wann sich der Prädiktionsfehler verändert und
    wann der produktive Concept-Drift-Indikator tatsächlich aktiviert wird.
    """
    if df.empty:
        return {
            "available": False,
            "reason": "Keine Daten für die zeitliche Analyse vorhanden.",
        }

    concept_df = df.loc[
        df["scenario"] == SCENARIO_CONCEPT_DRIFT
    ].copy()

    if concept_df.empty:
        return {
            "available": False,
            "reason": "Keine Concept-Drift-Beobachtungen vorhanden.",
        }

    concept_df = concept_df.sort_values("step")

    # Nur tatsächlich vorhandene numerische Werte verwenden.
    rolling_error = pd.to_numeric(
        concept_df["rolling_prediction_error"],
        errors="coerce",
    )

    performance_degraded = concept_df[
        "performance_degraded"
    ].astype(bool)

    concept_signal = concept_df[
        "concept_drift_signal"
    ].astype(bool)

    detected_type = (
        concept_df["predicted_anomaly_type"]
        == ANOMALY_CONCEPT_DRIFT
    )

    valid_error = rolling_error.dropna()

    result = {
        "available": True,
        "start_step": int(concept_df["step"].min()),
        "end_step": int(concept_df["step"].max()),
        "observations": int(len(concept_df)),
        "detected_concept_drift_observations": int(
            detected_type.sum()
        ),
        "concept_drift_recall": float(
            detected_type.mean()
        ),
        "performance_degraded_observations": int(
            performance_degraded.sum()
        ),
        "concept_signal_observations": int(
            concept_signal.sum()
        ),
    }

    if not valid_error.empty:
        result.update(
            {
                "rolling_error_mean": float(valid_error.mean()),
                "rolling_error_median": float(valid_error.median()),
                "rolling_error_q95": float(valid_error.quantile(0.95)),
                "rolling_error_max": float(valid_error.max()),
            }
        )

        # Erster Schritt, an dem die produktive Performance-Degradation
        # tatsächlich aktiviert wurde.
        degraded_steps = concept_df.loc[
            performance_degraded,
            "step",
        ]

        if not degraded_steps.empty:
            result["first_performance_degraded_step"] = int(
                degraded_steps.min()
            )
            result["performance_degraded_delay"] = int(
                degraded_steps.min() - concept_df["step"].min()
            )
        else:
            result["first_performance_degraded_step"] = None
            result["performance_degraded_delay"] = None

        # Erster tatsächlich als Concept Drift klassifizierter Schritt.
        detected_steps = concept_df.loc[
            detected_type,
            "step",
        ]

        if not detected_steps.empty:
            result["first_concept_drift_detection_step"] = int(
                detected_steps.min()
            )
            result["concept_drift_detection_delay"] = int(
                detected_steps.min() - concept_df["step"].min()
            )
        else:
            result["first_concept_drift_detection_step"] = None
            result["concept_drift_detection_delay"] = None

    # Analyse in fünf gleich großen Zeitabschnitten.
    segments = []
    segment_count = 5
    segment_size = max(1, int(np.ceil(len(concept_df) / segment_count)))

    for segment_index in range(segment_count):
        start = segment_index * segment_size
        end = min((segment_index + 1) * segment_size, len(concept_df))

        if start >= len(concept_df):
            break

        segment = concept_df.iloc[start:end]
        segment_errors = pd.to_numeric(
            segment["rolling_prediction_error"],
            errors="coerce",
        ).dropna()

        segments.append(
            {
                "segment": int(segment_index + 1),
                "start_step": int(segment["step"].min()),
                "end_step": int(segment["step"].max()),
                "observations": int(len(segment)),
                "mean_rolling_error": (
                    float(segment_errors.mean())
                    if not segment_errors.empty
                    else None
                ),
                "performance_degraded_rate": float(
                    segment["performance_degraded"].astype(bool).mean()
                ),
                "concept_drift_detection_rate": float(
                    (
                        segment["concept_drift_signal"]
                        .astype(bool)
                    ).mean()
                ),
                "concept_anomaly_rate": float(
                    (
                        segment["predicted_anomaly_type"]
                        == ANOMALY_CONCEPT_DRIFT
                    ).mean()
                ),
            }
        )

    result["temporal_segments"] = segments

    return result



# ============================================================================
# Auswertung
# ============================================================================

def evaluate_predictions(
    df: pd.DataFrame,
) -> Dict:
    """Berechnet alle RCA-relevanten Kennzahlen."""
    if df.empty:
        raise RuntimeError(
            "Keine Beobachtungen wurden evaluiert."
        )

    # ------------------------------------------------------------------
    # SYSTEM_STATE
    # ------------------------------------------------------------------
    system_metrics = classification_metrics(
        df["true_system_state"].astype(str).tolist(),
        df["predicted_system_state"].astype(str).tolist(),
        [
            str(SYSTEM_STATE_NORMAL),
            str(SYSTEM_STATE_ANOMALY),
        ],
    )

    # ------------------------------------------------------------------
    # ANOMALY_TYPE
    # ------------------------------------------------------------------
    anomaly_metrics = classification_metrics(
        df["true_anomaly_type"].tolist(),
        df["predicted_anomaly_type"].tolist(),
        [
            ANOMALY_NONE,
            ANOMALY_CONCEPT_DRIFT,
            ANOMALY_INFRASTRUCTURE,
        ],
    )

    # Nur echte Anomalien: NONE wird ausgeschlossen.
    anomaly_only = df.loc[
        df["true_anomaly_type"] != ANOMALY_NONE
    ].copy()

    conditional_metrics = classification_metrics(
        anomaly_only["true_anomaly_type"].tolist(),
        anomaly_only["predicted_anomaly_type"].tolist(),
        [
            ANOMALY_CONCEPT_DRIFT,
            ANOMALY_INFRASTRUCTURE,
        ],
    )

    # ------------------------------------------------------------------
    # Unabhängige Signale
    # ------------------------------------------------------------------
    true_data_drift = (
        df["scenario"] == SCENARIO_DATA_DRIFT
    )

    true_infrastructure = (
        df["scenario"]
        == SCENARIO_INFRASTRUCTURE_FAILURE
    )

    true_concept_drift = (
        df["scenario"] == SCENARIO_CONCEPT_DRIFT
    )

    data_metrics = binary_metrics(
        true_data_drift.tolist(),
        df["data_drift_signal"].astype(bool).tolist(),
    )

    infrastructure_metrics = binary_metrics(
        true_infrastructure.tolist(),
        df["infrastructure_signal"]
        .astype(bool)
        .tolist(),
    )

    concept_drift_metrics = binary_metrics(
        true_concept_drift.tolist(),
        df["concept_drift_signal"].astype(bool).tolist(),
    )

    # ------------------------------------------------------------------
    # Infrastructure Trigger RCA
    #
    # Für die eigentliche Trigger-Lokalisierung werden nur Beobachtungen
    # betrachtet, für die die Ground Truth einen konkreten Trigger besitzt.
    # ------------------------------------------------------------------
    infra_df = df.loc[
        (
            df["scenario"]
            == SCENARIO_INFRASTRUCTURE_FAILURE
        )
        & df["true_trigger_feature"].notna()
        & (
            df["true_trigger_feature"].astype(str)
            != ""
        )
    ].copy()

    infra_true = (
        infra_df["true_trigger_feature"]
        .astype(str)
        .tolist()
    )

    infra_pred = (
        infra_df["infrastructure_trigger"]
        .astype(str)
        .tolist()
    )

    infra_labels = [
        label
        for label in INFRA_TRIGGER_LABELS
        if (
            label in infra_true
            or label in infra_pred
        )
    ]

    infrastructure_rca = classification_metrics(
        infra_true,
        infra_pred,
        infra_labels,
    )

    return {
        "evaluation_version": (
            "RCA_PRODUCTION_REPLAY"
        ),
        "dataset_size": int(
            getattr(
                production.ingestor,
                "total_rows",
                0,
            )
        ),
        "evaluated_observations": int(len(df)),
        "reference_size": REFERENCE_SIZE,
        "calibration_size": CALIBRATION_SIZE,
        "evaluation_start": (
            REFERENCE_SIZE + CALIBRATION_SIZE
        ),
        "window_size": int(
            production.config.WINDOW_SIZE
        ),
        "statistical_step": int(
            production.LOCAL_DRIFT_CACHE_INTERVAL
        ),
        "feature_columns": FEATURES,
        "final_system_state": system_metrics,
        "final_anomaly_type": anomaly_metrics,
        "anomaly_type_conditional_on_anomaly": (
            conditional_metrics
        ),
        "independent_data_signal": data_metrics,
        "concept_drift_detection": concept_drift_metrics,
        "independent_infrastructure_signal": (
            infrastructure_metrics
        ),
        "independent_concept_drift_signal": (
            concept_drift_metrics
        ),
        "concept_drift_temporal_analysis": (
            analyze_concept_drift_temporally(df)
        ),
        "infrastructure_trigger_rca": (
            infrastructure_rca
        ),
        "rca_scope": {
            "infrastructure_trigger_evaluated_on": (
                "infrastructure observations with a concrete "
                "ground-truth trigger"
            ),
            "normal_and_pure_drift": (
                "not counted as RCA positives"
            ),
        },
        "counts": {
            "infrastructure_observations": int(
                true_infrastructure.sum()
            ),
            "concept_drift_observations": int(
                true_concept_drift.sum()
            ),
            "true_anomaly_observations": int(
                (
                    df["true_system_state"]
                    == SYSTEM_STATE_ANOMALY
                ).sum()
            ),
            "true_normal_observations": int(
                (
                    df["true_system_state"]
                    == SYSTEM_STATE_NORMAL
                ).sum()
            ),
        },
        "methodological_note": (
            "Der Evaluator verwendet den produktiven "
            "_process_observation()-Pfad direkt. Data Drift wird "
            "als NORMAL behandelt. Concept Drift wird durch eine "
            "in einem separaten Zeitintervall außerhalb des Data-Drift-"
            "Intervalls durch eine kontrollierte Änderung der "
            "Zielbeziehung X -> Y erzeugt und als ANOMALY_CONCEPT_DRIFT "
            "bewertet. Dadurch werden Data Drift und Concept Drift "
            "im Experiment nicht gleichzeitig injiziert. Infrastructure "
            "Failure wird unabhängig klassifiziert. Ground-Truth-Felder "
            "werden erst nach der Vorhersage für die Evaluation verwendet. "
            "Zusätzliche Performance-Felder im Predictions-CSV dienen "
            "ausschließlich der nachgelagerten Analyse der Concept-Drift-"
            "Schwelle und verändern die Produktionsentscheidung nicht."
        ),
        "output_files": {
            "predictions": str(
                RCA_PREDICTIONS_FILE
            ),
            "summary": str(
                RCA_RESULTS_FILE
            ),
        },
    }


# ============================================================================
# Hauptprogramm
# ============================================================================

def main() -> None:
    """Startet Replay, Evaluation und Speicherung."""
    df = run_production_replay()

    if df.empty:
        raise RuntimeError(
            "Der Produktions-Replay lieferte keine Evaluationsdaten."
        )

    summary = evaluate_predictions(df)

    df.to_csv(
        RCA_PREDICTIONS_FILE,
        index=False,
    )

    with open(
        RCA_RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print("=" * 72)
    print("RCA - PRODUKTIONSGETREUES ERGEBNIS")
    print("=" * 72)

    print(
        "1. SYSTEM_STATE"
    )
    print(
        f"   Accuracy : "
        f"{summary['final_system_state']['accuracy']:.4f}"
    )
    print(
        f"   Macro-F1 : "
        f"{summary['final_system_state']['macro_f1']:.4f}"
    )

    print(
        "2. ANOMALY_TYPE"
    )
    print(
        f"   Accuracy : "
        f"{summary['final_anomaly_type']['accuracy']:.4f}"
    )
    print(
        f"   Macro-F1 : "
        f"{summary['final_anomaly_type']['macro_f1']:.4f}"
    )

    print(
        "3. CONCEPT DRIFT"
    )
    print(
        f"   Precision : "
        f"{summary['concept_drift_detection']['precision']:.4f}"
    )
    print(
        f"   Recall    : "
        f"{summary['concept_drift_detection']['recall']:.4f}"
    )
    print(
        f"   F1        : "
        f"{summary['concept_drift_detection']['f1']:.4f}"
    )

    temporal = summary["concept_drift_temporal_analysis"]

    if temporal.get("available", False):
        print()
        print("   Concept-Drift-Zeitverlauf")
        print(
            f"   Start     : {temporal['start_step']}"
        )
        print(
            f"   Ende      : {temporal['end_step']}"
        )
        print(
            f"   Erkannt   : "
            f"{temporal['detected_concept_drift_observations']}/"
            f"{temporal['observations']}"
        )
        print(
            f"   Recall    : "
            f"{temporal['concept_drift_recall']:.4f}"
        )
        print(
            f"   Fehler Ø  : "
            f"{temporal.get('rolling_error_mean', float('nan')):.4f}"
        )
        print(
            f"   Fehler q95: "
            f"{temporal.get('rolling_error_q95', float('nan')):.4f}"
        )
        print(
            f"   Erste Performance-Degradation: "
            f"{temporal.get('first_performance_degraded_step')}"
        )
        print(
            f"   Erste Concept-Drift-Erkennung: "
            f"{temporal.get('first_concept_drift_detection_step')}"
        )

    print(
        "4. Infrastructure Signal"
    )
    print(
        f"   Precision : "
        f"{summary['independent_infrastructure_signal']['precision']:.4f}"
    )
    print(
        f"   Recall    : "
        f"{summary['independent_infrastructure_signal']['recall']:.4f}"
    )
    print(
        f"   F1        : "
        f"{summary['independent_infrastructure_signal']['f1']:.4f}"
    )

    print(
        "5. Infrastructure Trigger RCA"
    )
    print(
        f"   Accuracy : "
        f"{summary['infrastructure_trigger_rca']['accuracy']:.4f}"
    )
    print(
        f"   Macro-F1 : "
        f"{summary['infrastructure_trigger_rca']['macro_f1']:.4f}"
    )

    print()
    print(
        f"CSV : {RCA_PREDICTIONS_FILE}"
    )
    print(
        f"JSON: {RCA_RESULTS_FILE}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
