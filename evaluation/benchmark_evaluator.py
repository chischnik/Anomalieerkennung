"""
FINAL – Benchmark für Data Drift, Concept Drift und Infrastructure Failure.

Die operative Systemklassifikation verwendet zwei Klassen:

    0 = Normal
    1 = Anomaly

Die Anomalietypen werden getrennt betrachtet:

    NONE
    ANOMALY_CONCEPT_DRIFT
    INFRASTRUCTURE_FAILURE

Wichtige methodische Trennung:

    Data Drift:
        Änderung der Eingangsverteilung P(X).
        PSI und KS dienen ausschließlich zur Erkennung dieser
        statistischen Verteilungsänderung.
        Data Drift allein ist KEINE Systemanomalie.

    Concept Drift:
        Änderung der Beziehung X -> Y bzw. P(Y|X).
        Concept Drift wird über eine persistente Verschlechterung
        der Vorhersageleistung eines eingefrorenen Zielmodells erkannt.

    Infrastructure Failure:
        Unabhängige Anomaliefamilie auf Basis der kontrollierten
        Infrastrukturindikatoren des Experiments.

Vergleichsmethoden:

    - Isolation Forest
    - One-Class SVM
    - PSI
    - KS
    - PSI+KS
    - Concept Drift (Performance)

Das Benchmark verwendet die aktuelle Schnittstelle von
app.stream_ingestor.Elec2StreamIngestor.

Die Ground Truth wird ausschließlich für die spätere Evaluation
verwendet und beeinflusst keine Detektorentscheidung.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


# ============================================================================
# Projektpfad
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Projektmodule
# ============================================================================

from app import config
from app.stream_ingestor import create_ingestor


# ============================================================================
# Ergebnisverzeichnisse
# ============================================================================

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

PREDICTIONS_FILE = RESULTS_DIR / "final_benchmark_predictions.csv"
SUMMARY_FILE = RESULTS_DIR / "final_benchmark_summary.json"
EVENTS_FILE = RESULTS_DIR / "final_benchmark_events.csv"
DELAY_FILE = RESULTS_DIR / "final_benchmark_delay.csv"


# ============================================================================
# Eingangsmerkmale
# ============================================================================

FEATURES = [
    "nswprice",
    "nswdemand",
    "vicprice",
    "vicdemand",
    "transfer",
]


# ============================================================================
# Gemeinsame Versuchsparameter
# ============================================================================

# Die zentralen Versuchsparameter werden aus app.config übernommen.
# Dadurch verwenden Produktionspfad und Benchmark dieselben Basisparameter.

REFERENCE_SIZE = int(config.REFERENCE_SIZE)
CALIBRATION_SIZE = int(config.CALIBRATION_SIZE)
TEST_START = REFERENCE_SIZE + CALIBRATION_SIZE

WINDOW_SIZE = int(config.WINDOW_SIZE)
MIN_WINDOW_SIZE = int(config.MIN_WINDOW_SIZE)

DRIFT_PERSISTENCE_WINDOWS = int(
    config.DRIFT_CONFIRMATION_WINDOWS
)

PSI_THRESHOLD = float(config.PSI_THRESHOLD)
KS_ALPHA = float(config.KS_ALPHA)

IF_ESTIMATORS = int(
    config.ISOLATION_FOREST_ESTIMATORS
)

IF_CONTAMINATION = float(
    config.ISOLATION_FOREST_CONTAMINATION
)

OCSVM_KERNEL = config.ONE_CLASS_SVM_KERNEL
OCSVM_NU = float(config.ONE_CLASS_SVM_NU)
OCSVM_GAMMA = config.ONE_CLASS_SVM_GAMMA

CALIBRATION_QUANTILE = float(
    config.CALIBRATION_QUANTILE
)

RANDOM_STATE = int(
    config.RANDOM_STATE
)


# ============================================================================
# Parameter für die Concept-Drift-Erkennung
# ============================================================================

# Diese Parameter entsprechen bewusst der final validierten V16-Logik.

PERFORMANCE_WINDOW_SIZE = 50

PERFORMANCE_DEGRADATION_MARGIN = 0.10

PERFORMANCE_DEGRADATION_FACTOR = 3.0

PERFORMANCE_CONFIRMATION_WINDOWS = 3

# Absoluter Schwellenwert der finalen V16-Implementierung.
#
# Wichtig:
# Der Wert 0.65 wird bewusst eingefroren und nicht während des
# Benchmarks aus den Testdaten optimiert.

PERFORMANCE_ABSOLUTE_THRESHOLD = 0.65


# ============================================================================
# Statistische Evaluationsfrequenz
# ============================================================================

# PSI und KS werden nicht bei jeder Beobachtung neu berechnet.
# Dadurch bleibt die Laufzeit des vollständigen Benchmarks praktikabel.

STATISTICAL_STEP = int(
    config.STATISTICAL_STEP
)


# ============================================================================
# Hilfsfunktionen
# ============================================================================


def _feature_vector(
    observation: Any,
) -> Optional[np.ndarray]:
    """
    Erzeugt den ML-Feature-Vektor ohne Imputation.

    Enthält mindestens ein Merkmal NaN oder Inf, wird None zurückgegeben.
    """

    values = []

    for feature in FEATURES:
        value = getattr(
            observation,
            feature,
            np.nan,
        )

        if not np.isfinite(value):
            return None

        values.append(
            float(value)
        )

    return np.asarray(
        values,
        dtype=float,
    )


def _psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> Optional[float]:
    """
    Berechnet den Population Stability Index.

    PSI bewertet ausschließlich die Änderung der Eingangsverteilung.
    Es ist daher ein Data-Drift-Signal und kein Concept-Drift-Signal.
    """

    reference = reference[
        np.isfinite(reference)
    ]

    current = current[
        np.isfinite(current)
    ]

    if (
        len(reference) < MIN_WINDOW_SIZE
        or len(current) < MIN_WINDOW_SIZE
    ):
        return None

    edges = np.unique(
        np.quantile(
            reference,
            np.linspace(
                0.0,
                1.0,
                bins + 1,
            ),
        )
    )

    if len(edges) < 3:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(
        reference,
        bins=edges,
    )

    cur_counts, _ = np.histogram(
        current,
        bins=edges,
    )

    eps = 1e-6

    ref_ratio = np.maximum(
        ref_counts / max(
            ref_counts.sum(),
            1,
        ),
        eps,
    )

    cur_ratio = np.maximum(
        cur_counts / max(
            cur_counts.sum(),
            1,
        ),
        eps,
    )

    return float(
        np.sum(
            (cur_ratio - ref_ratio)
            * np.log(
                cur_ratio / ref_ratio
            )
        )
    )


def _statistical_test(
    previous: Dict[str, np.ndarray],
    current: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """
    Berechnet PSI und KS featureweise.

    Die Funktion erzeugt ausschließlich statistische Data-Drift-Signale.
    """

    features = {}

    psi_features = []
    ks_features = []
    combined_features = []

    for feature in FEATURES:
        ref = previous[feature]
        cur = current[feature]

        psi_value = _psi(
            ref,
            cur,
        )

        if (
            len(ref) >= MIN_WINDOW_SIZE
            and len(cur) >= MIN_WINDOW_SIZE
        ):
            _, ks_pvalue = ks_2samp(
                ref,
                cur,
                method="asymp",
            )

            ks_pvalue = float(
                ks_pvalue
            )
        else:
            ks_pvalue = None

        psi_trigger = (
            psi_value is not None
            and psi_value > PSI_THRESHOLD
        )

        ks_trigger = (
            ks_pvalue is not None
            and ks_pvalue < KS_ALPHA
        )

        combined_trigger = (
            psi_trigger
            and ks_trigger
        )

        if psi_trigger:
            psi_features.append(
                feature
            )

        if ks_trigger:
            ks_features.append(
                feature
            )

        if combined_trigger:
            combined_features.append(
                feature
            )

        features[feature] = {
            "psi": psi_value,
            "ks_pvalue": ks_pvalue,
            "psi_trigger": psi_trigger,
            "ks_trigger": ks_trigger,
            "combined_trigger": combined_trigger,
        }

    return {
        "features": features,
        "psi_detected": bool(
            psi_features
        ),
        "ks_detected": bool(
            ks_features
        ),
        "combined_detected": bool(
            combined_features
        ),
        "psi_features": psi_features,
        "ks_features": ks_features,
        "combined_features": combined_features,
    }


def _binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Berechnet Precision, Recall und F1 für Anomalie gegen Normal.
    """

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true > 0,
            y_pred > 0,
            average="binary",
            zero_division=0,
        )
    )

    normal_mask = (
        y_true == 0
    )

    if normal_mask.any():
        fpr = float(
            np.mean(
                y_pred[normal_mask] > 0
            )
        )
    else:
        fpr = 0.0

    return {
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1_score": float(
            f1
        ),
        "false_positive_rate_normal": fpr,
        "detected_observations": int(
            np.sum(
                y_pred > 0
            )
        ),
    }


def _first_detection(
    steps: np.ndarray,
    predictions: np.ndarray,
    start: int,
    end: int,
) -> Optional[int]:
    """
    Findet die erste Detektion innerhalb eines Szenarios.
    """

    mask = (
        (steps >= start)
        & (steps < end)
        & (predictions > 0)
    )

    candidates = steps[mask]

    if len(candidates) == 0:
        return None

    return int(
        candidates[0]
    )


def _delay(
    steps: np.ndarray,
    predictions: np.ndarray,
    scenario: str,
    start: int,
    end: int,
) -> Dict[str, Any]:
    """
    Berechnet die Detection Delay in Beobachtungen.
    """

    detection = _first_detection(
        steps,
        predictions,
        start,
        end,
    )

    if detection is None:
        return {
            "scenario": scenario,
            "start_step": int(start),
            "detection_step": None,
            "delay_observations": None,
            "detected": False,
        }

    return {
        "scenario": scenario,
        "start_step": int(start),
        "detection_step": detection,
        "delay_observations": int(
            detection - start
        ),
        "detected": True,
    }


def _event_log(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Erzeugt ein kompaktes Ereignisprotokoll für erkannte Signale.
    """

    events = []

    methods = {
        "Isolation Forest": "if_pred",
        "One-Class SVM": "svm_pred",
        "PSI": "psi_pred",
        "KS": "ks_pred",
        "PSI+KS": "statistical_pred",
        "Concept Drift (Performance)": "concept_drift_pred",
    }

    for method, column in methods.items():

        if column not in df.columns:
            continue

        detected = df[
            df[column] == 1
        ]

        for _, row in detected.iterrows():

            events.append(
                {
                    "step": int(
                        row["step"]
                    ),
                    "timestamp": row[
                        "timestamp"
                    ],
                    "detector": method,
                    "ground_truth_scenario": row[
                        "scenario"
                    ],
                    "if_score": row.get(
                        "if_score",
                        np.nan,
                    ),
                    "svm_score": row.get(
                        "svm_score",
                        np.nan,
                    ),
                    "cpu": row.get(
                        "cpu",
                        np.nan,
                    ),
                    "memory": row.get(
                        "memory",
                        np.nan,
                    ),
                    "restarts": row.get(
                        "restarts",
                        0,
                    ),
                    "prediction_error": row.get(
                        "prediction_error",
                        np.nan,
                    ),
                    "rolling_prediction_error": row.get(
                        "rolling_prediction_error",
                        np.nan,
                    ),
                }
            )

    return pd.DataFrame(
        events
    )


# ============================================================================
# Hauptbenchmark
# ============================================================================


def run_benchmark() -> Dict[str, Any]:
    """
    Führt den vollständigen finalen Benchmark batchbasiert aus.

    Die einzelnen Verfahren werden entsprechend ihrer methodischen
    Zielsetzung separat bewertet:

        IF / OCSVM
            Punktanomalie gegen den Systemzustand.

        PSI / KS / PSI+KS
            Data Drift.

        Concept Drift (Performance)
            Veränderung der Beziehung X -> Y anhand der
            Vorhersageleistung des eingefrorenen Zielmodells.
    """

    print("=" * 80)
    print(
        "FINALER ANOMALIE-DETEKTIONS-BENCHMARK V16"
    )
    print("=" * 80)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # 1. Datenstrom
    # ------------------------------------------------------------------------

    print(
        "\n[1/6] Initialisiere Elec2-Datenstrom ..."
    )

    ingestor = create_ingestor()

    print(
        f"Datensatz: {ingestor.total_rows}"
    )

    print(
        f"Testbeginn: {TEST_START}"
    )

    # Der Benchmark verwendet exakt dieselbe Szenariologik wie der
    # kontinuierliche Experimentstrom.
    #
    # include_concept_drift=True aktiviert die explizite Concept-Drift-Phase.
    # live_demo=True sorgt zusätzlich für deterministische, eindeutig
    # unterscheidbare Infrastrukturindikatoren.
    observations = [
        ingestor._create_observation(
            step,
            include_concept_drift=True,
            live_demo=True,
        )
        for step in range(
            ingestor.total_rows
        )
    ]

    # ------------------------------------------------------------------------
    # 2. Training und Kalibrierung
    # ------------------------------------------------------------------------

    print(
        "\n[2/6] Erzeuge Training und Kalibrierung ..."
    )

    train_vectors = []
    train_targets = []

    calibration_vectors = []
    calibration_targets = []

    for observation in observations[
        :REFERENCE_SIZE
    ]:

        vector = _feature_vector(
            observation
        )

        if vector is not None:

            train_vectors.append(
                vector
            )

            train_targets.append(
                str(
                    observation.target
                )
            )

    for observation in observations[
        REFERENCE_SIZE:TEST_START
    ]:

        vector = _feature_vector(
            observation
        )

        if vector is not None:

            calibration_vectors.append(
                vector
            )

            calibration_targets.append(
                str(
                    observation.target
                )
            )

    X_train = np.asarray(
        train_vectors,
        dtype=float,
    )

    X_calibration = np.asarray(
        calibration_vectors,
        dtype=float,
    )

    y_train = np.asarray(
        train_targets,
        dtype=str,
    )

    y_calibration = np.asarray(
        calibration_targets,
        dtype=str,
    )

    print(
        f"Training: {len(X_train)}"
    )

    print(
        f"Kalibrierung: {len(X_calibration)}"
    )

    # ------------------------------------------------------------------------
    # 3. Modelle trainieren
    # ------------------------------------------------------------------------

    print(
        "\n[3/6] Trainiere ML-Detektoren ..."
    )

    isolation_forest = IsolationForest(
        n_estimators=IF_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    isolation_forest.fit(
        X_train
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    svm = OneClassSVM(
        kernel=OCSVM_KERNEL,
        nu=OCSVM_NU,
        gamma=OCSVM_GAMMA,
    )

    svm.fit(
        X_train_scaled
    )

    # ------------------------------------------------------------------------
    # Eingefrorenes Zielmodell
    # ------------------------------------------------------------------------

    # Das Modell wird ausschließlich auf der gesunden Referenz trainiert.
    # Während des Tests wird es nicht nachtrainiert.
    #
    # Dadurch kann eine spätere Verschlechterung der Vorhersageleistung
    # als Indikator für Concept Drift verwendet werden.

    target_model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    target_model.fit(
        X_train,
        y_train,
    )

    calibration_target_predictions = (
        target_model.predict(
            X_calibration
        ).astype(str)
    )

    calibration_target_error = (
        calibration_target_predictions
        != y_calibration
    ).astype(float)

    performance_baseline_error = float(
        np.mean(
            calibration_target_error
        )
    )

    # ------------------------------------------------------------------------
    # Kalibrierung der Punktanomalie-Detektoren
    # ------------------------------------------------------------------------

    if_calibration_scores = (
        isolation_forest.decision_function(
            X_calibration
        )
    )

    svm_calibration_scores = (
        svm.decision_function(
            scaler.transform(
                X_calibration
            )
        )
    )

    if_threshold = float(
        np.quantile(
            if_calibration_scores,
            CALIBRATION_QUANTILE,
        )
    )

    svm_threshold = float(
        np.quantile(
            svm_calibration_scores,
            CALIBRATION_QUANTILE,
        )
    )

    print(
        f"IF Threshold: {if_threshold:.8f}"
    )

    print(
        f"OCSVM Threshold: {svm_threshold:.8f}"
    )

    print(
        f"Performance-Baseline: "
        f"{performance_baseline_error:.6f}"
    )

    print(
        f"Performance-Schwelle: "
        f"{PERFORMANCE_ABSOLUTE_THRESHOLD:.6f}"
    )

    print(
        f"Performance-Bestätigung: "
        f"{PERFORMANCE_CONFIRMATION_WINDOWS}"
    )

    # ------------------------------------------------------------------------
    # 4. Testdaten vorbereiten
    # ------------------------------------------------------------------------

    print(
        "\n[4/6] Bereite Testdaten batchweise vor ..."
    )

    test_observations = observations[
        TEST_START:
    ]

    # Die Testdaten enthalten bereits die vollständige, aktuelle
    # Szenarioabfolge des Elec2StreamIngestors:
    #
    #     NORMAL
    #         -> DATA_DRIFT
    #         -> CONCEPT_DRIFT
    #         -> NORMAL
    #         -> INFRASTRUCTURE_FAILURE
    #         -> NORMAL
    #
    # Die Concept-Drift-Phase wird nicht nachträglich über einen zweiten
    # künstlichen Bereich injiziert. Sie wird direkt durch
    # include_concept_drift=True in _create_observation() erzeugt.
    #
    # Dadurch stimmen Ground Truth und produktiver Experimentstrom
    # methodisch überein. Während der Concept-Drift-Phase bleiben die
    # Eingangsmerkmale X unverändert, während sich die Zielbeziehung X -> Y
    # verändert.

    rows = []

    valid_vectors = []
    valid_indices = []

    for local_index, observation in enumerate(
        test_observations
    ):

        vector = _feature_vector(
            observation
        )

        row = {
            "step": int(
                observation.step
            ),
            "timestamp": float(
                observation.timestamp
            ),
            "scenario": str(
                observation.scenario
            ),
            "classification": int(
                observation.classification
            ),
            "target_true": str(
                observation.target
            ),
            "system_state": int(
                getattr(
                    observation,
                    "system_state",
                    0,
                )
            ),
            "anomaly_type": str(
                getattr(
                    observation,
                    "anomaly_type",
                    "NONE",
                )
            ),
            "root_cause": str(
                getattr(
                    observation,
                    "root_cause",
                    "",
                )
            ),
            "trigger_feature": str(
                getattr(
                    observation,
                    "trigger_feature",
                    "",
                )
            ),
            "nswprice": float(
                observation.nswprice
            ),
            "nswdemand": float(
                observation.nswdemand
            ),
            "vicprice": float(
                observation.vicprice
            ),
            "vicdemand": float(
                observation.vicdemand
            ),
            "transfer": float(
                observation.transfer
            ),
            "cpu": float(
                observation.infrastructure_cpu_load
            ),
            "memory": float(
                observation.infrastructure_memory_load
            ),
            "restarts": int(
                observation.infrastructure_restarts
            ),
            "ml_valid": int(
                vector is not None
            ),
        }

        rows.append(
            row
        )

        if vector is not None:

            valid_vectors.append(
                vector
            )

            valid_indices.append(
                local_index
            )

    X_valid = np.asarray(
        valid_vectors,
        dtype=float,
    )

    print(
        f"Testbeobachtungen: "
        f"{len(test_observations)}"
    )

    print(
        f"ML-valid: {len(X_valid)}"
    )

    print(
        f"ML-invalid: "
        f"{len(test_observations) - len(X_valid)}"
    )

    # ------------------------------------------------------------------------
    # 5. ML-Detektoren
    # ------------------------------------------------------------------------

    print(
        "\n[5/6] Führe Detektoren batchweise aus ..."
    )

    if_pred = np.zeros(
        len(test_observations),
        dtype=int,
    )

    if_score = np.full(
        len(test_observations),
        np.nan,
    )

    svm_pred = np.zeros(
        len(test_observations),
        dtype=int,
    )

    svm_score = np.full(
        len(test_observations),
        np.nan,
    )

    target_pred = np.full(
        len(test_observations),
        None,
        dtype=object,
    )

    target_probability = np.full(
        len(test_observations),
        np.nan,
    )

    # ------------------------------------------------------------------------
    # Isolation Forest
    # ------------------------------------------------------------------------

    if_start_time = time.perf_counter()

    if len(X_valid):

        if_scores_valid = (
            isolation_forest.decision_function(
                X_valid
            )
        )

        if_predictions_valid = (
            if_scores_valid
            < if_threshold
        ).astype(int)

        valid_indices_array = np.asarray(
            valid_indices,
            dtype=int,
        )

        if_score[
            valid_indices_array
        ] = if_scores_valid

        if_pred[
            valid_indices_array
        ] = if_predictions_valid

    if_elapsed_ms = (
        time.perf_counter()
        - if_start_time
    ) * 1000.0

    # ------------------------------------------------------------------------
    # One-Class SVM
    # ------------------------------------------------------------------------

    svm_start_time = time.perf_counter()

    if len(X_valid):

        X_valid_scaled = scaler.transform(
            X_valid
        )

        svm_scores_valid = (
            svm.decision_function(
                X_valid_scaled
            )
        )

        svm_predictions_valid = (
            svm_scores_valid
            < svm_threshold
        ).astype(int)

        svm_score[
            valid_indices_array
        ] = svm_scores_valid

        svm_pred[
            valid_indices_array
        ] = svm_predictions_valid

    svm_elapsed_ms = (
        time.perf_counter()
        - svm_start_time
    ) * 1000.0

    # ------------------------------------------------------------------------
    # Eingefrorenes Zielmodell
    # ------------------------------------------------------------------------

    if len(X_valid):

        target_pred_valid = (
            target_model.predict(
                X_valid
            ).astype(str)
        )

        target_pred[
            valid_indices_array
        ] = target_pred_valid

        if hasattr(
            target_model,
            "predict_proba",
        ):

            target_prob_valid = np.max(
                target_model.predict_proba(
                    X_valid
                ),
                axis=1,
            )

            target_probability[
                valid_indices_array
            ] = target_prob_valid

    # ------------------------------------------------------------------------
    # ML-Ergebnisse in DataFrame schreiben
    # ------------------------------------------------------------------------

    for index, row in enumerate(
        rows
    ):

        row["if_score"] = (
            float(
                if_score[index]
            )
            if np.isfinite(
                if_score[index]
            )
            else np.nan
        )

        row["if_pred"] = int(
            if_pred[index]
        )

        row["svm_score"] = (
            float(
                svm_score[index]
            )
            if np.isfinite(
                svm_score[index]
            )
            else np.nan
        )

        row["svm_pred"] = int(
            svm_pred[index]
        )

        row["target_pred"] = (
            str(
                target_pred[index]
            )
            if target_pred[index]
            is not None
            else None
        )

        row["target_probability"] = (
            float(
                target_probability[index]
            )
            if np.isfinite(
                target_probability[index]
            )
            else np.nan
        )

        row["prediction_error"] = (
            int(
                str(
                    row["target_pred"]
                )
                != str(
                    row["target_true"]
                )
            )
            if row["target_pred"]
            is not None
            else np.nan
        )

    # ------------------------------------------------------------------------
    # Statistische Verfahren
    # ------------------------------------------------------------------------

    print(
        "\n[6/6] Führe PSI/KS-Auswertung durch ..."
    )

    statistical_start_time = (
        time.perf_counter()
    )

    psi_pred = np.zeros(
        len(test_observations),
        dtype=int,
    )

    ks_pred = np.zeros(
        len(test_observations),
        dtype=int,
    )

    statistical_pred = np.zeros(
        len(test_observations),
        dtype=int,
    )

    statistical_feature = [
        ""
        for _ in range(
            len(test_observations)
        )
    ]

    psi_values = {
        feature: np.full(
            len(test_observations),
            np.nan,
        )
        for feature in FEATURES
    }

    ks_values = {
        feature: np.full(
            len(test_observations),
            np.nan,
        )
        for feature in FEATURES
    }

    events = []

    persistence = 0

    feature_matrix = np.empty(
        (
            len(test_observations),
            len(FEATURES),
        ),
        dtype=float,
    )

    for index, observation in enumerate(
        test_observations
    ):

        feature_matrix[index] = np.asarray(
            [
                observation.nswprice,
                observation.nswdemand,
                observation.vicprice,
                observation.vicdemand,
                observation.transfer,
            ],
            dtype=float,
        )

    for current_end in range(
        WINDOW_SIZE,
        len(test_observations) + 1,
        STATISTICAL_STEP,
    ):

        current_start = (
            current_end
            - STATISTICAL_STEP
        )

        previous_start = (
            current_end
            - 2 * WINDOW_SIZE
        )

        previous_end = (
            current_end
            - WINDOW_SIZE
        )

        if previous_start < 0:
            continue

        previous_matrix = feature_matrix[
            previous_start:previous_end
        ]

        current_matrix = feature_matrix[
            current_end - WINDOW_SIZE:current_end
        ]

        previous_values = {
            feature: previous_matrix[:, index]
            for index, feature in enumerate(
                FEATURES
            )
        }

        current_values = {
            feature: current_matrix[:, index]
            for index, feature in enumerate(
                FEATURES
            )
        }

        statistical = _statistical_test(
            previous_values,
            current_values,
        )

        psi_detected = bool(
            statistical["psi_detected"]
        )

        ks_detected = bool(
            statistical["ks_detected"]
        )

        combined_detected = bool(
            statistical["combined_detected"]
        )

        if combined_detected:
            persistence += 1
        else:
            persistence = 0

        persistent_detection = (
            persistence
            >= DRIFT_PERSISTENCE_WINDOWS
        )

        write_start = current_start
        write_end = current_end

        psi_pred[
            write_start:write_end
        ] = int(
            psi_detected
        )

        ks_pred[
            write_start:write_end
        ] = int(
            ks_detected
        )

        statistical_pred[
            write_start:write_end
        ] = int(
            persistent_detection
        )

        trigger_features = (
            statistical[
                "combined_features"
            ]
        )

        trigger_feature = (
            trigger_features[0]
            if trigger_features
            else ""
        )

        for feature in FEATURES:

            metrics = statistical[
                "features"
            ][feature]

            psi_values[
                feature
            ][
                write_start:write_end
            ] = (
                metrics["psi"]
                if metrics["psi"]
                is not None
                else np.nan
            )

            ks_values[
                feature
            ][
                write_start:write_end
            ] = (
                metrics["ks_pvalue"]
                if metrics["ks_pvalue"]
                is not None
                else np.nan
            )

        for index in range(
            write_start,
            write_end,
        ):

            statistical_feature[
                index
            ] = trigger_feature

        events.append(
            {
                "step_start": int(
                    test_observations[
                        write_start
                    ].step
                ),
                "step_end": int(
                    test_observations[
                        write_end - 1
                    ].step
                ),
                "psi_detected": int(
                    psi_detected
                ),
                "ks_detected": int(
                    ks_detected
                ),
                "combined_detected": int(
                    combined_detected
                ),
                "persistent_detection": int(
                    persistent_detection
                ),
                "trigger_feature": trigger_feature,
            }
        )

    statistical_elapsed_ms = (
        time.perf_counter()
        - statistical_start_time
    ) * 1000.0

    for index, row in enumerate(
        rows
    ):

        row["psi_pred"] = int(
            psi_pred[index]
        )

        row["ks_pred"] = int(
            ks_pred[index]
        )

        row["statistical_pred"] = int(
            statistical_pred[index]
        )

        row[
            "statistical_trigger_feature"
        ] = statistical_feature[index]

        for feature in FEATURES:

            row[
                f"psi_{feature}"
            ] = psi_values[
                feature
            ][index]

            row[
                f"ks_{feature}"
            ] = ks_values[
                feature
            ][index]

    df = pd.DataFrame(
        rows
    )

    # ------------------------------------------------------------------------
    # Leistungsbasierte Concept-Drift-Erkennung
    # ------------------------------------------------------------------------

    # WICHTIG:
    #
    # Diese Erkennung verwendet NICHT PSI oder KS.
    #
    # Die Concept-Drift-Entscheidung basiert ausschließlich auf:
    #
    #     beobachtetem Target Y
    #             +
    #     Vorhersage des eingefrorenen Modells
    #             +
    #     persistenter Verschlechterung
    #
    # Damit wird die Veränderung der Beziehung X -> Y bewertet.

    df["rolling_prediction_error"] = (
        pd.to_numeric(
            df["prediction_error"],
            errors="coerce",
        )
        .rolling(
            window=PERFORMANCE_WINDOW_SIZE,
            min_periods=PERFORMANCE_WINDOW_SIZE,
        )
        .mean()
    )

    # Die finale V16-Schwelle ist eingefroren.
    #
    # Der Schwellenwert wird NICHT anhand des Testbereichs angepasst.
    #
    # Die Baseline und die relative Degradation werden weiterhin
    # dokumentiert, die finale operative Entscheidung verwendet jedoch
    # den festgelegten absoluten Mindestwert von 0.65.

    performance_threshold = (
        PERFORMANCE_ABSOLUTE_THRESHOLD
    )

    df["performance_degraded_raw"] = (
        df["rolling_prediction_error"]
        > performance_threshold
    ).fillna(False)

    # ------------------------------------------------------------------------
    # Persistente Bestätigung
    # ------------------------------------------------------------------------

    confirmation = 0

    concept_flags = []

    confirmation_counts = []

    for degraded in (
        df["performance_degraded_raw"]
        .astype(bool)
    ):

        if degraded:
            confirmation += 1
        else:
            confirmation = 0

        concept_detected = (
            confirmation
            >= PERFORMANCE_CONFIRMATION_WINDOWS
        )

        concept_flags.append(
            concept_detected
        )

        confirmation_counts.append(
            confirmation
        )

    df["performance_degraded"] = (
        df["performance_degraded_raw"]
        .astype(int)
    )

    df[
        "performance_confirmation_count"
    ] = confirmation_counts

    # ------------------------------------------------------------------------
    # FINAL:
    # Concept Drift ist NICHT von PSI/KS abhängig.
    # ------------------------------------------------------------------------

    df["concept_drift_pred"] = (
        np.asarray(
            concept_flags,
            dtype=int,
        )
    )

    # ------------------------------------------------------------------------
    # Primäre Evaluation
    # ------------------------------------------------------------------------

    scenario_series = (
        df["scenario"].astype(str)
    )

    anomaly_type_series = (
        df["anomaly_type"].astype(str)
    )

    # Reiner Data Drift.
    #
    # Dieser Bereich dient zur Bewertung der Fähigkeit von PSI/KS,
    # eine Änderung der Eingangsverteilung zu erkennen.

    data_drift_mask = (
        scenario_series
        == "data_drift"
    )

    # Infrastructure Failure.
    infrastructure_mask = (
        anomaly_type_series
        == "INFRASTRUCTURE_FAILURE"
    )

    # Ground Truth für Concept Drift.
    concept_drift_truth = (
        anomaly_type_series
        == "ANOMALY_CONCEPT_DRIFT"
    )

    # Operativer Systemzustand.
    #
    # Reiner Data Drift bleibt NORMAL.

    y_true = df[
        "system_state"
    ].to_numpy(
        dtype=int
    )

    # ------------------------------------------------------------------------
    # Rohsignale der einzelnen Verfahren
    # ------------------------------------------------------------------------

    if_prediction = df[
        "if_pred"
    ].to_numpy(
        dtype=int
    )

    svm_prediction = df[
        "svm_pred"
    ].to_numpy(
        dtype=int
    )

    psi_signal = df[
        "psi_pred"
    ].to_numpy(
        dtype=int
    )

    ks_signal = df[
        "ks_pred"
    ].to_numpy(
        dtype=int
    )

    statistical_signal = df[
        "statistical_pred"
    ].to_numpy(
        dtype=int
    )

    concept_performance_signal = (
        df[
            "concept_drift_pred"
        ].to_numpy(
            dtype=int
        )
    )

    predictions = {
        "Isolation Forest": if_prediction,
        "One-Class SVM": svm_prediction,
        "PSI": psi_signal,
        "KS": ks_signal,
        "PSI+KS": statistical_signal,
        "Concept Drift (Performance)": (
            concept_performance_signal
        ),
    }

    # ------------------------------------------------------------------------
    # Methodenspezifische Metriken
    # ------------------------------------------------------------------------

    method_metrics = {}

    for method, prediction in (
        predictions.items()
    ):

        # --------------------------------------------------------------
        # PSI / KS
        # --------------------------------------------------------------

        if method in {
            "PSI",
            "KS",
            "PSI+KS",
        }:

            method_metrics[
                method
            ] = {
                "data_drift_metrics": (
                    _binary_metrics(
                        data_drift_mask.to_numpy(
                            dtype=int
                        ),
                        prediction,
                    )
                )
            }

        # --------------------------------------------------------------
        # Concept Drift
        # --------------------------------------------------------------

        elif method == (
            "Concept Drift (Performance)"
        ):

            method_metrics[
                method
            ] = _binary_metrics(
                concept_drift_truth.to_numpy(
                    dtype=int
                ),
                prediction,
            )

        # --------------------------------------------------------------
        # Isolation Forest / One-Class SVM
        # --------------------------------------------------------------

        else:

            method_metrics[
                method
            ] = _binary_metrics(
                y_true,
                prediction,
            )

    # ------------------------------------------------------------------------
    # Szenariometriken
    # ------------------------------------------------------------------------

    scenario_metrics = {}

    for method, prediction in (
        predictions.items()
    ):

        scenario_metrics[
            method
        ] = {

            "data_drift": {
                "recall": (
                    float(
                        np.mean(
                            prediction[
                                data_drift_mask.to_numpy()
                            ]
                            > 0
                        )
                    )
                    if data_drift_mask.any()
                    else 0.0
                ),
                "support": int(
                    data_drift_mask.sum()
                ),
            },

            "concept_drift": {
                "recall": (
                    float(
                        np.mean(
                            prediction[
                                concept_drift_truth.to_numpy()
                            ]
                            > 0
                        )
                    )
                    if concept_drift_truth.any()
                    else 0.0
                ),
                "support": int(
                    concept_drift_truth.sum()
                ),
            },

            "infrastructure_failure": {
                "recall": (
                    float(
                        np.mean(
                            prediction[
                                infrastructure_mask.to_numpy()
                            ]
                            > 0
                        )
                    )
                    if infrastructure_mask.any()
                    else 0.0
                ),
                "support": int(
                    infrastructure_mask.sum()
                ),
            },

            "data_drift_false_positive_rate": (
                float(
                    np.mean(
                        prediction[
                            data_drift_mask.to_numpy()
                        ]
                        > 0
                    )
                )
                if data_drift_mask.any()
                else 0.0
            ),
        }

    # ------------------------------------------------------------------------
    # Detection Delay
    # ------------------------------------------------------------------------

    steps = df[
        "step"
    ].to_numpy(
        dtype=int
    )

    concept_steps = steps[
        concept_drift_truth.to_numpy()
    ]

    concept_start_step = (
        int(
            concept_steps.min()
        )
        if len(concept_steps)
        else int(
            ingestor.data_drift_start
        )
    )

    concept_end_step = (
        int(
            concept_steps.max()
        ) + 1
        if len(concept_steps)
        else int(
            ingestor.data_drift_end
        )
    )

    boundaries = {

        "data_drift": (
            int(
                ingestor.data_drift_start
            ),
            int(
                ingestor.data_drift_end
            ),
        ),

        "concept_drift": (
            concept_start_step,
            concept_end_step,
        ),

        "infrastructure_failure": (
            int(
                ingestor.infrastructure_start
            ),
            int(
                ingestor.infrastructure_end
            ),
        ),
    }

    delay_rows = []

    detection_delay = {}

    for method, prediction in (
        predictions.items()
    ):

        detection_delay[
            method
        ] = {}

        for scenario, (
            start,
            end,
        ) in boundaries.items():

            # ----------------------------------------------------------
            # Data Drift
            # ----------------------------------------------------------

            if scenario == "data_drift":

                if method in {
                    "PSI",
                    "KS",
                    "PSI+KS",
                }:

                    delay_prediction = (
                        prediction
                    )

                else:

                    # IF und OCSVM sind keine statistischen
                    # Data-Drift-Verfahren.
                    #
                    # Die Performance-Methode ist ebenfalls kein
                    # Data-Drift-Verfahren.
                    #
                    # Deshalb wird für diese Kombination keine
                    # Detection Delay ausgewiesen.

                    delay_prediction = (
                        np.zeros_like(
                            prediction
                        )
                    )

            # ----------------------------------------------------------
            # Concept Drift
            # ----------------------------------------------------------

            elif scenario == "concept_drift":

                if method == (
                    "Concept Drift (Performance)"
                ):

                    delay_prediction = (
                        concept_performance_signal
                    )

                else:

                    # PSI, KS, IF und OCSVM werden nicht
                    # als Concept-Drift-Verfahren gewertet.
                    #
                    # Die Detection Delay dieser Kombinationen
                    # ist daher methodisch nicht anwendbar.

                    delay_prediction = (
                        np.zeros_like(
                            prediction
                        )
                    )

            # ----------------------------------------------------------
            # Infrastructure Failure
            # ----------------------------------------------------------

            else:

                delay_prediction = (
                    prediction
                )

            result = _delay(
                steps,
                delay_prediction,
                scenario,
                start,
                end,
            )

            detection_delay[
                method
            ][scenario] = result

            delay_rows.append(
                {
                    "method": method,
                    **result,
                }
            )

    delay_df = pd.DataFrame(
        delay_rows
    )

    # ------------------------------------------------------------------------
    # Root-Cause-Auswertung
    # ------------------------------------------------------------------------

    # Die RCA verwendet ausschließlich beobachtbare Werte.
    #
    # Die Ground Truth wird nicht zur Berechnung der RCA verwendet.

    rca_predictions = []

    for _, row in df.iterrows():

        if row["restarts"] >= 1:

            rca = "POD_RESTART"

        elif row["cpu"] >= 90.0:

            rca = "HIGH_CPU"

        elif row["memory"] >= 90.0:

            rca = "HIGH_MEMORY"

        elif row[
            "concept_drift_pred"
        ] == 1:

            rca = "CONCEPT_DRIFT"

        elif row[
            "statistical_pred"
        ] == 1:

            rca = "DATA_DRIFT"

        else:

            rca = "NORMAL"

        rca_predictions.append(
            rca
        )

    df[
        "rca_prediction"
    ] = rca_predictions

    # ------------------------------------------------------------------------
    # Ergebnisdateien
    # ------------------------------------------------------------------------

    df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    delay_df.to_csv(
        DELAY_FILE,
        index=False,
    )

    events_df = _event_log(
        df
    )

    events_df.to_csv(
        EVENTS_FILE,
        index=False,
    )

    # ------------------------------------------------------------------------
    # Szenarioverteilung
    # ------------------------------------------------------------------------

    scenario_distribution = {
        str(key): int(value)
        for key, value in (
            df[
                "classification"
            ]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    # ------------------------------------------------------------------------
    # Zusammenfassung
    # ------------------------------------------------------------------------

    summary = {

        "benchmark_version": "FINAL_V16",

        "dataset_size": int(
            ingestor.total_rows
        ),

        "evaluated_observations": int(
            len(df)
        ),

        "ml_valid_observations": int(
            df["ml_valid"].sum()
        ),

        "ml_invalid_observations": int(
            (
                df["ml_valid"]
                == 0
            ).sum()
        ),

        "reference_size": (
            REFERENCE_SIZE
        ),

        "calibration_size": (
            CALIBRATION_SIZE
        ),

        "test_start": (
            TEST_START
        ),

        "window_size": (
            WINDOW_SIZE
        ),

        "min_window_size": (
            MIN_WINDOW_SIZE
        ),

        "drift_persistence_windows": (
            DRIFT_PERSISTENCE_WINDOWS
        ),

        "statistical_step": (
            STATISTICAL_STEP
        ),

        "psi_threshold": (
            PSI_THRESHOLD
        ),

        "ks_alpha": (
            KS_ALPHA
        ),

        "ks_method": "asymp",

        "calibration_quantile": (
            CALIBRATION_QUANTILE
        ),

        "classes": {

            "system_state": {
                "0": "Normal",
                "1": "Anomaly",
            },

            "anomaly_types": [
                "NONE",
                "ANOMALY_CONCEPT_DRIFT",
                "INFRASTRUCTURE_FAILURE",
            ],
        },

        "methods": list(
            predictions.keys()
        ),

        "method_scope": {
            "Isolation Forest": "system_state_anomaly",
            "One-Class SVM": "system_state_anomaly",
            "PSI": "data_drift",
            "KS": "data_drift",
            "PSI+KS": "data_drift",
            "Concept Drift (Performance)": "concept_drift",
        },

        "thresholds": {

            "Isolation Forest": (
                if_threshold
            ),

            "One-Class SVM": (
                svm_threshold
            ),
        },

        "performance_monitoring": {

            "window_size": (
                PERFORMANCE_WINDOW_SIZE
            ),

            "baseline_error": (
                performance_baseline_error
            ),

            "degradation_factor": (
                PERFORMANCE_DEGRADATION_FACTOR
            ),

            "degradation_margin": (
                PERFORMANCE_DEGRADATION_MARGIN
            ),

            "absolute_threshold": (
                PERFORMANCE_ABSOLUTE_THRESHOLD
            ),

            "confirmation_windows": (
                PERFORMANCE_CONFIRMATION_WINDOWS
            ),

            "degradation_threshold": (
                performance_threshold
            ),
        },

        "scenario_distribution": (
            scenario_distribution
        ),

        "ground_truth": {

            "normal": int(
                np.sum(
                    y_true == 0
                )
            ),

            "anomaly": int(
                np.sum(
                    y_true == 1
                )
            ),

            "data_drift": int(
                data_drift_mask.sum()
            ),

            "concept_drift": int(
                concept_drift_truth.sum()
            ),

            "infrastructure_failure": int(
                infrastructure_mask.sum()
            ),
        },

        "method_metrics": (
            method_metrics
        ),

        "scenario_metrics": (
            scenario_metrics
        ),

        "detection_delay": (
            detection_delay
        ),

        "latency_ms": {

            "Isolation Forest": {

                "batch_total_ms": float(
                    if_elapsed_ms
                ),

                "mean_ms_per_valid_observation": (
                    float(
                        if_elapsed_ms
                        / len(X_valid)
                    )
                    if len(X_valid)
                    else 0.0
                ),
            },

            "One-Class SVM": {

                "batch_total_ms": float(
                    svm_elapsed_ms
                ),

                "mean_ms_per_valid_observation": (
                    float(
                        svm_elapsed_ms
                        / len(X_valid)
                    )
                    if len(X_valid)
                    else 0.0
                ),
            },

            "Statistical (PSI/KS)": {

                "batch_total_ms": float(
                    statistical_elapsed_ms
                ),

                "mean_ms_per_observation": (
                    float(
                        statistical_elapsed_ms
                        / len(
                            test_observations
                        )
                    )
                    if len(
                        test_observations
                    )
                    else 0.0
                ),

                "evaluation_step": (
                    STATISTICAL_STEP
                ),

                "ks_method": "asymp",
            },
        },

        "output_files": {

            "predictions": (
                PREDICTIONS_FILE.name
            ),

            "summary": (
                SUMMARY_FILE.name
            ),

            "events": (
                EVENTS_FILE.name
            ),

            "delay": (
                DELAY_FILE.name
            ),
        },
    }

    # ------------------------------------------------------------------------
    # JSON speichern
    # ------------------------------------------------------------------------

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------------
    # Konsolenausgabe
    # ------------------------------------------------------------------------

    print(
        "\nBenchmark abgeschlossen."
    )

    print(
        "\nMethodenergebnisse:"
    )

    for name, metrics in (
        method_metrics.items()
    ):

        if "precision" in metrics:

            print(
                f"{name:<32} "
                f"P={metrics['precision']:.4f} "
                f"R={metrics['recall']:.4f} "
                f"F1={metrics['f1_score']:.4f} "
                f"FPR={metrics['false_positive_rate_normal']:.4f}"
            )

        elif "data_drift_metrics" in metrics:

            drift_metrics = (
                metrics[
                    "data_drift_metrics"
                ]
            )

            print(
                f"{name:<32} "
                f"P={drift_metrics['precision']:.4f} "
                f"R={drift_metrics['recall']:.4f} "
                f"F1={drift_metrics['f1_score']:.4f} "
                f"FPR={drift_metrics['false_positive_rate_normal']:.4f}"
            )

    print(
        "\nConcept-Drift-Konfiguration:"
    )

    print(
        f"  Fenstergröße: "
        f"{PERFORMANCE_WINDOW_SIZE}"
    )

    print(
        f"  Baseline Error: "
        f"{performance_baseline_error:.6f}"
    )

    print(
        f"  Degradation Factor: "
        f"{PERFORMANCE_DEGRADATION_FACTOR}"
    )

    print(
        f"  Degradation Margin: "
        f"{PERFORMANCE_DEGRADATION_MARGIN}"
    )

    print(
        f"  Absoluter Schwellenwert: "
        f"{PERFORMANCE_ABSOLUTE_THRESHOLD}"
    )

    print(
        f"  Bestätigungsfenster: "
        f"{PERFORMANCE_CONFIRMATION_WINDOWS}"
    )

    print(
        f"\nErgebnisse gespeichert in: "
        f"{RESULTS_DIR}"
    )

    return summary


# ============================================================================
# Programmeinstieg
# ============================================================================

if __name__ == "__main__":
    run_benchmark()