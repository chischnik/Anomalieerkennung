"""
Prometheus-Metriken für den Kubernetes-ML-Anomalie-Detektor.

Die Metriken trennen bewusst:

1. Systemklassifikation:
   NORMAL / ANOMALY

2. Leistungsbasiertes Concept-Drift-Monitoring:
   Vorhersagefehler, gleitender Fehler, Baseline, Schwellenwert
   und Bestätigung des Concept Drift.

3. Data-Drift-Monitoring:
   PSI und KS pro Feature. Data Drift ist kein Anomaliezustand.

4. Infrastruktur:
   CPU, Speicher, Pod-Restarts, Datenqualität, HTTP und Latenz.

5. Root-Cause-/Trigger-Informationen:
   erkannte Ursache und verantwortliches Feature.

Die Point-Anomaly-Metriken bleiben aus Kompatibilitätsgründen erhalten,
sind aber nicht Bestandteil der finalen Systemklassifikation.

Die Ground Truth des Experiments wird hier nicht als Laufzeitmetrik verwendet.
"""

from typing import Mapping, Optional

from prometheus_client import Counter, Gauge, Histogram


# ============================================================================
# Namespace
# ============================================================================

METRIC_NAMESPACE = "ml_anomaly"


# ============================================================================
# Statistische Metriken
# ============================================================================

RATE_OF_CHANGE = Gauge(
    "rate_of_change",
    "Momentane Änderungsrate des überwachten Signals.",
    namespace=METRIC_NAMESPACE,
)

PEAK_DENSITY = Gauge(
    "peak_density",
    "Dichte erkannter Peaks im überwachten Signal.",
    namespace=METRIC_NAMESPACE,
)

KS_PVALUE = Gauge(
    "ks_pvalue",
    "KS-p-value des überwachten Signals.",
    namespace=METRIC_NAMESPACE,
)

KS_STATISTIC = Gauge(
    "ks_statistic",
    "KS-Statistik des überwachten Signals.",
    namespace=METRIC_NAMESPACE,
)

ROLLING_MEAN = Gauge(
    "rolling_mean",
    "Gleitender Mittelwert.",
    namespace=METRIC_NAMESPACE,
)

ROLLING_STD = Gauge(
    "rolling_std",
    "Gleitende Standardabweichung.",
    namespace=METRIC_NAMESPACE,
)

STATISTICAL_SCORE = Gauge(
    "statistical_score",
    "Robuster statistischer Score.",
    namespace=METRIC_NAMESPACE,
)

Z_SCORE_THRESHOLD_METRIC = Gauge(
    "z_score_threshold",
    "Konfigurierter Schwellenwert des Z-Scores.",
    namespace=METRIC_NAMESPACE,
)


# ============================================================================
# Data-Drift-Monitoring
# ============================================================================

DRIFT_PSI = Gauge(
    "drift_psi",
    "Population Stability Index pro Feature.",
    namespace=METRIC_NAMESPACE,
    labelnames=("feature",),
)

DRIFT_KS_PVALUE = Gauge(
    "drift_ks_pvalue",
    "KS-p-value zwischen Referenz und aktuellem Fenster.",
    namespace=METRIC_NAMESPACE,
    labelnames=("feature",),
)

DRIFT_KS_STATISTIC = Gauge(
    "drift_ks_statistic",
    "KS-Statistik zwischen Referenz und aktuellem Fenster.",
    namespace=METRIC_NAMESPACE,
    labelnames=("feature",),
)

DRIFT_FEATURE_ANOMALY = Gauge(
    "drift_feature_anomaly",
    "Data-Drift-Status des Features: 1 = Drift, 0 = kein Drift.",
    namespace=METRIC_NAMESPACE,
    labelnames=("feature",),
)


# ============================================================================
# ML-Detektoren
# ============================================================================

ISOLATION_FOREST_SCORE = Gauge(
    "isolation_forest_score",
    "Score des Isolation Forest.",
    namespace=METRIC_NAMESPACE,
)

ONE_CLASS_SVM_DISTANCE = Gauge(
    "one_class_svm_distance",
    "Score bzw. Distanz der One-Class SVM.",
    namespace=METRIC_NAMESPACE,
)

ISOLATION_FOREST_ANOMALY = Gauge(
    "isolation_forest_anomaly",
    "Diagnostische Point-Anomaly-Entscheidung des Isolation Forest.",
    namespace=METRIC_NAMESPACE,
)

ONE_CLASS_SVM_ANOMALY = Gauge(
    "one_class_svm_anomaly",
    "Diagnostische Point-Anomaly-Entscheidung der One-Class SVM.",
    namespace=METRIC_NAMESPACE,
)

ISOLATION_FOREST_THRESHOLD = Gauge(
    "isolation_forest_threshold",
    "Kalibrierter Isolation-Forest-Schwellenwert.",
    namespace=METRIC_NAMESPACE,
)

ONE_CLASS_SVM_THRESHOLD = Gauge(
    "one_class_svm_threshold",
    "Kalibrierter One-Class-SVM-Schwellenwert.",
    namespace=METRIC_NAMESPACE,
)


# ============================================================================
# Legacy-Punktanomalien
# ============================================================================

CURRENT_ANOMALY = Gauge(
    "current_point_anomaly",
    "Legacy-Point-Anomaly-Zustand: 1 = erkannt, 0 = keine erkannt.",
    namespace=METRIC_NAMESPACE,
)

POINT_ANOMALY_DETECTED_TOTAL = Counter(
    "point_anomaly_detected_total",
    "Kumulierte Anzahl erkannter Legacy-Point-Anomalien.",
    namespace=METRIC_NAMESPACE,
)

POINT_ANOMALY_DETECTED_BY_MODEL_TOTAL = Counter(
    "point_anomaly_detected_by_model_total",
    "Kumulierte Point-Anomaly-Detektionen nach Modell.",
    namespace=METRIC_NAMESPACE,
    labelnames=("model",),
)

POINT_ANOMALY_BY_TYPE_TOTAL = Counter(
    "point_anomaly_by_type_total",
    "Kumulierte Point-Anomalien nach Typ und Trigger-Feature.",
    namespace=METRIC_NAMESPACE,
    labelnames=("anomaly_type", "trigger_feature"),
)

ANOMALY_TRIGGER_FEATURE = Gauge(
    "anomaly_trigger_feature",
    (
        "Legacy-Code des aktuellen Trigger-Features. "
        "Die textuelle Information erfolgt über "
        "anomaly_trigger_feature_info."
    ),
    namespace=METRIC_NAMESPACE,
)

ANOMALY_TRIGGER_FEATURE_INFO = Gauge(
    "anomaly_trigger_feature_info",
    "Aktuelles Trigger-Feature einer Legacy-Point-Anomaly.",
    namespace=METRIC_NAMESPACE,
    labelnames=("feature", "reason"),
)


# ============================================================================
# Systemklassifikation
# ============================================================================

SYSTEM_CLASSIFICATION = Gauge(
    "system_classification",
    (
        "Aktuelle Systemklassifikation: "
        "0 = Normal, 1 = Anomaly."
    ),
    namespace=METRIC_NAMESPACE,
)

SYSTEM_CLASSIFICATION_INFO = Gauge(
    "system_classification_info",
    "Textuelle Darstellung der aktuellen Systemklassifikation.",
    namespace=METRIC_NAMESPACE,
    labelnames=("classification",),
)


# ============================================================================
# Root Cause / Remediation
# ============================================================================

ROOT_CAUSE_CLASSIFICATION = Gauge(
    "root_cause_classification",
    (
        "Root-Cause-Code: "
        "0 = normal, "
        "1 = Concept Drift, "
        "2 = Infrastructure Failure."
    ),
    namespace=METRIC_NAMESPACE,
)

ROOT_CAUSE_ACTIVE = Gauge(
    "root_cause_active",
    "Aktive Root Cause mit Trigger-Feature.",
    namespace=METRIC_NAMESPACE,
    labelnames=("root_cause", "trigger_feature"),
)

DIAGNOSTIC_ACTIVE = Gauge(
    "diagnostic_active",
    (
        "Konsolidierter aktueller Diagnosezustand mit "
        "Anomalietyp, Root Cause, Trigger und Remediation."
    ),
    namespace=METRIC_NAMESPACE,
    labelnames=(
        "anomaly_type",
        "root_cause",
        "trigger_feature",
        "remediation",
    ),
)

# Initialer Zustand direkt nach dem Start.
DIAGNOSTIC_ACTIVE.labels(
    anomaly_type="NONE",
    root_cause="NORMAL",
    trigger_feature="none",
    remediation="KEINE AKTION",
).set(1.0)

REMEDIATION_SIGNAL = Gauge(
    "remediation_signal",
    (
        "Remediation-Signal: "
        "0 = keine Aktion, "
        "1 = Retraining, "
        "2 = Infrastruktur-Remediation."
    ),
    namespace=METRIC_NAMESPACE,
)


# ============================================================================
# Leistungsbasiertes Concept-Drift-Monitoring
# ============================================================================
#
# Concept Drift wird nicht aus PSI/KS und nicht aus Infrastrukturmetriken
# abgeleitet.
#
# Entscheidend ist die Veränderung der X-Y-Beziehung, die sich über eine
# anhaltende Verschlechterung der beobachteten Vorhersageleistung zeigt.
#
# Data Drift bleibt davon getrennt.
# ============================================================================

PREDICTION_ERROR = Gauge(
    "prediction_error",
    "Aktueller binärer Vorhersagefehler: 1 = falsch, 0 = korrekt.",
    namespace=METRIC_NAMESPACE,
)

ROLLING_PREDICTION_ERROR = Gauge(
    "rolling_prediction_error",
    "Gleitender Vorhersagefehler über das Performance-Fenster.",
    namespace=METRIC_NAMESPACE,
)

PERFORMANCE_BASELINE_ERROR = Gauge(
    "performance_baseline_error",
    "Referenzfehler des Zielprädiktors unter Normalbedingungen.",
    namespace=METRIC_NAMESPACE,
)

PERFORMANCE_DEGRADATION_THRESHOLD = Gauge(
    "performance_degradation_threshold",
    "Schwellenwert für die leistungsbasierte Concept-Drift-Erkennung.",
    namespace=METRIC_NAMESPACE,
)

PERFORMANCE_CONFIRMATION_COUNT = Gauge(
    "performance_confirmation_count",
    "Anzahl aufeinanderfolgender bestätigender Performance-Fenster.",
    namespace=METRIC_NAMESPACE,
)

PERFORMANCE_DEGRADED = Gauge(
    "performance_degraded",
    "Leistungsabfall des Zielprädiktors: 1 = erkannt, 0 = nicht erkannt.",
    namespace=METRIC_NAMESPACE,
)

CONCEPT_DRIFT_DETECTED = Gauge(
    "concept_drift_detected",
    "Finales Concept-Drift-Signal: 1 = erkannt, 0 = nicht erkannt.",
    namespace=METRIC_NAMESPACE,
)


def update_concept_drift_metrics(
    prediction_error: Optional[float] = None,
    rolling_prediction_error: Optional[float] = None,
    performance_baseline_error: Optional[float] = None,
    performance_degradation_threshold: Optional[float] = None,
    performance_confirmation_count: Optional[float] = None,
    performance_degraded: Optional[bool] = None,
    concept_drift_detected: Optional[bool] = None,
) -> None:
    """
    Aktualisiert die Metriken der leistungsbasierten
    Concept-Drift-Erkennung.
    """

    _set_if_available(
        PREDICTION_ERROR,
        prediction_error,
    )

    _set_if_available(
        ROLLING_PREDICTION_ERROR,
        rolling_prediction_error,
    )

    _set_if_available(
        PERFORMANCE_BASELINE_ERROR,
        performance_baseline_error,
    )

    _set_if_available(
        PERFORMANCE_DEGRADATION_THRESHOLD,
        performance_degradation_threshold,
    )

    _set_if_available(
        PERFORMANCE_CONFIRMATION_COUNT,
        performance_confirmation_count,
    )

    if performance_degraded is not None:
        PERFORMANCE_DEGRADED.set(
            1.0 if performance_degraded else 0.0
        )

    if concept_drift_detected is not None:
        CONCEPT_DRIFT_DETECTED.set(
            1.0 if concept_drift_detected else 0.0
        )


# ============================================================================
# Infrastruktur und Datenqualität
# ============================================================================

CPU_USAGE = Gauge(
    "cpu_usage_percent",
    (
        "Kontrollierter experimenteller "
        "Infrastrukturindikator für CPU-Auslastung in Prozent."
    ),
    namespace=METRIC_NAMESPACE,
)

MEMORY_USAGE = Gauge(
    "memory_usage_percent",
    (
        "Kontrollierter experimenteller "
        "Infrastrukturindikator für Speicherauslastung in Prozent."
    ),
    namespace=METRIC_NAMESPACE,
)

POD_RESTARTS = Gauge(
    "pod_restarts_total",
    (
        "Kontrollierter experimenteller "
        "Infrastrukturindikator für Pod-Neustarts."
    ),
    namespace=METRIC_NAMESPACE,
)

MISSING_VALUE_RATE = Gauge(
    "missing_value_rate",
    "Anteil fehlender Werte pro Feature.",
    namespace=METRIC_NAMESPACE,
    labelnames=("feature",),
)

OUT_OF_BOUNDS_RATE = Gauge(
    "out_of_bounds_rate",
    "Anteil ungültiger Werte pro Feature.",
    namespace=METRIC_NAMESPACE,
    labelnames=("feature",),
)

SCHEMA_VALIDATION_FAILURES = Counter(
    "schema_validation_failures_total",
    "Anzahl fehlgeschlagener Schema-Validierungen.",
    namespace=METRIC_NAMESPACE,
)

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Anzahl eingegangener HTTP-Anfragen.",
    namespace=METRIC_NAMESPACE,
    labelnames=("method", "endpoint", "status"),
)

HTTP_ERRORS = Counter(
    "http_errors_total",
    "Anzahl fehlerhafter HTTP-Anfragen.",
    namespace=METRIC_NAMESPACE,
    labelnames=("method", "endpoint", "status"),
)


# ============================================================================
# Latenz
# ============================================================================

INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Gesamtlatenz einer Inferenz.",
    namespace=METRIC_NAMESPACE,
)

ML_INFERENCE_LATENCY = Histogram(
    "ml_inference_latency_seconds",
    "Latenz einzelner ML-Detektoren.",
    namespace=METRIC_NAMESPACE,
    labelnames=("model",),
)

Z_SCORE_LATENCY = Gauge(
    "z_score_latency_ms",
    "Latenz der robusten Z-Score-Berechnung.",
    namespace=METRIC_NAMESPACE,
)

ISOLATION_FOREST_LATENCY = Gauge(
    "isolation_forest_latency_ms",
    "Latenz der Isolation-Forest-Inferenz.",
    namespace=METRIC_NAMESPACE,
)

ONE_CLASS_SVM_LATENCY = Gauge(
    "one_class_svm_latency_ms",
    "Latenz der One-Class-SVM-Inferenz.",
    namespace=METRIC_NAMESPACE,
)


# ============================================================================
# Streaming
# ============================================================================

STREAM_RUNNING = Gauge(
    "stream_running",
    "Status des Datenstroms: 1 = aktiv, 0 = gestoppt.",
    namespace=METRIC_NAMESPACE,
)

STREAM_STEP = Gauge(
    "stream_step",
    "Aktueller Schritt des Datenstroms.",
    namespace=METRIC_NAMESPACE,
)

STREAM_CYCLE = Gauge(
    "stream_cycle",
    "Aktueller Zyklus des Datenstroms.",
    namespace=METRIC_NAMESPACE,
)

CYCLE_STEP = Gauge(
    "cycle_step",
    "Aktueller Schritt innerhalb des Zyklus.",
    namespace=METRIC_NAMESPACE,
)

WINDOW_SIZE = Gauge(
    "window_size",
    "Aktuelle Größe des gleitenden Fensters.",
    namespace=METRIC_NAMESPACE,
)

SCENARIO_CODE = Gauge(
    "scenario_code",
    "Experiment-Szenario-Code des laufenden Experiments.",
    namespace=METRIC_NAMESPACE,
)


# ============================================================================
# Elec2-Rohdaten
# ============================================================================

NSWPRICE = Gauge(
    "nswprice",
    "Aktueller NSW-Preis.",
    namespace=METRIC_NAMESPACE,
)

NSWDEMAND = Gauge(
    "nswdemand",
    "Aktuelle NSW-Nachfrage.",
    namespace=METRIC_NAMESPACE,
)

VICPRICE = Gauge(
    "vicprice",
    "Aktueller Victoria-Preis.",
    namespace=METRIC_NAMESPACE,
)

VICDEMAND = Gauge(
    "vicdemand",
    "Aktuelle Victoria-Nachfrage.",
    namespace=METRIC_NAMESPACE,
)

TRANSFER = Gauge(
    "transfer",
    "Aktueller Energiefluss.",
    namespace=METRIC_NAMESPACE,
)


# ============================================================================
# Hilfsfunktionen
# ============================================================================

def _set_if_available(
    metric: Gauge,
    value: Optional[float],
) -> None:
    """Setzt eine Gauge nur bei einem vorhandenen Wert."""

    if value is not None:
        metric.set(
            float(value)
        )


# ============================================================================
# Statistische Metriken aktualisieren
# ============================================================================

def update_statistical_metrics(
    rate_of_change: Optional[float] = None,
    peak_density: Optional[float] = None,
    ks_pvalue: Optional[float] = None,
    rolling_mean: Optional[float] = None,
    rolling_std: Optional[float] = None,
    z_score: Optional[float] = None,
    ks_statistic: Optional[float] = None,
) -> None:
    """Aktualisiert die statistischen Metriken."""

    _set_if_available(
        RATE_OF_CHANGE,
        rate_of_change,
    )

    _set_if_available(
        PEAK_DENSITY,
        peak_density,
    )

    _set_if_available(
        KS_PVALUE,
        ks_pvalue,
    )

    _set_if_available(
        KS_STATISTIC,
        ks_statistic,
    )

    _set_if_available(
        ROLLING_MEAN,
        rolling_mean,
    )

    _set_if_available(
        ROLLING_STD,
        rolling_std,
    )

    _set_if_available(
        STATISTICAL_SCORE,
        z_score,
    )


# ============================================================================
# ML-Metriken aktualisieren
# ============================================================================

def update_ml_metrics(
    isolation_forest_score: Optional[float] = None,
    one_class_svm_distance: Optional[float] = None,
    isolation_forest_anomaly: Optional[bool] = None,
    one_class_svm_anomaly: Optional[bool] = None,
    isolation_forest_threshold: Optional[float] = None,
    one_class_svm_threshold: Optional[float] = None,
    model_latencies_ms: Optional[Mapping[str, float]] = None,
) -> None:
    """Aktualisiert die Metriken der ML-Detektoren."""

    _set_if_available(
        ISOLATION_FOREST_SCORE,
        isolation_forest_score,
    )

    _set_if_available(
        ONE_CLASS_SVM_DISTANCE,
        one_class_svm_distance,
    )

    if isolation_forest_anomaly is not None:
        ISOLATION_FOREST_ANOMALY.set(
            1.0 if isolation_forest_anomaly else 0.0
        )

    if one_class_svm_anomaly is not None:
        ONE_CLASS_SVM_ANOMALY.set(
            1.0 if one_class_svm_anomaly else 0.0
        )

    _set_if_available(
        ISOLATION_FOREST_THRESHOLD,
        isolation_forest_threshold,
    )

    _set_if_available(
        ONE_CLASS_SVM_THRESHOLD,
        one_class_svm_threshold,
    )

    if model_latencies_ms:
        if model_latencies_ms.get(
            "isolation_forest"
        ) is not None:

            latency = float(
                model_latencies_ms[
                    "isolation_forest"
                ]
            )

            _set_if_available(
                ISOLATION_FOREST_LATENCY,
                latency,
            )

            ML_INFERENCE_LATENCY.labels(
                model="isolation_forest"
            ).observe(
                latency / 1000.0
            )

        if model_latencies_ms.get(
            "one_class_svm"
        ) is not None:

            latency = float(
                model_latencies_ms[
                    "one_class_svm"
                ]
            )

            _set_if_available(
                ONE_CLASS_SVM_LATENCY,
                latency,
            )

            ML_INFERENCE_LATENCY.labels(
                model="one_class_svm"
            ).observe(
                latency / 1000.0
            )


# ============================================================================
# Data-Drift-Metriken aktualisieren
# ============================================================================

def update_drift_metrics(
    psi_by_feature: Optional[Mapping[str, float]] = None,
    ks_pvalue_by_feature: Optional[Mapping[str, float]] = None,
    ks_statistic_by_feature: Optional[Mapping[str, float]] = None,
    drift_by_feature: Optional[Mapping[str, bool]] = None,
) -> None:
    """Aktualisiert PSI- und KS-Metriken pro Feature."""

    features = set()

    if psi_by_feature:
        features.update(
            psi_by_feature.keys()
        )

    if ks_pvalue_by_feature:
        features.update(
            ks_pvalue_by_feature.keys()
        )

    if ks_statistic_by_feature:
        features.update(
            ks_statistic_by_feature.keys()
        )

    if drift_by_feature:
        features.update(
            drift_by_feature.keys()
        )

    for feature in features:

        psi_value = (
            None
            if psi_by_feature is None
            else psi_by_feature.get(feature)
        )

        ks_pvalue = (
            None
            if ks_pvalue_by_feature is None
            else ks_pvalue_by_feature.get(feature)
        )

        ks_statistic = (
            None
            if ks_statistic_by_feature is None
            else ks_statistic_by_feature.get(feature)
        )

        drift_value = (
            None
            if drift_by_feature is None
            else drift_by_feature.get(feature)
        )

        DRIFT_PSI.labels(
            feature=feature
        ).set(
            float(psi_value)
            if psi_value is not None
            else float("nan")
        )

        DRIFT_KS_PVALUE.labels(
            feature=feature
        ).set(
            float(ks_pvalue)
            if ks_pvalue is not None
            else float("nan")
        )

        DRIFT_KS_STATISTIC.labels(
            feature=feature
        ).set(
            float(ks_statistic)
            if ks_statistic is not None
            else float("nan")
        )

        DRIFT_FEATURE_ANOMALY.labels(
            feature=feature
        ).set(
            float(drift_value)
            if drift_value is not None
            else 0.0
        )


# ============================================================================
# Infrastrukturmetriken aktualisieren
# ============================================================================

def update_infrastructure_metrics(
    cpu_usage_percent: Optional[float] = None,
    memory_usage_percent: Optional[float] = None,
    pod_restarts: Optional[float] = None,
    missing_value_rates: Optional[Mapping[str, float]] = None,
    out_of_bounds_rates: Optional[Mapping[str, float]] = None,
) -> None:
    """Aktualisiert kontrollierte Infrastruktur- und Datenqualitätsmetriken."""

    _set_if_available(
        CPU_USAGE,
        cpu_usage_percent,
    )

    _set_if_available(
        MEMORY_USAGE,
        memory_usage_percent,
    )

    _set_if_available(
        POD_RESTARTS,
        pod_restarts,
    )

    if missing_value_rates:
        for feature, value in missing_value_rates.items():
            if value is not None:
                MISSING_VALUE_RATE.labels(
                    feature=feature
                ).set(
                    float(value)
                )

    if out_of_bounds_rates:
        for feature, value in out_of_bounds_rates.items():
            if value is not None:
                OUT_OF_BOUNDS_RATE.labels(
                    feature=feature
                ).set(
                    float(value)
                )


# ============================================================================
# Systemklassifikation aktualisieren
# ============================================================================

def update_classification(
    classification: Optional[int] = None,
    root_cause: Optional[str] = None,
    trigger_feature: Optional[str] = None,
    remediation_signal: Optional[int] = None,
    anomaly_type: Optional[str] = None,
) -> None:
    """
    Aktualisiert den zweistufigen Systemzustand und die Diagnoseinformation.

    Systemzustand:

        0 = NORMAL
        1 = ANOMALY

    Ein reiner Data Drift verändert den Systemzustand nicht.

    Concept Drift und Infrastructure Failure werden als Anomalien
    klassifiziert.
    """

    # ------------------------------------------------------------------------
    # Systemzustand
    # ------------------------------------------------------------------------

    if classification is not None:

        SYSTEM_CLASSIFICATION.set(
            float(classification)
        )

        names = {
            0: "NORMAL",
            1: "ANOMALY",
        }

        SYSTEM_CLASSIFICATION_INFO.clear()

        SYSTEM_CLASSIFICATION_INFO.labels(
            classification=names.get(
                int(classification),
                "UNKNOWN",
            )
        ).set(1.0)

    # ------------------------------------------------------------------------
    # Diagnoseinformationen
    # ------------------------------------------------------------------------

    root_cause_value = (
        root_cause
        or "NORMAL"
    )

    feature = (
        trigger_feature
        or "none"
    )

    anomaly_type_value = (
        anomaly_type
        or "NONE"
    )

    # ------------------------------------------------------------------------
    # Root-Cause-Codes
    #
    # 0 = NORMAL
    # 1 = CONCEPT_DRIFT
    # 2 = INFRASTRUCTURE
    # ------------------------------------------------------------------------

    root_cause_codes = {
        "NORMAL": 0,
        "DRIFT": 1,
        "CONCEPT_DRIFT": 1,
        "INFRASTRUCTURE": 2,
    }

    ROOT_CAUSE_CLASSIFICATION.set(
        float(
            root_cause_codes.get(
                root_cause_value,
                0,
            )
        )
    )

    # ------------------------------------------------------------------------
    # Aktive Root Cause
    # ------------------------------------------------------------------------

    ROOT_CAUSE_ACTIVE.clear()

    ROOT_CAUSE_ACTIVE.labels(
        root_cause=root_cause_value,
        trigger_feature=feature,
    ).set(1.0)

    # ------------------------------------------------------------------------
    # Remediation
    # ------------------------------------------------------------------------

    remediation_names = {
        0: "KEINE AKTION",
        1: "RETRAINING",
        2: "INFRA-REMEDIATION",
    }

    remediation_value = remediation_names.get(
        int(
            remediation_signal
            or 0
        ),
        "KEINE AKTION",
    )

    # ------------------------------------------------------------------------
    # Konsolidierte Diagnose
    # ------------------------------------------------------------------------

    DIAGNOSTIC_ACTIVE.clear()

    DIAGNOSTIC_ACTIVE.labels(
        anomaly_type=anomaly_type_value,
        root_cause=root_cause_value,
        trigger_feature=feature,
        remediation=remediation_value,
    ).set(1.0)

    if remediation_signal is not None:
        REMEDIATION_SIGNAL.set(
            float(remediation_signal)
        )


# ============================================================================
# Legacy-Point-Anomaly-Metriken
# ============================================================================

def update_point_anomaly(
    point_anomaly: bool,
    anomaly_type: str = "",
    trigger_feature: str = "",
    isolation_forest_anomaly: Optional[bool] = None,
    one_class_svm_anomaly: Optional[bool] = None,
) -> None:
    """
    Aktualisiert den Legacy-Point-Anomaly-Zustand.

    Diese Metriken dienen ausschließlich der Kompatibilität und Diagnose.
    Sie bestimmen nicht die finale Systemklassifikation.
    """

    detected = bool(
        point_anomaly
    )

    CURRENT_ANOMALY.set(
        1.0 if detected else 0.0
    )

    if isolation_forest_anomaly:
        POINT_ANOMALY_DETECTED_BY_MODEL_TOTAL.labels(
            model="isolation_forest"
        ).inc()

    if one_class_svm_anomaly:
        POINT_ANOMALY_DETECTED_BY_MODEL_TOTAL.labels(
            model="one_class_svm"
        ).inc()

    if not detected:
        ANOMALY_TRIGGER_FEATURE_INFO.clear()
        ANOMALY_TRIGGER_FEATURE.set(0.0)
        return

    POINT_ANOMALY_DETECTED_TOTAL.inc()

    anomaly_type = (
        anomaly_type
        or "POINT_ANOMALY"
    )

    trigger_feature = (
        trigger_feature
        or "unknown"
    )

    POINT_ANOMALY_BY_TYPE_TOTAL.labels(
        anomaly_type=anomaly_type,
        trigger_feature=trigger_feature,
    ).inc()

    ANOMALY_TRIGGER_FEATURE_INFO.clear()

    ANOMALY_TRIGGER_FEATURE_INFO.labels(
        feature=trigger_feature,
        reason=anomaly_type,
    ).set(1.0)

    ANOMALY_TRIGGER_FEATURE.set(1.0)


# ============================================================================
# Rohdaten
# ============================================================================

def update_raw_data(
    nswprice: Optional[float] = None,
    nswdemand: Optional[float] = None,
    vicprice: Optional[float] = None,
    vicdemand: Optional[float] = None,
    transfer: Optional[float] = None,
) -> None:
    """Aktualisiert die sichtbaren Elec2-Rohdaten."""

    _set_if_available(
        NSWPRICE,
        nswprice,
    )

    _set_if_available(
        NSWDEMAND,
        nswdemand,
    )

    _set_if_available(
        VICPRICE,
        vicprice,
    )

    _set_if_available(
        VICDEMAND,
        vicdemand,
    )

    _set_if_available(
        TRANSFER,
        transfer,
    )


# ============================================================================
# Streamstatus
# ============================================================================

def update_stream_status(
    running: Optional[bool],
) -> None:
    """Aktualisiert den Status des Datenstroms."""

    if running is not None:
        STREAM_RUNNING.set(
            1.0 if running else 0.0
        )


def update_scenario(
    scenario_code: Optional[int],
) -> None:
    """Aktualisiert das aktuelle Experiment-Szenario."""

    if scenario_code is not None:
        SCENARIO_CODE.set(
            float(scenario_code)
        )


def update_stream_metrics(
    step: Optional[int] = None,
    current_window_size: Optional[int] = None,
    cycle: Optional[int] = None,
    cycle_step: Optional[int] = None,
) -> None:
    """Aktualisiert die Streaming-Metriken."""

    if step is not None:
        STREAM_STEP.set(
            float(step)
        )

    if current_window_size is not None:
        WINDOW_SIZE.set(
            float(current_window_size)
        )

    if cycle is not None:
        STREAM_CYCLE.set(
            float(cycle)
        )

    if cycle_step is not None:
        CYCLE_STEP.set(
            float(cycle_step)
        )


# ============================================================================
# HTTP
# ============================================================================

def record_http_request(
    method: str,
    endpoint: str,
    status: int,
) -> None:
    """Erfasst HTTP-Anfragen und Fehlerantworten."""

    status_text = str(status)

    HTTP_REQUESTS.labels(
        method=method,
        endpoint=endpoint,
        status=status_text,
    ).inc()

    if status >= 400:
        HTTP_ERRORS.labels(
            method=method,
            endpoint=endpoint,
            status=status_text,
        ).inc()


def record_schema_validation_failure() -> None:
    """Erhöht den Zähler fehlgeschlagener Schema-Validierungen."""

    SCHEMA_VALIDATION_FAILURES.inc()


def observe_inference_latency(
    seconds: float,
) -> None:
    """Erfasst die Gesamtlatenz einer Inferenz."""

    if seconds >= 0:
        INFERENCE_LATENCY.observe(
            float(seconds)
        )


# ============================================================================
# Zentrale Metrikaktualisierung
# ============================================================================

def update_all_metrics(
    *,
    rate_of_change: Optional[float] = None,
    peak_density: Optional[float] = None,
    ks_pvalue: Optional[float] = None,
    ks_statistic: Optional[float] = None,
    rolling_mean: Optional[float] = None,
    rolling_std: Optional[float] = None,
    z_score: Optional[float] = None,

    isolation_forest_score: Optional[float] = None,
    one_class_svm_distance: Optional[float] = None,
    isolation_forest_anomaly: Optional[bool] = None,
    one_class_svm_anomaly: Optional[bool] = None,
    isolation_forest_threshold: Optional[float] = None,
    one_class_svm_threshold: Optional[float] = None,

    isolation_forest_latency_ms: Optional[float] = None,
    one_class_svm_latency_ms: Optional[float] = None,
    z_score_latency_ms: Optional[float] = None,

    psi: Optional[float] = None,
    psi_by_feature: Optional[Mapping[str, float]] = None,
    ks_pvalue_by_feature: Optional[Mapping[str, float]] = None,
    ks_statistic_by_feature: Optional[Mapping[str, float]] = None,
    drift_by_feature: Optional[Mapping[str, bool]] = None,

    # ------------------------------------------------------------------------
    # Leistungsbasierte Concept-Drift-Metriken
    #
    # Diese Parameter waren in der vorherigen Version fehlend und führten
    # dadurch zu einem NameError innerhalb von update_all_metrics().
    # ------------------------------------------------------------------------

    prediction_error: Optional[float] = None,
    rolling_prediction_error: Optional[float] = None,
    performance_baseline_error: Optional[float] = None,
    performance_degradation_threshold: Optional[float] = None,
    performance_confirmation_count: Optional[float] = None,
    performance_degraded: Optional[bool] = None,
    concept_drift_detected: Optional[bool] = None,

    # ------------------------------------------------------------------------
    # Infrastruktur
    # ------------------------------------------------------------------------

    cpu_usage_percent: Optional[float] = None,
    memory_usage_percent: Optional[float] = None,
    pod_restarts: Optional[float] = None,
    missing_value_rates: Optional[Mapping[str, float]] = None,
    out_of_bounds_rates: Optional[Mapping[str, float]] = None,

    # ------------------------------------------------------------------------
    # Klassifikation
    # ------------------------------------------------------------------------

    classification: Optional[int] = None,
    root_cause: Optional[str] = None,
    trigger_feature: Optional[str] = None,
    remediation_signal: Optional[int] = None,
    anomaly_type: Optional[str] = None,

    # ------------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------------

    scenario_code: Optional[int] = None,
    step: Optional[int] = None,
    current_window_size: Optional[int] = None,
    cycle: Optional[int] = None,
    cycle_step: Optional[int] = None,
    stream_running: Optional[bool] = None,

    # ------------------------------------------------------------------------
    # Sonstige Metriken
    # ------------------------------------------------------------------------

    z_score_threshold: Optional[float] = None,

    nswprice: Optional[float] = None,
    nswdemand: Optional[float] = None,
    vicprice: Optional[float] = None,
    vicdemand: Optional[float] = None,
    transfer: Optional[float] = None,

    inference_latency_seconds: Optional[float] = None,

    # ------------------------------------------------------------------------
    # Legacy Point Anomaly
    # ------------------------------------------------------------------------

    point_anomaly: Optional[bool] = None,
    anomaly_trigger_feature: Optional[str] = None,
) -> None:
    """
    Zentrale Schnittstelle für alle Laufzeitmetriken.

    Die finale Systemklassifikation basiert auf NORMAL/ANOMALY.

    Data Drift wird separat über PSI/KS überwacht.

    Concept Drift wird separat über die beobachtete
    Vorhersageleistungsverschlechterung überwacht.

    Isolation Forest und One-Class SVM bleiben diagnostische bzw.
    vergleichende Verfahren.
    """

    # ------------------------------------------------------------------------
    # Statistische Metriken
    # ------------------------------------------------------------------------

    update_statistical_metrics(
        rate_of_change=rate_of_change,
        peak_density=peak_density,
        ks_pvalue=ks_pvalue,
        ks_statistic=ks_statistic,
        rolling_mean=rolling_mean,
        rolling_std=rolling_std,
        z_score=z_score,
    )

    # ------------------------------------------------------------------------
    # ML-Detektoren
    # ------------------------------------------------------------------------

    update_ml_metrics(
        isolation_forest_score=isolation_forest_score,
        one_class_svm_distance=one_class_svm_distance,
        isolation_forest_anomaly=isolation_forest_anomaly,
        one_class_svm_anomaly=one_class_svm_anomaly,
        isolation_forest_threshold=isolation_forest_threshold,
        one_class_svm_threshold=one_class_svm_threshold,
        model_latencies_ms={
            "isolation_forest": isolation_forest_latency_ms,
            "one_class_svm": one_class_svm_latency_ms,
        },
    )

    # ------------------------------------------------------------------------
    # Data Drift
    # ------------------------------------------------------------------------

    update_drift_metrics(
        psi_by_feature=psi_by_feature,
        ks_pvalue_by_feature=ks_pvalue_by_feature,
        ks_statistic_by_feature=ks_statistic_by_feature,
        drift_by_feature=drift_by_feature,
    )

    # ------------------------------------------------------------------------
    # Concept Drift
    #
    # Wichtig:
    # Die sieben Parameter werden direkt aus update_all_metrics() an die
    # dedizierte Concept-Drift-Funktion weitergegeben.
    # ------------------------------------------------------------------------

    update_concept_drift_metrics(
        prediction_error=prediction_error,
        rolling_prediction_error=rolling_prediction_error,
        performance_baseline_error=performance_baseline_error,
        performance_degradation_threshold=(
            performance_degradation_threshold
        ),
        performance_confirmation_count=(
            performance_confirmation_count
        ),
        performance_degraded=performance_degraded,
        concept_drift_detected=concept_drift_detected,
    )

    # ------------------------------------------------------------------------
    # Infrastruktur
    # ------------------------------------------------------------------------

    update_infrastructure_metrics(
        cpu_usage_percent=cpu_usage_percent,
        memory_usage_percent=memory_usage_percent,
        pod_restarts=pod_restarts,
        missing_value_rates=missing_value_rates,
        out_of_bounds_rates=out_of_bounds_rates,
    )

    # ------------------------------------------------------------------------
    # Finale Klassifikation und Root Cause
    # ------------------------------------------------------------------------

    update_classification(
        classification=classification,
        root_cause=root_cause,
        trigger_feature=trigger_feature,
        remediation_signal=remediation_signal,
        anomaly_type=anomaly_type,
    )

    # ------------------------------------------------------------------------
    # Legacy Point-Anomaly-Metriken
    # ------------------------------------------------------------------------

    if point_anomaly is not None:
        update_point_anomaly(
            point_anomaly=point_anomaly,
            anomaly_type=anomaly_type or "",
            trigger_feature=(
                anomaly_trigger_feature
                or ""
            ),
            isolation_forest_anomaly=(
                isolation_forest_anomaly
            ),
            one_class_svm_anomaly=(
                one_class_svm_anomaly
            ),
        )

    # ------------------------------------------------------------------------
    # Szenario und Streaming
    # ------------------------------------------------------------------------

    update_scenario(
        scenario_code
    )

    update_stream_status(
        stream_running
    )

    update_stream_metrics(
        step=step,
        current_window_size=current_window_size,
        cycle=cycle,
        cycle_step=cycle_step,
    )

    # ------------------------------------------------------------------------
    # Schwellenwert
    # ------------------------------------------------------------------------

    _set_if_available(
        Z_SCORE_THRESHOLD_METRIC,
        z_score_threshold,
    )

    # ------------------------------------------------------------------------
    # Rohdaten
    # ------------------------------------------------------------------------

    update_raw_data(
        nswprice=nswprice,
        nswdemand=nswdemand,
        vicprice=vicprice,
        vicdemand=vicdemand,
        transfer=transfer,
    )

    # ------------------------------------------------------------------------
    # Latenzen
    # ------------------------------------------------------------------------

    if z_score_latency_ms is not None:
        _set_if_available(
            Z_SCORE_LATENCY,
            z_score_latency_ms,
        )

    if inference_latency_seconds is not None:
        observe_inference_latency(
            inference_latency_seconds
        )