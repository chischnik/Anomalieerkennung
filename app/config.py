from pathlib import Path


# ============================================================================
# Pfade und allgemeine Laufzeitparameter
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "elec2.csv"

# Intervall des Live-Demostreams.
# 0,01 s pro Beobachtung ermöglicht einen schnellen, aber weiterhin
# beobachtbaren Demo-Zyklus.
# Der Benchmark verwendet weiterhin seinen eigenen stream_fast()-Pfad.
STREAM_INTERVAL_SECONDS = 0.01

MAX_STREAM_ROWS = None
RANDOM_STATE = 42


# ============================================================================
# Fenster- und Referenzkonfiguration
# ============================================================================

WINDOW_SIZE = 200
MIN_WINDOW_SIZE = 30
REFERENCE_SIZE = 3000

# Anzahl aufeinanderfolgender statistischer Driftentscheidungen,
# die für eine bestätigte operative Drift erforderlich sind.
DRIFT_CONFIRMATION_WINDOWS = 3

# Intervall für die periodische statistische Auswertung im Benchmark.
STATISTICAL_STEP = 10


# ============================================================================
# Statistische Schwellenwerte
# ============================================================================

Z_SCORE_THRESHOLD = 3.0

# Schwellenwert für die Peak-Erkennung:
# Peak > Mittelwert + PEAK_SIGMA_FACTOR * Standardabweichung
PEAK_SIGMA_FACTOR = 2.0

KS_ALPHA = 0.05
PSI_THRESHOLD = 0.25


# ============================================================================
# Isolation Forest
# ============================================================================

ISOLATION_FOREST_ESTIMATORS = 100
ISOLATION_FOREST_CONTAMINATION = 0.05
ISOLATION_FOREST_RANDOM_STATE = RANDOM_STATE


# ============================================================================
# One-Class SVM
# ============================================================================

ONE_CLASS_SVM_KERNEL = "rbf"
ONE_CLASS_SVM_NU = 0.05
ONE_CLASS_SVM_GAMMA = "scale"


# ============================================================================
# Experimentelle Szenario-Injektion
# ============================================================================

ENABLE_SCENARIO_INJECTION = True

# Zufallsraten für den nicht-deterministischen Modus.
# Im Live-Demomodus werden die Trigger über feste Teilabschnitte erzeugt.
INFRA_MISSING_RATE = 0.02
INFRA_EXTREME_RATE = 0.01

INJECTION_RANDOM_STATE = RANDOM_STATE


# ============================================================================
# Data-Drift-Phase
# ============================================================================

# Die bisher als CONCEPT_DRIFT bezeichnete Phase erzeugt ausschließlich
# eine Veränderung der Eingangsverteilungen P(X).
#
# Deshalb wird diese Phase im finalen Versuchsprotokoll als DATA_DRIFT
# bezeichnet.
DATA_DRIFT_START_RATIO = 0.20
DATA_DRIFT_END_RATIO = 0.30

# Maximale Stärke der kontrollierten Feature-Verschiebung.
# Die eigentliche Transformation erfolgt in stream_ingestor.py.
DATA_DRIFT_MAX_SHIFT = 0.10


# ============================================================================
# Concept-Drift-Phase
# ============================================================================

# In dieser Phase wird ausschließlich die Beziehung P(Y|X) verändert.
# Die Eingangsfeatures X bleiben unverändert; das beobachtete Target Y
# wird kontrolliert invertiert.
CONCEPT_DRIFT_START_RATIO = 0.30
CONCEPT_DRIFT_END_RATIO = 0.40


# ============================================================================
# Infrastructure-Failure-Phase
# ============================================================================

INFRASTRUCTURE_START_RATIO = 0.45
INFRASTRUCTURE_END_RATIO = 0.50

INFRA_CPU_START_PERCENT = 80.0
INFRA_CPU_END_PERCENT = 98.0

INFRA_MEMORY_START_PERCENT = 75.0
INFRA_MEMORY_END_PERCENT = 95.0

INFRA_RESTART_PROGRESS = 0.60

# Die Live-Demonstration verwendet feste Teilabschnitte innerhalb der
# Infrastructure-Phase. Dadurch werden alle vorgesehenen Trigger
# reproduzierbar sichtbar und hängen nicht vom Zufall ab.
INFRA_DEMO_DETERMINISTIC = True


# ============================================================================
# Feature-Grenzen
# ============================================================================

FEATURE_BOUNDS = {
    "nswprice": None,
    "nswdemand": None,
    "vicprice": None,
    "vicdemand": None,
    "transfer": None,
}

REFERENCE_BOUND_FACTOR = 3.0


# ============================================================================
# API-Konfiguration
# ============================================================================

API_HOST = "0.0.0.0"
API_PORT = 8000

PREDICT_ENDPOINT = "/predict"
HEALTH_ENDPOINT = "/health"
METRICS_ENDPOINT = "/metrics"


# ============================================================================
# Infrastruktur- und Qualitätsalarme
# ============================================================================

MISSING_VALUE_ALERT_RATE = 0.05
OUT_OF_BOUNDS_ALERT_RATE = 0.05

LATENCY_ALERT_SECONDS = 1.0

CPU_ALERT_PERCENT = 90.0
MEMORY_ALERT_PERCENT = 90.0

POD_RESTART_ALERT_COUNT = 1

HTTP_ERROR_ALERT_RATE = 0.05


# ============================================================================
# Zustands- und Anomalietypen
# ============================================================================

# Der operative Systemzustand besitzt ausschließlich zwei Zustände:
#
#     0 = NORMAL
#     1 = ANOMALY
#
# Ein reiner Data Drift ist KEINE Anomalie.
# Ein Concept Drift und ein Infrastructure Failure sind Anomalien.

SYSTEM_STATE_NORMAL = 0
SYSTEM_STATE_ANOMALY = 1

SYSTEM_STATE_NAMES = {
    SYSTEM_STATE_NORMAL: "Normal",
    SYSTEM_STATE_ANOMALY: "Anomaly",
}


# ============================================================================
# Anomalietypen
# ============================================================================

# Keine Anomalie.
ANOMALY_NONE = "NONE"

# Concept Drift ist eine echte Anomalie, weil sich die Beziehung
# zwischen Eingangsvariablen X und Zielvariable Y verändert.
ANOMALY_CONCEPT_DRIFT = "ANOMALY_CONCEPT_DRIFT"

# Infrastrukturfehler bilden eine eigenständige Anomaliefamilie.
ANOMALY_INFRASTRUCTURE = "INFRASTRUCTURE_FAILURE"

ANOMALY_NAMES = {
    ANOMALY_NONE: "None",
    ANOMALY_CONCEPT_DRIFT: "Concept Drift",
    ANOMALY_INFRASTRUCTURE: "Infrastructure Failure",
}


# ============================================================================
# Experimentelle Klassen
# ============================================================================

# Das experimentelle Szenario wird bewusst vom operativen Systemzustand
# getrennt.
#
# DATA_DRIFT:
#     Veränderung von P(X), aber keine Änderung von P(Y|X).
#
# CONCEPT_DRIFT:
#     Veränderung von P(Y|X).
#
# INFRASTRUCTURE_FAILURE:
#     Technischer Fehler der Laufzeitumgebung.

CLASS_NORMAL = SYSTEM_STATE_NORMAL
CLASS_DATA_DRIFT = 1
CLASS_CONCEPT_DRIFT = 2
CLASS_INFRASTRUCTURE = 3

CLASS_NAMES = {
    CLASS_NORMAL: "Normal",
    CLASS_DATA_DRIFT: "Data Drift",
    CLASS_CONCEPT_DRIFT: "Concept Drift",
    CLASS_INFRASTRUCTURE: "Infrastructure Failure",
}


# ============================================================================
# Root Causes
# ============================================================================

ROOT_CAUSE_NORMAL = "NORMAL"

# Ein reiner Data Drift ist keine Anomalie.
# Dieser Wert wird deshalb nur zur Beschreibung der Driftursache verwendet.
ROOT_CAUSE_DRIFT = "DRIFT"

ROOT_CAUSE_CONCEPT_DRIFT = "CONCEPT_DRIFT"

ROOT_CAUSE_INFRASTRUCTURE = "INFRASTRUCTURE"


# ============================================================================
# Trigger
# ============================================================================

# Kein Anomalie-Trigger.
TRIGGER_NONE = "NONE"

# Legacy-Trigger für ältere Auswertungen.
# Diese Werte werden im finalen E2E-Protokoll nicht mehr als eigenständige
# Anomaliekategorien verwendet.
TRIGGER_PRICE_PEAK = "PRICE_PEAK"
TRIGGER_DEMAND_PEAK = "DEMAND_PEAK"
TRIGGER_FEATURE_PEAK = "FEATURE_PEAK"

# Infrastruktur-Trigger.
TRIGGER_EXTREME_VALUE = "EXTREME_VALUE"
TRIGGER_HIGH_CPU = "HIGH_CPU"
TRIGGER_HIGH_MEMORY = "HIGH_MEMORY"
TRIGGER_POD_RESTART = "POD_RESTART"
TRIGGER_MISSING_VALUE = "MISSING_VALUE"


# ============================================================================
# Szenarien
# ============================================================================

SCENARIO_NORMAL = "normal"

# Reiner Data Drift:
# P(X) verändert sich, während Y unverändert bleibt.
SCENARIO_DATA_DRIFT = "data_drift"

# Echter Concept Drift:
# Die Beziehung P(Y|X) wird im kontrollierten Experiment verändert.
SCENARIO_CONCEPT_DRIFT = "concept_drift"

# Technischer Fehler der Infrastruktur.
SCENARIO_INFRASTRUCTURE_FAILURE = "infrastructure_failure"


SCENARIO_NAMES = {
    SCENARIO_NORMAL: "Normal",
    SCENARIO_DATA_DRIFT: "Data Drift",
    SCENARIO_CONCEPT_DRIFT: "Concept Drift",
    SCENARIO_INFRASTRUCTURE_FAILURE: "Infrastructure Failure",
}


# ============================================================================
# Infrastrukturinterne Fehlerklassen
# ============================================================================

INFRA_MISSING_DATA = "INFRA_MISSING_DATA"
INFRA_VALUE_EXTREME = "INFRA_VALUE_EXTREME"
INFRA_SCHEMA_ERROR = "INFRA_SCHEMA_ERROR"
INFRA_RESOURCE = "INFRA_RESOURCE"

INFRA_ROOT_CAUSES = {
    INFRA_MISSING_DATA,
    INFRA_VALUE_EXTREME,
    INFRA_SCHEMA_ERROR,
    INFRA_RESOURCE,
}


# ============================================================================
# Bekannte Feature-Spalten
# ============================================================================

PRICE_COLUMN = "nswprice"
DEMAND_COLUMN = "nswdemand"
TRANSFER_COLUMN = "transfer"

TIME_COLUMNS = [
    "date",
    "day",
    "period",
]

FEATURE_COLUMNS = [
    "nswprice",
    "nswdemand",
    "vicprice",
    "vicdemand",
    "transfer",
]


# ============================================================================
# Zielvariable
# ============================================================================

# Die Elec2-Zielvariable enthält die beobachteten Klassen UP und DOWN.
# Sie wird nicht als Eingangsfeature für Isolation Forest oder One-Class SVM
# verwendet.
TARGET_COLUMN = "class"


# ============================================================================
# Kalibrierung
# ============================================================================

CALIBRATION_SIZE = 1000
CALIBRATION_QUANTILE = 0.01