"""
End-to-End-Experiment V16 für den FastAPI-Anomaliedetektor.

Die vier finalen Szenarien werden gegen den laufenden /predict-Endpunkt
getestet:

1. Normal
2. Reiner Data Drift
3. Concept Drift
4. Infrastructure Failure

Die Ground Truth wird ausschließlich zur späteren Bewertung verwendet.
Sie wird niemals an den FastAPI-Service übertragen und beeinflusst dessen
operative Entscheidung nicht.

Der E2E-Test verwendet dieselben offiziellen Szenariogrenzen und dieselbe
Concept-Drift-Erzeugung wie die finale V16-Benchmark- und RCA-Evaluation.

Wichtige semantische Trennung:

- Data Drift beschreibt eine Veränderung der Eingabeverteilung P(X).
  Data Drift ist für sich genommen keine Systemanomalie.

- Concept Drift beschreibt eine Veränderung der Beziehung X -> Y.
  Er wird über eine persistente Verschlechterung der beobachteten
  Vorhersageleistung erkannt.

- Infrastructure Failure wird unabhängig über kontrollierte
  Infrastrukturindikatoren erkannt.

- Isolation Forest und One-Class SVM dienen als vergleichende bzw.
  diagnostische Verfahren. Sie bestimmen nicht allein die finale
  Systemklassifikation.

Kommentare und Docstrings sind auf Deutsch.
"""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests


# ============================================================================
# Projektpfad
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Projektmodule
# ============================================================================

from app.config import (
    ANOMALY_CONCEPT_DRIFT,
    ANOMALY_INFRASTRUCTURE,
    ANOMALY_NONE,
    CPU_ALERT_PERCENT,
    MEMORY_ALERT_PERCENT,
    POD_RESTART_ALERT_COUNT,
    SYSTEM_STATE_ANOMALY,
    SYSTEM_STATE_NORMAL,
)

from app.stream_ingestor import create_ingestor


# ============================================================================
# FastAPI-Endpunkte
# ============================================================================

BASE_URL = "http://127.0.0.1:8000"

PREDICT_URL = f"{BASE_URL}/predict"
RESET_URL = f"{BASE_URL}/runtime/reset"
HEALTH_URL = f"{BASE_URL}/health"


# ============================================================================
# Ergebnisdatei
# ============================================================================

RESULT_DIR = PROJECT_ROOT / "evaluation" / "results"

RESULT_FILE = RESULT_DIR / "e2e_v16_results.csv"


# ============================================================================
# Elec2-Eingangsmerkmale
# ============================================================================

FEATURES = (
    "nswprice",
    "nswdemand",
    "vicprice",
    "vicdemand",
    "transfer",
)


# ============================================================================
# HTTP-Hilfsfunktionen
# ============================================================================

def request_reset(
    session: requests.Session,
) -> Dict[str, Any]:
    """Setzt den kurzfristigen Laufzeitstatus des Services zurück."""

    response = session.post(
        RESET_URL,
        timeout=15,
    )

    response.raise_for_status()

    return response.json()


def request_health(
    session: requests.Session,
) -> Dict[str, Any]:
    """Prüft, ob der FastAPI-Service erreichbar und betriebsbereit ist."""

    response = session.get(
        HEALTH_URL,
        timeout=15,
    )

    response.raise_for_status()

    return response.json()


# ============================================================================
# Payload-Erzeugung
# ============================================================================

def observation_to_payload(
    observation: Any,
) -> Dict[str, Any]:
    """
    Wandelt eine Elec2-Beobachtung in eine /predict-Payload um.

    Die Ground-Truth-Klassifikation wird bewusst nicht übertragen.

    Das beobachtete Target bleibt Bestandteil der Payload, weil der
    Produktionsdetektor die beobachtete Zielvariable zur Überwachung der
    Vorhersageleistung und damit zur experimentellen Concept-Drift-Erkennung
    benötigt.
    """

    return {
        # --------------------------------------------------------------------
        # Eingangsmerkmale X
        # --------------------------------------------------------------------

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

        # --------------------------------------------------------------------
        # Beobachtetes Target Y
        # --------------------------------------------------------------------

        "target": str(
            getattr(
                observation,
                "target",
                "",
            )
        ),

        # --------------------------------------------------------------------
        # Zeitinformationen
        # --------------------------------------------------------------------

        "step": int(
            observation.step
        ),

        "timestamp": float(
            observation.timestamp
        ),

        # --------------------------------------------------------------------
        # Kontrollierte Infrastrukturindikatoren
        #
        # Diese Werte stammen aus dem kontrollierten Experiment und stellen
        # keine direkt aus Kubernetes ausgelesenen Pod-Metriken dar.
        # --------------------------------------------------------------------

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
    }


# ============================================================================
# Beobachtung an den echten Service senden
# ============================================================================

def post_observation(
    session: requests.Session,
    observation: Any,
) -> Dict[str, Any]:
    """
    Sendet eine Beobachtung an den echten FastAPI-/predict-Endpunkt.

    Die Ground Truth für Systemzustand und Anomalietyp wird nicht an den
    Service übertragen. Sie wird erst nach der Service-Antwort lokal für
    die spätere Evaluation gespeichert.
    """

    # Vor dem JSON-Versand werden alle Eingangsmerkmale auf gültige,
    # endliche numerische Werte geprüft.
    if not all(
        math.isfinite(
            float(
                getattr(
                    observation,
                    feature,
                )
            )
        )
        for feature in FEATURES
    ):
        raise ValueError(
            "Die Beobachtung enthält NaN/Inf und kann nicht als "
            "standardkonformes JSON an /predict übertragen werden."
        )

    payload = observation_to_payload(
        observation
    )

    response = session.post(
        PREDICT_URL,
        json=payload,
        timeout=15,
    )

    response.raise_for_status()

    result = response.json()

    # ------------------------------------------------------------------------
    # Ground Truth
    # ------------------------------------------------------------------------
    #
    # Diese Informationen stammen ausschließlich aus dem experimentellen
    # Ingestor und werden nicht zur operativen Entscheidung verwendet.
    # ------------------------------------------------------------------------

    row: Dict[str, Any] = {
        "timestamp": time.time(),

        "step": int(
            observation.step
        ),

        "scenario": str(
            getattr(
                observation,
                "scenario",
                "",
            )
        ),

        "ground_truth_target": str(
            getattr(
                observation,
                "target",
                "",
            )
        ),

        "ground_truth_system_state": int(
            getattr(
                observation,
                "system_state",
                SYSTEM_STATE_NORMAL,
            )
        ),

        "ground_truth_anomaly_type": str(
            getattr(
                observation,
                "anomaly_type",
                ANOMALY_NONE,
            )
        ),

        "ground_truth_root_cause": str(
            getattr(
                observation,
                "root_cause",
                "",
            )
        ),

        "ground_truth_trigger_feature": str(
            getattr(
                observation,
                "trigger_feature",
                "",
            )
        ),

        # --------------------------------------------------------------------
        # Finale Systemvorhersage
        # --------------------------------------------------------------------

        "pred_system_state": result.get(
            "system_state"
        ),

        "pred_system_state_name": result.get(
            "system_state_name"
        ),

        "pred_anomaly_type": result.get(
            "anomaly_type"
        ),

        # --------------------------------------------------------------------
        # Zielvariable und Concept Drift
        # --------------------------------------------------------------------

        "pred_target_prediction": result.get(
            "target_prediction"
        ),

        "pred_prediction_error": result.get(
            "prediction_error"
        ),

        "pred_rolling_prediction_error": result.get(
            "rolling_prediction_error"
        ),

        "pred_performance_baseline_error": result.get(
            "performance_baseline_error"
        ),

        "pred_performance_degradation_threshold": result.get(
            "performance_degradation_threshold"
        ),

        "pred_performance_confirmation_count": result.get(
            "performance_confirmation_count"
        ),

        "pred_performance_degraded": result.get(
            "performance_degraded"
        ),

        "pred_concept_drift_detected": result.get(
            "concept_drift_detected"
        ),

        # --------------------------------------------------------------------
        # Statistische Data-Drift-Signale
        # --------------------------------------------------------------------

        "pred_drift_detected": result.get(
            "drift_detected"
        ),

        "pred_psi": result.get(
            "psi"
        ),

        # --------------------------------------------------------------------
        # Root Cause
        # --------------------------------------------------------------------

        "pred_root_cause": result.get(
            "root_cause"
        ),

        "pred_trigger_feature": result.get(
            "trigger_feature"
        ),

        # --------------------------------------------------------------------
        # Infrastruktur
        # --------------------------------------------------------------------

        "pred_infrastructure_failure": result.get(
            "infrastructure_failure"
        ),

        "pred_infrastructure_root_cause": result.get(
            "infrastructure_root_cause"
        ),

        "pred_infrastructure_trigger_feature": result.get(
            "infrastructure_trigger_feature"
        ),

        # --------------------------------------------------------------------
        # Vergleichsverfahren
        # --------------------------------------------------------------------

        "if_anomaly": result.get(
            "isolation_forest",
            {},
        ).get(
            "is_anomaly"
        ),

        "if_score": result.get(
            "isolation_forest",
            {},
        ).get(
            "score"
        ),

        "ocsvm_anomaly": result.get(
            "one_class_svm",
            {},
        ).get(
            "is_anomaly"
        ),

        "ocsvm_score": result.get(
            "one_class_svm",
            {},
        ).get(
            "score"
        ),

        # --------------------------------------------------------------------
        # Laufzeit
        # --------------------------------------------------------------------

        "latency_seconds": result.get(
            "latency_seconds"
        ),
    }

    return row


# ============================================================================
# Beobachtungen erzeugen
# ============================================================================

def observations_for_steps(
    ingestor: Any,
    steps: Iterable[int],
) -> List[Any]:
    """Erzeugt deterministisch die gewünschten Elec2-Beobachtungen."""

    observations: List[Any] = []

    for step in steps:
        observation = ingestor._create_observation(
            int(step)
        )

        observations.append(
            observation
        )

    return observations


# ============================================================================
# Beobachtungssequenz ausführen
# ============================================================================

def run_sequence(
    session: requests.Session,
    observations: Iterable[Any],
    label: str,
) -> List[Dict[str, Any]]:
    """Führt eine Beobachtungssequenz gegen den echten Service aus."""

    rows: List[Dict[str, Any]] = []

    for index, observation in enumerate(
        observations,
        start=1,
    ):
        row = post_observation(
            session,
            observation,
        )

        row["test"] = label

        row["sequence_index"] = index

        rows.append(
            row
        )

    return rows


# ============================================================================
# Validierung: erwartete finale Klassifikation
# ============================================================================

def assert_expected_prediction(
    rows: List[Dict[str, Any]],
    *,
    expected_system_state: int,
    expected_anomaly_type: str,
    label: str,
) -> None:
    """
    Prüft die finale Klassifikation des Services.

    Die Prüfung basiert ausschließlich auf der vom Service gelieferten
    Prediction.
    """

    if not rows:
        raise AssertionError(
            f"{label}: Der Test lieferte keine Beobachtungen."
        )

    last = rows[-1]

    predicted_state = int(
        last["pred_system_state"]
    )

    predicted_type = str(
        last["pred_anomaly_type"]
    )

    if predicted_state != expected_system_state:
        raise AssertionError(
            f"{label}: Unerwarteter Systemzustand: "
            f"{predicted_state} statt "
            f"{expected_system_state}."
        )

    if predicted_type != expected_anomaly_type:
        raise AssertionError(
            f"{label}: Unerwarteter Anomalietyp: "
            f"{predicted_type} statt "
            f"{expected_anomaly_type}."
        )


# ============================================================================
# Validierung: reiner Data Drift
# ============================================================================

def assert_pure_data_drift_is_normal(
    rows: List[Dict[str, Any]],
) -> None:
    """
    Prüft die finale Data-Drift-Semantik.

    Data Drift darf als statistische Veränderung der Eingabeverteilung
    erkannt werden, darf aber allein keine Systemanomalie und keinen
    Concept Drift auslösen.

    Falls während des Data Drift dennoch Systemanomalien auftreten,
    werden die betroffenen Beobachtungen vollständig ausgegeben.
    Dadurch kann unterschieden werden, ob beispielsweise eine
    Infrastrukturregel oder eine andere Klassifikationslogik die
    unerwartete Anomalie verursacht.
    """

    if not rows:
        raise AssertionError(
            "DATA_DRIFT: Keine Beobachtungen vorhanden."
        )

    # ------------------------------------------------------------------------
    # Es muss ein statistisches Data-Drift-Signal vorhanden sein.
    # ------------------------------------------------------------------------

    data_drift_detected = any(
        bool(
            row.get(
                "pred_drift_detected",
                False,
            )
        )
        for row in rows
    )

    if not data_drift_detected:
        raise AssertionError(
            "DATA_DRIFT: Trotz des vorgesehenen Data-Drift-Bereichs "
            "wurde kein statistisches Drift-Signal erkannt."
        )

    # ------------------------------------------------------------------------
    # Data Drift darf keinen Systemzustand ANOMALY erzeugen.
    #
    # Die betroffenen Zeilen werden vor dem Abbruch detailliert ausgegeben.
    # Dadurch wird nicht nur die Anzahl, sondern auch die tatsächliche
    # Ursache der unerwarteten Klassifikation sichtbar.
    # ------------------------------------------------------------------------

    false_system_anomalies = [
        row
        for row in rows
        if int(
            row.get(
                "pred_system_state",
                SYSTEM_STATE_NORMAL,
            )
        ) != SYSTEM_STATE_NORMAL
    ]

    if false_system_anomalies:
        print()
        print("=" * 80)
        print(
            "DIAGNOSE – ANOMALIEN WÄHREND DES REINEN DATA DRIFT"
        )
        print("=" * 80)

        print(
            f"Anzahl unerwarteter Systemanomalien: "
            f"{len(false_system_anomalies)}"
        )

        print()

        for row in false_system_anomalies:
            print(
                f"Step={row.get('step')} | "
                f"System={row.get('pred_system_state')} | "
                f"Anomaly={row.get('pred_anomaly_type')} | "
                f"RootCause={row.get('pred_root_cause')} | "
                f"Trigger={row.get('pred_trigger_feature')} | "
                f"Infra={row.get('pred_infrastructure_failure')} | "
                f"InfraRootCause="
                f"{row.get('pred_infrastructure_root_cause')} | "
                f"InfraTrigger="
                f"{row.get('pred_infrastructure_trigger_feature')} | "
                f"DataDrift={row.get('pred_drift_detected')} | "
                f"ConceptDrift="
                f"{row.get('pred_concept_drift_detected')} | "
                f"PredictionError="
                f"{row.get('pred_prediction_error')} | "
                f"RollingError="
                f"{row.get('pred_rolling_prediction_error')} | "
                f"PerformanceDegraded="
                f"{row.get('pred_performance_degraded')} | "
                f"IF={row.get('if_anomaly')} | "
                f"OCSVM={row.get('ocsvm_anomaly')}"
            )

        print()
        print("=" * 80)

        raise AssertionError(
            "DATA_DRIFT: Reiner Data Drift wurde fälschlich als "
            f"Systemanomalie klassifiziert "
            f"({len(false_system_anomalies)} Beobachtungen)."
        )

    # ------------------------------------------------------------------------
    # Data Drift darf keinen Concept Drift auslösen.
    # ------------------------------------------------------------------------

    false_concept_drift = [
        row
        for row in rows
        if bool(
            row.get(
                "pred_concept_drift_detected",
                False,
            )
        )
    ]

    if false_concept_drift:
        raise AssertionError(
            "DATA_DRIFT: Reiner Data Drift wurde fälschlich als "
            f"Concept Drift klassifiziert "
            f"({len(false_concept_drift)} Beobachtungen)."
        )

    # ------------------------------------------------------------------------
    # Der finale Anomalietyp muss NONE bleiben.
    # ------------------------------------------------------------------------

    false_anomaly_types = [
        row
        for row in rows
        if str(
            row.get(
                "pred_anomaly_type",
                ANOMALY_NONE,
            )
        ) != ANOMALY_NONE
    ]

    if false_anomaly_types:
        raise AssertionError(
            "DATA_DRIFT: Der finale Anomalietyp wurde trotz reinem "
            f"Data Drift verändert "
            f"({len(false_anomaly_types)} Beobachtungen)."
        )


# ============================================================================
# Validierung: Concept Drift
# ============================================================================

def assert_concept_drift_detected(
    rows: List[Dict[str, Any]],
) -> None:
    """
    Prüft die performancebasierte Concept-Drift-Erkennung.

    Concept Drift muss sich in einer Verschlechterung der beobachteten
    X -> Y-Vorhersageleistung zeigen und nach der definierten
    Bestätigungslogik als Concept Drift erkannt werden.
    """

    if not rows:
        raise AssertionError(
            "CONCEPT_DRIFT: Keine Beobachtungen vorhanden."
        )

    # ------------------------------------------------------------------------
    # Bestätigte Concept-Drift-Beobachtungen suchen.
    # ------------------------------------------------------------------------

    confirmed = [
        row
        for row in rows
        if bool(
            row.get(
                "pred_concept_drift_detected",
                False,
            )
        )
    ]

    if not confirmed:
        raise AssertionError(
            "CONCEPT_DRIFT: Kein bestätigter Concept Drift erkannt."
        )

    # ------------------------------------------------------------------------
    # Es muss eine messbare Prediction-Error-Verschlechterung vorliegen.
    # ------------------------------------------------------------------------

    performance_degradation = [
        row
        for row in confirmed
        if float(
            row.get(
                "pred_rolling_prediction_error",
                0.0,
            )
            or 0.0
        ) > 0.0
    ]

    if not performance_degradation:
        raise AssertionError(
            "CONCEPT_DRIFT: Es wurde kein erhöhter gleitender "
            "Vorhersagefehler beobachtet."
        )

    # ------------------------------------------------------------------------
    # Die bestätigte Drift muss operativ als Anomalie klassifiziert werden.
    # ------------------------------------------------------------------------

    last = confirmed[-1]

    if int(
        last.get(
            "pred_system_state",
            SYSTEM_STATE_NORMAL,
        )
    ) != SYSTEM_STATE_ANOMALY:
        raise AssertionError(
            "CONCEPT_DRIFT: Der bestätigte Concept Drift wurde "
            "nicht als ANOMALY klassifiziert."
        )

    if str(
        last.get(
            "pred_anomaly_type"
        )
    ) != ANOMALY_CONCEPT_DRIFT:
        raise AssertionError(
            "CONCEPT_DRIFT: Falscher Anomalietyp: "
            f"{last.get('pred_anomaly_type')}"
        )


# ============================================================================
# Validierung: Infrastructure Failure
# ============================================================================

def assert_infrastructure_failure(
    rows: List[Dict[str, Any]],
) -> None:
    """
    Prüft die finale Klassifikation eines kontrollierten
    Infrastrukturfehlers.
    """

    if not rows:
        raise AssertionError(
            "INFRASTRUCTURE_FAILURE: Keine Beobachtungen vorhanden."
        )

    last = rows[-1]

    # ------------------------------------------------------------------------
    # Systemzustand
    # ------------------------------------------------------------------------

    if int(
        last.get(
            "pred_system_state",
            SYSTEM_STATE_NORMAL,
        )
    ) != SYSTEM_STATE_ANOMALY:
        raise AssertionError(
            "INFRASTRUCTURE_FAILURE: Der Infrastrukturfehler "
            "wurde nicht als ANOMALY klassifiziert."
        )

    # ------------------------------------------------------------------------
    # Anomalietyp
    # ------------------------------------------------------------------------

    if str(
        last.get(
            "pred_anomaly_type"
        )
    ) != ANOMALY_INFRASTRUCTURE:
        raise AssertionError(
            "INFRASTRUCTURE_FAILURE: Falscher Anomalietyp: "
            f"{last.get('pred_anomaly_type')}"
        )

    # ------------------------------------------------------------------------
    # Infrastrukturstatus
    # ------------------------------------------------------------------------

    infrastructure_failure = bool(
        last.get(
            "pred_infrastructure_failure",
            False,
        )
    )

    if not infrastructure_failure:
        raise AssertionError(
            "INFRASTRUCTURE_FAILURE: Der Service meldet keinen "
            "Infrastrukturfehler."
        )


# ============================================================================
# Ausgabe einer Testzusammenfassung
# ============================================================================

def print_test_result(
    label: str,
    rows: List[Dict[str, Any]],
) -> None:
    """Gibt eine kompakte Zusammenfassung eines Tests aus."""

    if not rows:
        print(
            f"{label}: keine Beobachtungen"
        )
        return

    last = rows[-1]

    print()
    print(label)
    print("-" * 72)

    print(
        f"Beobachtungen:        "
        f"{len(rows)}"
    )

    print(
        f"Letzter Step:         "
        f"{last['step']}"
    )

    print(
        "Ground Truth:         "
        f"{last['ground_truth_system_state']} / "
        f"{last['ground_truth_anomaly_type']}"
    )

    print(
        "Prediction:           "
        f"{last['pred_system_state']} / "
        f"{last['pred_anomaly_type']}"
    )

    print(
        "Data Drift erkannt:   "
        f"{last.get('pred_drift_detected')}"
    )

    print(
        "Concept Drift:        "
        f"{last.get('pred_concept_drift_detected')}"
    )

    print(
        "Prediction Error:     "
        f"{last.get('pred_prediction_error')}"
    )

    print(
        "Rolling Error:        "
        f"{last.get('pred_rolling_prediction_error')}"
    )

    print(
        "Performance Degraded: "
        f"{last.get('pred_performance_degraded')}"
    )

    print(
        "Root Cause:           "
        f"{last.get('pred_root_cause')}"
    )

    print(
        "Trigger:              "
        f"{last.get('pred_trigger_feature')}"
    )


# ============================================================================
# Hauptprogramm
# ============================================================================

def main() -> None:
    """
    Führt die vollständige End-to-End-Validierung der vier finalen
    Szenarien durch.
    """

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("END-TO-END-EXPERIMENT – V16")
    print("=" * 80)

    print(
        "Szenarien: "
        "NORMAL / DATA_DRIFT / CONCEPT_DRIFT / "
        "INFRASTRUCTURE_FAILURE"
    )

    session = requests.Session()

    # ========================================================================
    # 1. FastAPI-Service prüfen
    # ========================================================================

    print(
        "\n[1/5] Prüfe FastAPI-Service ..."
    )

    try:
        health = request_health(
            session
        )

    except Exception as exc:
        raise RuntimeError(
            "FastAPI-Service nicht erreichbar. "
            "Starte zuerst: "
            "uvicorn app.anomaly_detector:app "
            "--host 0.0.0.0 --port 8000"
        ) from exc

    print(
        "Service: OK"
    )

    print(
        f"Model ready:       "
        f"{health.get('model_ready')}"
    )

    print(
        f"Calibration ready: "
        f"{health.get('calibration_ready')}"
    )

    if not health.get(
        "model_ready"
    ):
        raise RuntimeError(
            "Das Modell ist im FastAPI-Service nicht bereit."
        )

    if not health.get(
        "calibration_ready"
    ):
        raise RuntimeError(
            "Die Modellkalibrierung ist im FastAPI-Service "
            "nicht bereit."
        )

    # ========================================================================
    # Ingestor erzeugen
    # ========================================================================

    ingestor = create_ingestor()

    all_rows: List[Dict[str, Any]] = []

    # ========================================================================
    # 2. Test 1 – Normal
    # ========================================================================

    print(
        "\n[2/5] Test 1 – NORMAL ..."
    )

    request_reset(
        session
    )

    normal_observations = observations_for_steps(
        ingestor,
        range(0, 100),
    )

    rows = run_sequence(
        session,
        normal_observations,
        "NORMAL",
    )

    all_rows.extend(
        rows
    )

    print_test_result(
        "TEST 1 – NORMAL",
        rows,
    )

    assert_expected_prediction(
        rows,
        expected_system_state=SYSTEM_STATE_NORMAL,
        expected_anomaly_type=ANOMALY_NONE,
        label="TEST 1 – NORMAL",
    )

    # ========================================================================
    # 3. Test 2 – Reiner Data Drift
    # ========================================================================

    print(
        "\n[3/5] Test 2 – REINER DATA DRIFT ..."
    )

    request_reset(
        session
    )

    drift_start = int(
        ingestor.data_drift_start
    )

    data_drift_observations: List[Any] = []

    # ------------------------------------------------------------------------
    # Gesunder Kontext vor dem Data Drift.
    # ------------------------------------------------------------------------

    for step in range(
        max(
            0,
            drift_start - 200,
        ),
        drift_start,
    ):
        observation = ingestor._create_observation(
            step
        )

        data_drift_observations.append(
            observation
        )

    # ------------------------------------------------------------------------
    # Reiner Data Drift.
    #
    # Der Ingestor verändert die Eingangsverteilung P(X).
    # Das beobachtete Target Y wird nicht künstlich verändert.
    # ------------------------------------------------------------------------

    for step in range(
        drift_start,
        drift_start + 800,
    ):
        observation = ingestor._create_observation(
            step
        )

        data_drift_observations.append(
            observation
        )

    rows = run_sequence(
        session,
        data_drift_observations,
        "DATA_DRIFT",
    )

    all_rows.extend(
        rows
    )

    print_test_result(
        "TEST 2 – REINER DATA DRIFT",
        rows,
    )

    assert_pure_data_drift_is_normal(
        rows
    )

    assert_expected_prediction(
        rows,
        expected_system_state=SYSTEM_STATE_NORMAL,
        expected_anomaly_type=ANOMALY_NONE,
        label="TEST 2 – REINER DATA DRIFT",
    )

    # ========================================================================
    # 4. Test 3 – Concept Drift
    # ========================================================================

    print(
        "\n[4/5] Test 3 – CONCEPT DRIFT ..."
    )

    request_reset(
        session
    )

    concept_drift_observations: List[Any] = []

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # Der Startpunkt wird direkt aus dem offiziellen V16-Ingestor übernommen.
    #
    # Dadurch bleibt dieser E2E-Test mit derselben Szenariodefinition
    # ausgerichtet wie Benchmark und RCA-Evaluation.
    # ------------------------------------------------------------------------

    concept_drift_start = int(
        ingestor.concept_drift_start
    )

    # ------------------------------------------------------------------------
    # Gesunder Kontext unmittelbar vor dem Concept Drift.
    # ------------------------------------------------------------------------

    for step in range(
        max(
            0,
            concept_drift_start - 200,
        ),
        concept_drift_start,
    ):
        observation = ingestor._create_observation(
            step,
            include_concept_drift=True,
        )

        concept_drift_observations.append(
            observation
        )

    # ------------------------------------------------------------------------
    # Offizieller Concept-Drift-Bereich des Ingestors.
    #
    # Wichtig:
    # Der Concept Drift wird nicht mehr manuell über
    # apply_target_concept_drift() injiziert.
    #
    # _create_observation(..., include_concept_drift=True) erzeugt die
    # kontrollierte Änderung der Beziehung X -> Y. Die fünf Eingangsmerkmale
    # X bleiben dabei unverändert; das beobachtete Target Y ändert sich.
    #
    # Der FastAPI-Detektor muss diese Änderung ausschließlich über die
    # Verschlechterung der Vorhersageleistung erkennen.
    # ------------------------------------------------------------------------

    for step in range(
        concept_drift_start,
        concept_drift_start + 600,
    ):
        observation = ingestor._create_observation(
            step,
            include_concept_drift=True,
        )

        concept_drift_observations.append(
            observation
        )

    rows = run_sequence(
        session,
        concept_drift_observations,
        "CONCEPT_DRIFT",
    )

    all_rows.extend(
        rows
    )

    print_test_result(
        "TEST 3 – CONCEPT DRIFT",
        rows,
    )

    assert_concept_drift_detected(
        rows
    )

    assert_expected_prediction(
        rows,
        expected_system_state=SYSTEM_STATE_ANOMALY,
        expected_anomaly_type=ANOMALY_CONCEPT_DRIFT,
        label="TEST 3 – CONCEPT DRIFT",
    )

    # ========================================================================
    # 5. Test 4 – Infrastructure Failure
    # ========================================================================

    print(
        "\n[5/5] Test 4 – INFRASTRUCTURE FAILURE ..."
    )

    request_reset(
        session
    )

    infra_start = int(
        ingestor.infrastructure_start
    )

    infra_end = int(
        ingestor.infrastructure_end
    )

    infra_target = None

    # ------------------------------------------------------------------------
    # Suche nach einer Beobachtung mit einem kontrollierten
    # Infrastrukturfehler.
    # ------------------------------------------------------------------------

    for observation in (
        ingestor._create_observation(
            step,
            live_demo=True,
        )
        for step in range(
            infra_start,
            infra_end + 1,
        )
    ):
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

        cpu_failure = (
            math.isfinite(cpu)
            and cpu >= CPU_ALERT_PERCENT
        )

        memory_failure = (
            math.isfinite(memory)
            and memory >= MEMORY_ALERT_PERCENT
        )

        restart_failure = (
            restarts >= POD_RESTART_ALERT_COUNT
        )

        if (
            cpu_failure
            or memory_failure
            or restart_failure
        ):
            if all(
                math.isfinite(
                    float(
                        getattr(
                            observation,
                            feature,
                        )
                    )
                )
                for feature in FEATURES
            ):
                infra_target = observation
                break

    if infra_target is None:
        raise RuntimeError(
            "Im Infrastrukturfenster wurde kein konkreter "
            "Infrastrukturfehler gefunden."
        )

    # ------------------------------------------------------------------------
    # Gesunder Kontext unmittelbar vor dem Infrastrukturfehler.
    # ------------------------------------------------------------------------

    infra_context: List[Any] = []

    for step in range(
        max(
            0,
            infra_start - 30,
        ),
        infra_start,
    ):
        observation = ingestor._create_observation(
            step,
            live_demo=True,
        )

        if all(
            math.isfinite(
                float(
                    getattr(
                        observation,
                        feature,
                    )
                )
            )
            for feature in FEATURES
        ):
            infra_context.append(
                observation
            )

    rows = run_sequence(
        session,
        infra_context + [infra_target],
        "INFRASTRUCTURE_FAILURE",
    )

    all_rows.extend(
        rows
    )

    print(
        f"Gefundener Infrastruktur-Step: "
        f"{infra_target.step}"
    )

    print_test_result(
        "TEST 4 – INFRASTRUCTURE FAILURE",
        rows,
    )

    assert_infrastructure_failure(
        rows
    )

    assert_expected_prediction(
        rows,
        expected_system_state=SYSTEM_STATE_ANOMALY,
        expected_anomaly_type=ANOMALY_INFRASTRUCTURE,
        label="TEST 4 – INFRASTRUCTURE FAILURE",
    )

    # ========================================================================
    # Ergebnisdatei schreiben
    # ========================================================================

    fieldnames = sorted(
        {
            key
            for row in all_rows
            for key in row.keys()
        }
    )

    with RESULT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        writer.writerows(
            all_rows
        )

    # ========================================================================
    # Abschluss
    # ========================================================================

    print()
    print("=" * 80)
    print("END-TO-END-EXPERIMENT V16 ABGESCHLOSSEN")
    print("=" * 80)

    print(
        f"Gesamtbeobachtungen: "
        f"{len(all_rows)}"
    )

    print(
        f"Ergebnisdatei:       "
        f"{RESULT_FILE}"
    )


if __name__ == "__main__":
    main()