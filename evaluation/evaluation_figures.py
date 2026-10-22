"""
Génération des figures finales pour l'évaluation expérimentale.

Ce script lit uniquement les résultats déjà calculés dans
evaluation/results/. Aucune expérience n'est relancée.
"""

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_json(filename):
    """Lädt eine JSON-Datei aus dem Ergebnisverzeichnis."""
    path = RESULTS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Die Ergebnisdatei wurde nicht gefunden: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_figure(filename):
    """Speichert die aktuelle Matplotlib-Abbildung."""
    path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gespeichert: {path}")


def get_confusion_matrix(section):
    """Sucht eine Konfusionsmatrix in verschiedenen möglichen Strukturen."""
    if not isinstance(section, dict):
        return None

    for key in ("confusion_matrix", "confusion", "matrix"):
        value = section.get(key)
        if value is not None:
            return np.asarray(value)

    for parent_key in (
        "classification",
        "classification_metrics",
        "confusion_metrics",
        "results",
        "metrics",
    ):
        parent = section.get(parent_key)

        if isinstance(parent, dict):
            for key in ("confusion_matrix", "confusion", "matrix"):
                value = parent.get(key)
                if value is not None:
                    return np.asarray(value)

    return None


def get_labels(section, default_labels, matrix):
    """Ermittelt die Klassenbezeichnungen passend zur Matrix."""
    labels = None

    if isinstance(section, dict):
        labels = section.get("labels")

        if labels is None and isinstance(section.get("classification"), dict):
            labels = section["classification"].get("labels")

    if labels is None:
        labels = default_labels

    labels = list(labels)

    if matrix is not None and matrix.ndim >= 2:
        labels = labels[:matrix.shape[0]]

    return labels


# ---------------------------------------------------------------------------
# 1. Method performance
# ---------------------------------------------------------------------------

def plot_method_performance(benchmark):
    """Vergleicht Precision, Recall und F1-Score."""

    methods_data = benchmark["method_metrics"]

    methods = []
    precision = []
    recall = []
    f1 = []

    for method, values in methods_data.items():

        if "anomaly_metrics" in values:
            values = values["anomaly_metrics"]

        if not all(
            key in values
            for key in ("precision", "recall", "f1_score")
        ):
            continue

        methods.append(method)
        precision.append(float(values["precision"]))
        recall.append(float(values["recall"]))
        f1.append(float(values["f1_score"]))

    x = np.arange(len(methods))
    width = 0.25

    plt.figure(figsize=(13, 7))

    plt.bar(
        x - width,
        precision,
        width,
        label="Precision",
        hatch="//",
    )

    plt.bar(
        x,
        recall,
        width,
        label="Recall",
        hatch="..",
    )

    plt.bar(
        x + width,
        f1,
        width,
        label="F1-Score",
        hatch="xx",
    )

    plt.xticks(x, methods, rotation=25, ha="right")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("Vergleich der Erkennungsleistung der Verfahren")
    plt.legend()

    save_figure("method_performance.png")


# ---------------------------------------------------------------------------
# 2. False Positive Rate
# ---------------------------------------------------------------------------

def plot_normal_fpr(benchmark):
    """Vergleicht die False-Positive-Rate auf Normalbeobachtungen."""

    methods_data = benchmark["method_metrics"]

    methods = []
    fpr = []

    for method, values in methods_data.items():

        if "anomaly_metrics" in values:
            values = values["anomaly_metrics"]

        value = values.get("false_positive_rate_normal")

        if value is None:
            value = values.get("false_positive_rate")

        if value is None:
            continue

        methods.append(method)
        fpr.append(float(value))

    x = np.arange(len(methods))

    plt.figure(figsize=(13, 7))

    plt.bar(x, fpr, hatch="//")

    plt.xticks(x, methods, rotation=25, ha="right")
    plt.ylabel("False-Positive-Rate")
    plt.ylim(0, 1.05)
    plt.title("False-Positive-Rate auf Normalbeobachtungen")

    save_figure("normal_false_positive_rate.png")


# ---------------------------------------------------------------------------
# 3. Detection delay
# ---------------------------------------------------------------------------

def plot_detection_delay(benchmark):
    """Vergleicht die Detektionsverzögerung der Verfahren."""

    delays = benchmark["detection_delay"]

    methods = []
    drift_delay = []
    infra_delay = []

    for method, values in delays.items():

        drift = values.get("concept_drift", {}).get(
            "delay_observations"
        )

        infra = values.get("infrastructure_failure", {}).get(
            "delay_observations"
        )

        methods.append(method)
        drift_delay.append(
            float(drift) if drift is not None else np.nan
        )
        infra_delay.append(
            float(infra) if infra is not None else np.nan
        )

    x = np.arange(len(methods))
    width = 0.35

    plt.figure(figsize=(13, 7))

    plt.bar(
        x - width / 2,
        drift_delay,
        width,
        label="Concept Drift",
        hatch="//",
    )

    plt.bar(
        x + width / 2,
        infra_delay,
        width,
        label="Infrastrukturfehler",
        hatch="..",
    )

    plt.xticks(x, methods, rotation=25, ha="right")
    plt.ylabel("Detektionsverzögerung [Beobachtungen]")
    plt.title("Detektionsverzögerung der Verfahren")
    plt.legend()

    save_figure("detection_delay.png")


# ---------------------------------------------------------------------------
# 4. Latency
# ---------------------------------------------------------------------------

def plot_latency(benchmark):
    """Vergleicht die mittlere Verarbeitungszeit pro Beobachtung."""

    latency = benchmark["latency_ms"]

    methods = []
    values = []

    for method, data in latency.items():

        value = data.get("mean_ms_per_valid_observation")

        if value is None:
            value = data.get("mean_ms_per_observation")

        if value is None:
            continue

        methods.append(method)
        values.append(float(value))

    x = np.arange(len(methods))

    plt.figure(figsize=(11, 7))

    plt.bar(x, values, hatch="//")

    plt.xticks(x, methods, rotation=20, ha="right")
    plt.ylabel("Mittlere Verarbeitungszeit [ms/Beobachtung]")
    plt.title("Verarbeitungszeit der Erkennungsverfahren")

    save_figure("method_latency.png")


# ---------------------------------------------------------------------------
# 5. System-state confusion matrix
# ---------------------------------------------------------------------------

def plot_system_state_confusion_matrix(rca):
    """Erstellt die Konfusionsmatrix des binären Systemzustands."""

    section = rca.get("final_system_state", {})
    matrix = get_confusion_matrix(section)

    if matrix is None:
        for key in (
            "system_state_confusion_matrix",
            "system_state_confusion",
        ):
            if key in rca:
                matrix = np.asarray(rca[key])
                break

    if matrix is None:
        print(
            "Übersprungen: Keine Konfusionsmatrix für "
            "final_system_state gefunden."
        )
        return

    if matrix.shape != (2, 2):
        print(
            "Übersprungen: Die Systemzustandsmatrix hat nicht die "
            "erwartete Form 2x2."
        )
        return

    # Die internen Codes 0 und 1 werden in der Abbildung bewusst durch
    # die semantischen Klassenbezeichnungen ersetzt.
    labels = ["NORMAL", "ANOMALY"]

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix)

    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)

    plt.xlabel("Vorhersage")
    plt.ylabel("Ground Truth")
    plt.title("Konfusionsmatrix des Systemzustands")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(
                j,
                i,
                f"{int(matrix[i, j]):,}".replace(",", " "),
                ha="center",
                va="center",
            )

    save_figure("system_state_confusion_matrix.png")


# ---------------------------------------------------------------------------
# 6. Anomaly type confusion matrix
# ---------------------------------------------------------------------------

def plot_anomaly_type_confusion_matrix(rca):
    """Erstellt die Konfusionsmatrix des Anomalietyps."""

    section = rca.get("final_anomaly_type", {})
    matrix = get_confusion_matrix(section)

    if matrix is None:
        for key in (
            "anomaly_type_confusion_matrix",
            "anomaly_type_confusion",
        ):
            if key in rca:
                matrix = np.asarray(rca[key])
                break

    if matrix is None:
        print(
            "Übersprungen: Keine Konfusionsmatrix für "
            "final_anomaly_type gefunden."
        )
        return

    if matrix.shape != (3, 3):
        print(
            "Übersprungen: Die Anomalietyp-Matrix hat nicht die "
            "erwartete Form 3x3."
        )
        return

    # Die semantischen Anomalietypen werden unabhängig von einer
    # numerischen internen Codierung direkt in der Abbildung dargestellt.
    labels = [
        "NONE",
        "ANOMALY_CONCEPT_DRIFT",
        "INFRASTRUCTURE_FAILURE",
    ]

    plt.figure(figsize=(9, 7))
    plt.imshow(matrix)

    plt.xticks(
        range(len(labels)),
        labels,
        rotation=20,
        ha="right",
    )
    plt.yticks(range(len(labels)), labels)

    plt.xlabel("Vorhersage")
    plt.ylabel("Ground Truth")
    plt.title("Konfusionsmatrix des Anomalietyps")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(
                j,
                i,
                f"{int(matrix[i, j]):,}".replace(",", " "),
                ha="center",
                va="center",
            )

    save_figure("anomaly_type_confusion_matrix.png")


# ---------------------------------------------------------------------------
# 7. Infrastructure RCA
# ---------------------------------------------------------------------------

def plot_infrastructure_rca(rca):
    """Visualisiert die Root-Cause-Lokalisierung von Infrastrukturfehlern."""

    data = rca.get("infrastructure_trigger_rca")

    if not isinstance(data, dict):
        print(
            "Übersprungen: infrastructure_trigger_rca nicht vorhanden."
        )
        return

    matrix = get_confusion_matrix(data)

    if matrix is None:
        print(
            "Übersprungen: Keine Konfusionsmatrix für "
            "infrastructure_trigger_rca gefunden."
        )
        return

    correct = np.diag(matrix)

    labels = get_labels(
        data,
        [
            "HIGH_CPU",
            "MISSING_VALUE",
            "EXTREME_VALUE",
        ],
        matrix,
    )

    x = np.arange(len(labels))

    plt.figure(figsize=(10, 6))

    plt.bar(
        x,
        correct[:len(labels)],
        hatch="//",
    )

    plt.xticks(
        x,
        labels,
        rotation=20,
        ha="right",
    )

    plt.ylabel("Korrekt lokalisierte Beobachtungen")
    plt.title(
        "Root-Cause-Lokalisierung von Infrastrukturfehlern"
    )

    save_figure("infrastructure_root_cause.png")


# ---------------------------------------------------------------------------
# 8. Drift anomaly feature localization
# ---------------------------------------------------------------------------

def plot_concept_drift_performance(rca):
    """Visualisiert die Leistungsüberwachung während des Concept Drifts."""

    data = rca.get("concept_drift_temporal_analysis")

    if not isinstance(data, dict) or not data.get("available", False):
        print(
            "Übersprungen: Keine zeitliche Analyse des Concept Drifts vorhanden."
        )
        return

    labels = [
        "Rolling Error",
        "Concept-Drift-Erkennung",
    ]

    values = [
        float(data.get("rolling_error_mean", 0.0)),
        float(data.get("concept_drift_recall", 0.0)),
    ]

    x = np.arange(len(labels))

    plt.figure(figsize=(9, 6))

    plt.bar(x, values, hatch="//")

    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("Wert")
    plt.ylim(0, 1.05)
    plt.title("Leistungsbasierte Erkennung von Concept Drift")

    for index, value in enumerate(values):
        plt.text(
            index,
            min(value + 0.02, 1.02),
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    save_figure("concept_drift_performance.png")


def plot_concept_drift_detection_delay(rca):
    """Visualisiert die Detektionsverzögerung des Concept Drifts."""

    data = rca.get("concept_drift_temporal_analysis")

    if not isinstance(data, dict) or not data.get("available", False):
        print(
            "Übersprungen: Keine zeitliche Analyse des Concept Drifts vorhanden."
        )
        return

    delay = data.get("concept_drift_detection_delay")

    if delay is None:
        print(
            "Übersprungen: Keine Detektionsverzögerung für Concept Drift vorhanden."
        )
        return

    plt.figure(figsize=(8, 6))

    plt.bar(
        ["Concept Drift"],
        [float(delay)],
        hatch="//",
    )

    plt.ylabel("Detektionsverzögerung [Beobachtungen]")
    plt.title("Detektionsverzögerung des Concept Drifts")

    plt.text(
        0,
        float(delay),
        f"{int(delay)}",
        ha="center",
        va="bottom",
    )

    save_figure("concept_drift_detection_delay.png")


# ---------------------------------------------------------------------------
# 9. RCA metrics overview
# ---------------------------------------------------------------------------

def plot_rca_metrics(rca):
    """Vergleicht die zentralen RCA-Metriken."""

    metrics = {}

    sections = [
        ("Systemzustand", "final_system_state"),
        ("Anomalietyp", "final_anomaly_type"),
        ("Infrastruktur-RCA", "infrastructure_trigger_rca"),
        ("Concept Drift", "concept_drift_detection"),
    ]

    for label, key in sections:

        section = rca.get(key)

        if isinstance(section, dict) and "accuracy" in section:
            metrics[label] = float(section["accuracy"])

    if not metrics:
        print(
            "Übersprungen: Keine RCA-Accuracy-Werte gefunden."
        )
        return

    labels = list(metrics.keys())
    values = list(metrics.values())

    x = np.arange(len(labels))

    plt.figure(figsize=(11, 7))

    plt.bar(x, values, hatch="//")

    plt.xticks(
        x,
        labels,
        rotation=20,
        ha="right",
    )

    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.title(
        "Genauigkeit der Zustands- und Anomalieklassifikation"
    )

    save_figure("rca_accuracy_overview.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Erzeugt alle zentralen Evaluationsabbildungen."""

    print("=" * 70)
    print("Generierung der Evaluationsabbildungen")
    print("=" * 70)

    benchmark = load_json("final_benchmark_summary.json")
    rca = load_json("rca_results.json")

    plot_method_performance(benchmark)
    plot_normal_fpr(benchmark)
    plot_detection_delay(benchmark)
    plot_latency(benchmark)

    plot_system_state_confusion_matrix(rca)
    plot_anomaly_type_confusion_matrix(rca)
    plot_infrastructure_rca(rca)
    plot_concept_drift_performance(rca)
    plot_concept_drift_detection_delay(rca)
    plot_rca_metrics(rca)

    print()
    print("=" * 70)
    print("Alle Abbildungen wurden erfolgreich erzeugt.")
    print(f"Ausgabeverzeichnis: {FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
