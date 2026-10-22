"""
Verbesserte RCA-Abbildungen für die Masterarbeit.

Dieses Skript verwendet ausschließlich die bereits erzeugte
RCA-Prediktionsdatei. Es führt keine neue Evaluation und kein Training durch.

Erzeugt werden drei zentrale RCA-Abbildungen:
1. Konfusionsmatrix der Anomalietypen
2. Leistungsbasierte Concept-Drift-Erkennung
3. Abgrenzung von Data Drift und Concept Drift

Die Darstellung folgt der finalen fachlichen Logik:
- Data Drift ist ein Verteilungssignal und keine Anomalie.
- Concept Drift wird über eine anhaltende Verschlechterung der
  Vorhersageleistung erkannt.
- Der Systemzustand wird separat betrachtet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Pfade und Parameter
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "results" / "rca_predictions.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "results" / "figures"
DEFAULT_RCA_JSON = SCRIPT_DIR / "results" / "rca_results.json"

# Absoluter Schwellenwert des finalen Concept-Drift-Monitors.
CONCEPT_DRIFT_THRESHOLD = 0.65


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, output_path: Path) -> None:
    """Speichert eine publikationsgeeignete PNG-Abbildung."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {output_path}")


def load_data(csv_path: Path) -> pd.DataFrame:
    """Lädt die RCA-Daten und prüft die benötigten Spalten."""
    if not csv_path.exists():
        raise FileNotFoundError(f"RCA-Datei nicht gefunden: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    required = {
        "step",
        "scenario",
        "data_drift_signal",
        "concept_drift_signal",
        "predicted_system_state",
        "rolling_prediction_error",
        "performance_confirmation_count",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Folgende Spalten fehlen in der RCA-Datei:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )

    return df.sort_values("step").reset_index(drop=True)


def contiguous_runs(mask: pd.Series, steps: pd.Series) -> list[tuple[int, int]]:
    """Ermittelt zusammenhängende Zeitbereiche für ein boolesches Signal."""
    values = mask.astype(bool).to_numpy()
    x = steps.to_numpy()

    if len(values) == 0:
        return []

    runs: list[tuple[int, int]] = []
    start_index: int | None = None

    for index, value in enumerate(values):
        if value and start_index is None:
            start_index = index
        elif not value and start_index is not None:
            runs.append((int(x[start_index]), int(x[index - 1])))
            start_index = None

    if start_index is not None:
        runs.append((int(x[start_index]), int(x[-1])))

    return runs


def add_scenario_background(
    ax: plt.Axes,
    df: pd.DataFrame,
    y_bottom: float,
    y_top: float,
) -> None:
    """Hebt die vier experimentellen Szenarien dezent hervor."""
    scenario_order = [
        ("normal", "Normal"),
        ("data_drift", "Data Drift"),
        ("concept_drift", "Concept Drift"),
        ("infrastructure_failure", "Infrastructure Failure"),
    ]

    for scenario, label in scenario_order:
        subset = df[df["scenario"].eq(scenario)]
        if subset.empty:
            continue

        start = int(subset["step"].min())
        end = int(subset["step"].max())
        ax.axvspan(start, end, alpha=0.08)

        # Die Beschriftung wird nur für die drei relevanten Störungsphasen
        # verwendet, damit die Grafik nicht mit Text überladen wird.
        if scenario != "normal":
            ax.text(
                (start + end) / 2,
                y_top - (y_top - y_bottom) * 0.08,
                label,
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
            )


def add_phase_boundaries(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Markiert die Übergänge zwischen den experimentellen Phasen."""
    scenario_series = df["scenario"].astype(str)
    changes = scenario_series.ne(scenario_series.shift())

    for step in df.loc[changes, "step"].iloc[1:]:
        ax.axvline(int(step), linestyle=":", linewidth=0.8, alpha=0.6)


# ---------------------------------------------------------------------------
# 1. Verbesserte Concept-Drift-Leistungsdarstellung
# ---------------------------------------------------------------------------

def figure_concept_drift_performance(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Zeigt die leistungsbasierte Concept-Drift-Erkennung mit möglichst wenig
    visuellem Rauschen.

    Oben wird der gleitende Vorhersagefehler mit der festen Schwelle
    dargestellt. Unten werden Ground Truth und erkannte Signalbereiche
    ausschließlich innerhalb der Concept-Drift-Phase gezeigt.
    """
    concept_gt = df["scenario"].eq("concept_drift")

    if not concept_gt.any():
        raise ValueError("Keine Concept-Drift-Phase in der RCA-Datei gefunden.")

    concept_df = df.loc[concept_gt].copy()
    gt_start = int(concept_df["step"].min())
    gt_end = int(concept_df["step"].max())

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=False,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # -----------------------------------------------------------------------
    # Oberer Bereich: Leistungsabfall
    # -----------------------------------------------------------------------
    ax = axes[0]

    # Nur die Concept-Drift-Phase darstellen. Dadurch bleibt die Abbildung
    # auf die eigentliche RCA-Frage fokussiert und vermeidet unnötiges Rauschen.
    ax.plot(
        concept_df["step"],
        concept_df["rolling_prediction_error"],
        linewidth=1.2,
        label="Rolling Prediction Error",
    )

    ax.axhline(
        CONCEPT_DRIFT_THRESHOLD,
        linestyle="--",
        linewidth=1.2,
        label=f"Erkennungsschwelle = {CONCEPT_DRIFT_THRESHOLD:.2f}",
    )

    ax.axvspan(
        gt_start,
        gt_end,
        alpha=0.14,
        label="Ground Truth: Concept Drift",
    )

    ax.axvline(
        gt_start,
        linestyle=":",
        linewidth=0.9,
        alpha=0.65,
    )
    ax.axvline(
        gt_end,
        linestyle=":",
        linewidth=0.9,
        alpha=0.65,
    )

    ax.set_ylabel("Rolling Prediction Error")
    ax.set_ylim(0, 1.05)
    ax.set_title("RCA: Leistungsbasierte Erkennung von Concept Drift")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper left")

    # -----------------------------------------------------------------------
    # Unterer Bereich: Nur die Concept-Drift-Phase
    # -----------------------------------------------------------------------
    signal_ax = axes[1]

    concept_steps = concept_df["step"].to_numpy()
    detected = concept_df["concept_drift_signal"].astype(bool).to_numpy()

    # Ground Truth als durchgehende Referenz.
    signal_ax.axhspan(
        0.52,
        1.0,
        alpha=0.16,
        label="Ground Truth: Concept Drift",
    )

    # Erkannte Bereiche als einzelne horizontale Segmente.
    for start_step, end_step in contiguous_runs(
        pd.Series(detected),
        pd.Series(concept_steps),
    ):
        signal_ax.axvspan(
            start_step,
            end_step,
            ymin=0.0,
            ymax=0.48,
            alpha=0.60,
        )

    # Eine schlichte binäre Linie macht die Detektion eindeutig.
    signal_ax.step(
        concept_steps,
        detected.astype(int),
        where="post",
        linewidth=1.4,
        label="Concept-Drift-Signal",
    )

    # Erste Detektion bestimmen.
    detection_steps = concept_steps[detected]
    if len(detection_steps) > 0:
        first_detection = int(detection_steps[0])
        delay = first_detection - gt_start

        signal_ax.scatter(
            [first_detection],
            [1.0],
            s=42,
            zorder=5,
        )
        signal_ax.annotate(
            f"Erste Detektion: +{delay} Beobachtungen",
            xy=(first_detection, 1.0),
            xytext=(
                first_detection + max(80, int((gt_end - gt_start) * 0.08)),
                0.72,
            ),
            arrowprops={"arrowstyle": "->"},
            fontsize=10,
        )

    signal_ax.set_xlim(gt_start, gt_end)
    signal_ax.set_ylim(-0.05, 1.12)
    signal_ax.set_yticks([0, 1])
    signal_ax.set_yticklabels(["kein Signal", "erkannt"])
    signal_ax.set_xlabel("Beobachtungsindex")
    signal_ax.set_ylabel("Detektion")
    signal_ax.grid(axis="x", alpha=0.15)
    signal_ax.legend(loc="upper right")

    save_figure(
        fig,
        output_dir / "rca_concept_drift_performance_improved.png",
    )


# ---------------------------------------------------------------------------
# 2. Verbesserte Abgrenzung Data Drift / Concept Drift
# ---------------------------------------------------------------------------

def figure_data_drift_vs_concept_drift(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Stellt die fachliche Trennung von Data Drift und Concept Drift klar dar.

    Data Drift wird als Änderung von P(X) dargestellt und ist im Versuchsdesign
    keine Anomalie. Concept Drift wird über die Änderung der X→Y-Beziehung
    und den daraus resultierenden Leistungsabfall bewertet.

    Die Grafik soll bewusst einfach interpretierbar sein:
    - Blau: erkanntes Data-Drift-Signal
    - Orange: erkanntes Concept-Drift-Signal
    - Grün: Ground-Truth-Anomaliezustand
    """
    phases = ["data_drift", "concept_drift"]
    phase_labels = ["Data Drift", "Concept Drift"]

    data_rates: list[float] = []
    concept_rates: list[float] = []

    for phase in phases:
        subset = df[df["scenario"].eq(phase)]

        if subset.empty:
            data_rates.append(0.0)
            concept_rates.append(0.0)
            continue

        data_rates.append(
            float(subset["data_drift_signal"].astype(bool).mean()) * 100.0
        )
        concept_rates.append(
            float(subset["concept_drift_signal"].astype(bool).mean()) * 100.0
        )

    # Ground Truth des Versuchsdesigns:
    # Data Drift allein ist keine Anomalie; Concept Drift ist eine Anomalie.
    ground_truth = [0.0, 100.0]

    x = np.arange(len(phases))
    width = 0.23

    # Constrained Layout verhindert Überlappungen zwischen Achse,
    # Legende und Erläuterung.
    fig, ax = plt.subplots(
        figsize=(12.5, 7.6),
        constrained_layout=True,
    )

    bars_data = ax.bar(
        x - width,
        data_rates,
        width,
        label="Data-Drift-Signal",
    )

    bars_concept = ax.bar(
        x,
        concept_rates,
        width,
        label="Concept-Drift-Signal",
    )

    bars_gt = ax.bar(
        x + width,
        ground_truth,
        width,
        label="Ground Truth: Anomalie",
    )

    # Prozentwerte oberhalb der Balken.
    for bars in (bars_data, bars_concept, bars_gt):
        for bar in bars:
            value = bar.get_height()

            if value == 0:
                offset = 0.8
            elif value >= 95:
                offset = 1.2
            else:
                offset = 1.2

            ax.annotate(
                f"{value:.1f} %",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels, fontsize=12)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Anteil [%]", fontsize=11)
    ax.set_xlabel("Experimentelle Phase", fontsize=11, labelpad=8)
    ax.set_title(
        "RCA: Data Drift ≠ Concept Drift",
        fontsize=15,
        pad=12,
    )

    ax.grid(axis="y", alpha=0.2)

    # Die Legende wird außerhalb des Plotbereichs platziert.
    # Dadurch bleibt insbesondere die 100-%-Beschriftung des Ground Truth
    # vollständig sichtbar und wird nicht von der Legende überdeckt.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        framealpha=0.95,
    )

    save_figure(
        fig,
        output_dir / "rca_data_drift_vs_concept_drift_final.png",
    )


# ---------------------------------------------------------------------------
# 3. Verbesserte Konfusionsmatrix der Anomalietypen
# ---------------------------------------------------------------------------

def figure_anomaly_type_confusion_matrix(
    json_path: Path,
    output_dir: Path,
) -> None:
    """Zeigt die finale Anomalietyp-Klassifikation zeilenweise normalisiert."""
    if not json_path.exists():
        raise FileNotFoundError(f"RCA-Ergebnisdatei nicht gefunden: {json_path}")

    with json_path.open("r", encoding="utf-8") as handle:
        results = json.load(handle)

    section = results.get("final_anomaly_type")
    if not section:
        raise ValueError("Abschnitt 'final_anomaly_type' fehlt in rca_results.json.")

    labels = section["labels"]
    matrix = np.asarray(section["confusion_matrix"], dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0
    )

    display_labels = ["NONE", "CONCEPT\nDRIFT", "INFRASTRUCTURE\nFAILURE"]

    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    image = ax.imshow(normalized, vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(display_labels)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(display_labels)
    ax.set_xlabel("Vorhersage")
    ax.set_ylabel("Ground Truth")
    ax.set_title("RCA: Konfusionsmatrix der Anomalietypen")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            count = int(matrix[row, col])
            percentage = normalized[row, col] * 100
            text_color = "white" if normalized[row, col] > 0.55 else "black"
            ax.text(
                col, row, f"{count:,}\n({percentage:.1f} %)",
                ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=text_color,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Anteil innerhalb der Ground-Truth-Klasse")

    ax.text(
        0.5, -0.13,
        "Zeilenweise normalisiert; absolute Fallzahlen stehen zusätzlich in den Zellen.",
        transform=ax.transAxes, ha="center", va="top", fontsize=9.5,
    )

    save_figure(fig, output_dir / "rca_anomaly_type_confusion_matrix_improved.png")

# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def main() -> None:
    """Erzeugt die drei verbesserten RCA-Abbildungen."""
    parser = argparse.ArgumentParser(
        description="Erzeugt drei publikationsgeeignete RCA-Figuren aus den bestehenden RCA-Ergebnissen."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Pfad zur RCA-Prediktionsdatei.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_RCA_JSON,
        help="Pfad zur bestehenden rca_results.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Ausgabeverzeichnis für die PNG-Dateien.",
    )
    args = parser.parse_args()

    df = load_data(args.csv)

    figure_concept_drift_performance(df, args.output)
    figure_data_drift_vs_concept_drift(df, args.output)
    figure_anomaly_type_confusion_matrix(args.json, args.output)

    print("\n[OK] Drei verbesserte RCA-Abbildungen wurden erzeugt.")
    print(f"[OK] Ausgabeverzeichnis: {args.output.resolve()}")


if __name__ == "__main__":
    main()
