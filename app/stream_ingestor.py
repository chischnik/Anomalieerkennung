"""
Experimenteller Elec2-Datenstrom für die Laufzeitvalidierung.

Der Datenstrom trennt drei für die Evaluation relevante Phänomene:

1. Einen zeitlich anhaltenden multivariaten Feature-/Data-Drift.
2. Einen kontrolliert veränderten Zusammenhang P(Y|X) für Concept Drift.
3. Infrastrukturfehler mit Missing Data, Extremwerten und
   Ressourcenproblemen.

Wichtig für die finale Validierung:

- Der operative Systemzustand besitzt nur zwei Zustände:
      0 = Normal
      1 = Anomaly

- Infrastructure Failure ist kein dritter Systemzustand.
  Er wird als separater Anomalietyp mit eigener Root Cause behandelt.

- Data Drift verändert die Eingabeverteilung P(X), verändert aber nicht
  das beobachtete Target Y.

- Concept Drift wird ausschließlich über
  apply_target_concept_drift() erzeugt. Dabei wird die Beziehung X -> Y
  gezielt verändert.

- Das Feld ``classification`` bleibt aus Kompatibilitätsgründen erhalten.
  Bei diesem Feld handelt es sich um den experimentellen Szenariocode des
  Ingestors und nicht um die operative Systemzustandsklassifikation.

Kommentare und Docstrings sind auf Deutsch.
"""

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from app import config
from app.config import (
    CLASS_CONCEPT_DRIFT,
    CLASS_DATA_DRIFT,
    CLASS_INFRASTRUCTURE,
    CLASS_NORMAL,
    SYSTEM_STATE_NORMAL,
    SYSTEM_STATE_ANOMALY,
    DATA_DRIFT_END_RATIO,
    DATA_DRIFT_MAX_SHIFT,
    DATA_DRIFT_START_RATIO,
    CONCEPT_DRIFT_END_RATIO,
    CONCEPT_DRIFT_START_RATIO,
    DATA_PATH,
    FEATURE_COLUMNS,
    INFRA_CPU_END_PERCENT,
    INFRA_CPU_START_PERCENT,
    INFRA_MEMORY_END_PERCENT,
    INFRA_MEMORY_START_PERCENT,
    INFRA_RESTART_PROGRESS,
    INFRASTRUCTURE_END_RATIO,
    INFRASTRUCTURE_START_RATIO,
    INFRA_EXTREME_RATE,
    INFRA_MISSING_RATE,
    INFRA_MISSING_DATA,
    INFRA_VALUE_EXTREME,
    RANDOM_STATE,
    ROOT_CAUSE_CONCEPT_DRIFT,
    ROOT_CAUSE_INFRASTRUCTURE,
    ROOT_CAUSE_NORMAL,
    SCENARIO_DATA_DRIFT,
    SCENARIO_CONCEPT_DRIFT,
    SCENARIO_INFRASTRUCTURE_FAILURE,
    SCENARIO_NORMAL,
    STREAM_INTERVAL_SECONDS,
    TARGET_COLUMN,
)


# ============================================================================
# Szenario- und Anomalietypen
# ============================================================================

# Die experimentellen Szenarioklassen stammen aus app.config und werden
# bewusst vom operativen Systemzustand getrennt behandelt.

ANOMALY_NONE = "NONE"
ANOMALY_CONCEPT_DRIFT = "ANOMALY_CONCEPT_DRIFT"
ANOMALY_INFRASTRUCTURE = "INFRASTRUCTURE_FAILURE"


# ============================================================================
# Datenstruktur einer Beobachtung
# ============================================================================

@dataclass
class StreamObservation:
    step: int
    cycle: int
    cycle_step: int
    timestamp: float
    date: float
    day: int
    period: float

    nswprice: float
    nswdemand: float
    vicprice: float
    vicdemand: float
    transfer: float

    # Die Elec2-Zielklasse wird unverändert aus dem vorbereiteten CSV
    # übernommen. Je nach Datensatz lautet sie beispielsweise UP oder DOWN.
    target: Any
    target_original: Any
    concept_drift_target_changed: bool

    original_nswprice: float
    original_nswdemand: float
    original_vicprice: float
    original_vicdemand: float
    original_transfer: float

    scenario: str

    # Kompatibilitätsfeld für bestehende experimentelle Auswertungen.
    # 0 = Normal, 1 = Data Drift, 2 = Concept Drift,
    # 3 = Infrastructure Failure.
    classification: int

    # Operativer Systemzustand:
    # 0 = NORMAL
    # 1 = ANOMALY
    system_state: int

    anomaly_type: str

    root_cause: str
    trigger_feature: str
    phase_progress: float

    infrastructure_cpu_load: float
    infrastructure_memory_load: float
    infrastructure_restarts: int

    missing_value_injected: bool
    extreme_value_injected: bool
    schema_error_injected: bool


# ============================================================================
# Elec2 Stream Ingestor
# ============================================================================

class Elec2StreamIngestor:
    """Erzeugt einen reproduzierbaren multivariaten Elec2-Datenstrom."""

    def __init__(
        self,
        data_path=DATA_PATH,
        interval_seconds: float = STREAM_INTERVAL_SECONDS,
        random_state: int = RANDOM_STATE,
    ) -> None:

        self.data_path = Path(data_path)

        self.interval_seconds = max(
            0.0,
            float(interval_seconds),
        )

        self.random_state = int(
            random_state
        )

        # Separate Zufallsquelle für reproduzierbare Szenario-Injektionen.
        self.rng = np.random.RandomState(
            self.random_state
        )

        self.data = self._load_data()

        self.total_rows = len(
            self.data
        )

        self.data_drift_start = int(
            self.total_rows
            * DATA_DRIFT_START_RATIO
        )

        self.data_drift_end = int(
            self.total_rows
            * DATA_DRIFT_END_RATIO
        )

        self.concept_drift_start = int(
            self.total_rows
            * CONCEPT_DRIFT_START_RATIO
        )

        self.concept_drift_end = int(
            self.total_rows
            * CONCEPT_DRIFT_END_RATIO
        )

        self.infrastructure_start = int(
            self.total_rows
            * INFRASTRUCTURE_START_RATIO
        )

        self.infrastructure_end = int(
            self.total_rows
            * INFRASTRUCTURE_END_RATIO
        )

        if not (
            0
            <= self.data_drift_start
            < self.data_drift_end
            <= self.concept_drift_start
            < self.concept_drift_end
            < self.infrastructure_start
            < self.infrastructure_end
            <= self.total_rows
        ):
            raise ValueError(
                "Die berechneten Phasenindizes sind für die aktuelle "
                "Datensatzgröße nicht gültig."
            )

        # Infrastruktur-Injektionen werden ausschließlich innerhalb der
        # Infrastructure-Phase zugelassen.
        self.missing_injection = (
            self.rng.rand(
                self.total_rows
            )
            < INFRA_MISSING_RATE
        )

        self.extreme_injection = (
            self.rng.rand(
                self.total_rows
            )
            < INFRA_EXTREME_RATE
        )

        indices = np.arange(
            self.total_rows
        )

        infrastructure_mask = (
            (indices >= self.infrastructure_start)
            & (indices < self.infrastructure_end)
        )

        self.missing_injection &= infrastructure_mask
        self.extreme_injection &= infrastructure_mask

    # =========================================================================
    # Daten laden
    # =========================================================================

    def _load_data(self) -> pd.DataFrame:
        """Lädt und validiert den vollständigen Elec2-Datensatz."""

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Die Elec2-Datei wurde nicht gefunden: "
                f"{self.data_path}"
            )

        df = pd.read_csv(
            self.data_path
        )

        required = [
            "date",
            "day",
            "period",
            *FEATURE_COLUMNS,
            TARGET_COLUMN,
        ]

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Folgende erforderliche Spalten fehlen: {missing}"
            )

        numeric = [
            "date",
            "period",
            *FEATURE_COLUMNS,
        ]

        for column in numeric:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        if df[numeric].isna().any().any():
            raise ValueError(
                "Der Quelldatensatz enthält ungültige numerische Werte."
            )

        return df.reset_index(
            drop=True
        )

    # =========================================================================
    # Phasenfortschritt
    # =========================================================================

    @staticmethod
    def _phase_progress(
        step: int,
        start: int,
        end: int,
    ) -> float:
        """Berechnet den normierten Fortschritt einer Phase."""

        duration = max(
            1,
            end - start,
        )

        return float(
            np.clip(
                (step - start) / duration,
                0.0,
                1.0,
            )
        )

    # =========================================================================
    # Szenario bestimmen
    # =========================================================================

    def _scenario(
        self,
        step: int,
        include_concept_drift: bool = False,
    ) -> str:
        """
        Bestimmt die primäre experimentelle Phase.

        Der Concept Drift wird für den Live-Demostream gezielt aktiviert.
        Der schnelle Benchmark-Stream bleibt standardmäßig bei seiner
        bisherigen Phasenlogik, damit bestehende Benchmark-Pfade nicht
        unbeabsichtigt verändert werden.
        """

        if (
            self.data_drift_start
            <= step
            < self.data_drift_end
        ):
            return SCENARIO_DATA_DRIFT

        if (
            include_concept_drift
            and self.concept_drift_start
            <= step
            < self.concept_drift_end
        ):
            return SCENARIO_CONCEPT_DRIFT

        if (
            self.infrastructure_start
            <= step
            < self.infrastructure_end
        ):
            return SCENARIO_INFRASTRUCTURE_FAILURE

        return SCENARIO_NORMAL

    # =========================================================================
    # Data Drift injizieren
    # =========================================================================

    def _inject_concept_drift(
        self,
        row: Dict,
        progress: float,
    ) -> Dict:
        """
        Injiziert einen kontrollierten multivariaten Data Drift.

        WICHTIG:

        Diese Methode verändert ausschließlich X und NICHT Y.

        Der Drift wird schrittweise aufgebaut. Gleichzeitig wird die
        Verschiebung pro Feature begrenzt. Dadurch wird verhindert, dass
        das reine Data-Drift-Szenario durch eine starke Verschiebung der
        Merkmale künstlich in ein Concept-Drift-Szenario übergeht.

        Die Methode heißt aus Kompatibilitätsgründen weiterhin
        ``_inject_concept_drift``. Semantisch handelt es sich hier jedoch
        ausschließlich um Data Drift.
        """

        progress = float(
            np.clip(
                progress,
                0.0,
                1.0,
            )
        )

        # Die konfigurierte maximale Verschiebung wird für das reine
        # Data-Drift-Szenario bewusst begrenzt.
        #
        # DATA_DRIFT_MAX_SHIFT kann beispielsweise 0.10 betragen.
        # Für die Laufzeitvalidierung wird jedoch nur ein kontrollierter
        # Anteil davon verwendet. Dadurch bleibt die Verschiebung
        # ausreichend sichtbar, ohne den gelernten Zusammenhang X -> Y
        # unnötig stark zu zerstören.
        effective_shift = min(
            float(DATA_DRIFT_MAX_SHIFT),
            0.03,
        )

        shift_strength = (
            effective_shift
            * progress
        )

        for feature in FEATURE_COLUMNS:

            value = float(
                row[feature]
            )

            if not np.isfinite(value):
                continue

            # Kontrollierte Verschiebung in Richtung höherer Werte.
            #
            # Die Transformation bleibt innerhalb des normierten Bereichs
            # [0, 1] und erzeugt dadurch einen echten Unterschied in P(X).
            shifted_value = (
                value
                + (
                    (1.0 - value)
                    * shift_strength
                )
            )

            row[feature] = float(
                np.clip(
                    shifted_value,
                    0.0,
                    1.0,
                )
            )

        return row

    # =========================================================================
    # Infrastruktur: Extremwert
    # =========================================================================

    def _inject_extreme_value(
        self,
        row: Dict,
    ) -> str:
        """Erzeugt einen Infrastruktur-Extremwert."""

        feature_index = int(
            self.rng.randint(
                0,
                len(FEATURE_COLUMNS),
            )
        )

        feature = FEATURE_COLUMNS[
            feature_index
        ]

        value = float(
            row[feature]
        )

        if np.isfinite(value):
            row[feature] = value * 5.0

        return feature

    # =========================================================================
    # Infrastruktur: Missing Value
    # =========================================================================

    def _inject_missing_value(
        self,
        row: Dict,
    ) -> str:
        """Erzeugt einen Missing-Value-Fehler auf einem Eingangsfeature."""

        feature_index = int(
            self.rng.randint(
                0,
                len(FEATURE_COLUMNS),
            )
        )

        feature = FEATURE_COLUMNS[
            feature_index
        ]

        row[feature] = np.nan

        return feature

    # =========================================================================
    # Concept Drift auf Y
    # =========================================================================

    @staticmethod
    def apply_target_concept_drift(
        observation: StreamObservation,
        strength: float = 1.0,
    ) -> StreamObservation:
        """
        Erzeugt kontrolliert einen Concept Drift in P(Y|X).

        Die Eingangsfeatures X bleiben unverändert.

        Nur das beobachtete Target Y wird verändert. Bei voller Stärke
        wird die binäre Beziehung zwischen UP und DOWN invertiert.

        Diese Methode wird ausschließlich für das gezielte
        Concept-Drift-Experiment verwendet.
        """

        strength = float(
            np.clip(
                strength,
                0.0,
                1.0,
            )
        )

        if strength >= 0.5:

            # Die vorbereiteten Elec2-Labels werden als Strings behandelt.
            original = str(
                observation.target_original
            ).strip()

            if original.upper() == "UP":
                observation.target = "DOWN"

            elif original.upper() == "DOWN":
                observation.target = "UP"

            else:
                raise ValueError(
                    "Der kontrollierte Concept Drift benötigt aktuell "
                    "die binären Elec2-Zielklassen UP/DOWN. "
                    f"Gefunden wurde: {original!r}"
                )

            observation.concept_drift_target_changed = True

            # Die Ground Truth wird nur für das explizite Concept-Drift-
            # Experiment gesetzt.
            observation.system_state = (
                SYSTEM_STATE_ANOMALY
            )

            observation.anomaly_type = (
                ANOMALY_CONCEPT_DRIFT
            )

            observation.root_cause = (
                ROOT_CAUSE_CONCEPT_DRIFT
            )

            observation.trigger_feature = (
                "TARGET_RELATIONSHIP"
            )

        return observation

    # =========================================================================
    # Infrastrukturmetriken
    # =========================================================================

    def _infrastructure_metrics(
        self,
        progress: float,
        deterministic: bool = False,
    ):
        """Erzeugt reproduzierbare Infrastrukturbelastung."""

        progress = float(
            np.clip(
                progress,
                0.0,
                1.0,
            )
        )

        if deterministic:
            # Die Ressourcenwerte werden in kontrollierten Teilabschnitten
            # erzeugt, damit jeder Ressourcentrigger eindeutig sichtbar wird.
            if progress < 0.20:
                cpu = max(
                    92.0,
                    INFRA_CPU_START_PERCENT,
                )
                memory = INFRA_MEMORY_START_PERCENT
                restarts = 0

            elif progress < 0.40:
                cpu = 70.0
                memory = max(
                    92.0,
                    INFRA_MEMORY_START_PERCENT,
                )
                restarts = 0

            elif progress < 0.60:
                cpu = 70.0
                memory = 70.0
                restarts = 1

            else:
                cpu = 70.0
                memory = 70.0
                restarts = 0

        else:
            # Im schnellen Benchmark bleibt die bisherige kontinuierliche
            # Belastungskurve unverändert.
            cpu = (
                INFRA_CPU_START_PERCENT
                + (
                    INFRA_CPU_END_PERCENT
                    - INFRA_CPU_START_PERCENT
                )
                * progress
            )

            memory = (
                INFRA_MEMORY_START_PERCENT
                + (
                    INFRA_MEMORY_END_PERCENT
                    - INFRA_MEMORY_START_PERCENT
                )
                * progress
            )

            restarts = int(
                progress >= INFRA_RESTART_PROGRESS
            )

        return (
            float(cpu),
            float(memory),
            restarts,
        )

    def _deterministic_infrastructure_trigger(
        self,
        progress: float,
    ) -> str:
        """
        Liefert für die Live-Demonstration einen reproduzierbaren
        Infrastruktur-Trigger für den aktuellen Teilabschnitt.

        Dadurch werden HIGH_CPU, HIGH_MEMORY, POD_RESTART, MISSING_VALUE und
        EXTREME_VALUE unabhängig von einer zufälligen Trefferwahrscheinlichkeit
        sichtbar. Die Methode wird nur verwendet, wenn die entsprechende
        Demo-Option in config aktiviert ist.
        """

        progress = float(
            np.clip(
                progress,
                0.0,
                1.0,
            )
        )

        if progress < 0.20:
            return "HIGH_CPU"

        if progress < 0.40:
            return "HIGH_MEMORY"

        if progress < 0.60:
            return "POD_RESTART"

        if progress < 0.80:
            return "MISSING_VALUE"

        return "EXTREME_VALUE"

    # =========================================================================
    # Beobachtung erzeugen
    # =========================================================================

    def _create_observation(
        self,
        step: int,
        cycle: int = 1,
        cycle_step: Optional[int] = None,
        include_concept_drift: bool = False,
        live_demo: bool = False,
    ) -> StreamObservation:
        """
        Erzeugt eine Beobachtung mit der Ground-Truth-Semantik
        für die Evaluation.
        """

        source = self.data.iloc[
            step
        ]

        row = source.to_dict()

        original_values = {
            feature: float(
                row[feature]
            )
            for feature in FEATURE_COLUMNS
        }

        # Die Zielvariable wird nicht numerisch umgewandelt.
        target_original = row[
            TARGET_COLUMN
        ]

        target = target_original

        concept_drift_target_changed = False

        scenario = self._scenario(
            step,
            include_concept_drift=include_concept_drift,
        )

        # ---------------------------------------------------------------------
        # Initialer operativer Zustand
        # ---------------------------------------------------------------------

        system_state = (
            SYSTEM_STATE_NORMAL
        )

        anomaly_type = (
            ANOMALY_NONE
        )

        root_cause = (
            ROOT_CAUSE_NORMAL
        )

        trigger_feature = ""

        phase_progress = 0.0

        cpu = 0.0
        memory = 0.0
        restarts = 0

        missing = False
        extreme = False
        schema_error = False

        # =====================================================================
        # Data Drift
        # =====================================================================

        if scenario == SCENARIO_DATA_DRIFT:

            # Ein reiner Data Drift ist KEINE Systemanomalie.
            system_state = (
                SYSTEM_STATE_NORMAL
            )

            anomaly_type = (
                ANOMALY_NONE
            )

            root_cause = (
                ROOT_CAUSE_NORMAL
            )

            trigger_feature = ""

            phase_progress = self._phase_progress(
                step,
                self.data_drift_start,
                self.data_drift_end,
            )

            # Nur die Eingangsverteilung P(X) wird verändert.
            #
            # Das Target Y bleibt vollständig unverändert.
            row = self._inject_concept_drift(
                row,
                phase_progress,
            )

        # =====================================================================
        # Concept Drift
        # =====================================================================

        elif scenario == SCENARIO_CONCEPT_DRIFT:

            # Concept Drift verändert ausschließlich die Zielvariable Y.
            # Die Eingangsfeatures X bleiben unverändert. Dadurch wird die
            # Beziehung P(Y|X) kontrolliert verändert.
            system_state = (
                SYSTEM_STATE_ANOMALY
            )

            anomaly_type = (
                ANOMALY_CONCEPT_DRIFT
            )

            root_cause = (
                ROOT_CAUSE_CONCEPT_DRIFT
            )

            trigger_feature = (
                "PREDICTION_ERROR"
            )

            phase_progress = self._phase_progress(
                step,
                self.concept_drift_start,
                self.concept_drift_end,
            )

            # Die vollständige Inversion der binären Elec2-Zielklasse erzeugt
            # eine eindeutige und reproduzierbare Änderung von P(Y|X).
            original_target = str(
                target_original
            ).strip()

            if original_target.upper() == "UP":
                target = "DOWN"
            elif original_target.upper() == "DOWN":
                target = "UP"
            else:
                raise ValueError(
                    "Der kontrollierte Concept Drift benötigt aktuell "
                    "die binären Elec2-Zielklassen UP/DOWN. "
                    f"Gefunden wurde: {original_target!r}"
                )

            concept_drift_target_changed = True

        # =====================================================================
        # Infrastructure Failure
        # =====================================================================

        elif scenario == SCENARIO_INFRASTRUCTURE_FAILURE:

            # Infrastrukturfehler sind eine eigenständige Anomaliefamilie
            # und setzen den Systemzustand auf ANOMALY.
            system_state = (
                SYSTEM_STATE_ANOMALY
            )

            anomaly_type = (
                ANOMALY_INFRASTRUCTURE
            )

            root_cause = (
                ROOT_CAUSE_INFRASTRUCTURE
            )

            phase_progress = self._phase_progress(
                step,
                self.infrastructure_start,
                self.infrastructure_end,
            )

            cpu, memory, restarts = (
                self._infrastructure_metrics(
                    phase_progress,
                    deterministic=live_demo,
                )
            )

            if (
                live_demo
                and getattr(config, "INFRA_DEMO_DETERMINISTIC", False)
            ):

                demo_trigger = self._deterministic_infrastructure_trigger(
                    phase_progress
                )

                if demo_trigger == "MISSING_VALUE":
                    missing = True
                    trigger_feature = "MISSING_VALUE"
                    self._inject_missing_value(row)

                elif demo_trigger == "EXTREME_VALUE":
                    extreme = True
                    trigger_feature = "EXTREME_VALUE"
                    self._inject_extreme_value(row)

                elif demo_trigger == "HIGH_CPU":
                    trigger_feature = "HIGH_CPU"

                elif demo_trigger == "HIGH_MEMORY":
                    trigger_feature = "HIGH_MEMORY"

                elif demo_trigger == "POD_RESTART":
                    trigger_feature = "POD_RESTART"

            elif self.missing_injection[step]:

                missing = True
                trigger_feature = "MISSING_VALUE"
                self._inject_missing_value(row)

            elif self.extreme_injection[step]:

                extreme = True
                trigger_feature = "EXTREME_VALUE"
                self._inject_extreme_value(row)

            elif cpu >= 90.0:

                trigger_feature = "HIGH_CPU"

            elif memory >= 90.0:

                trigger_feature = "HIGH_MEMORY"

            elif restarts >= 1:

                trigger_feature = "POD_RESTART"

            else:

                trigger_feature = "KUBERNETES_RESOURCES"

        # =====================================================================
        # Experimentelles Szenariolabel
        # =====================================================================

        if scenario == SCENARIO_DATA_DRIFT:

            classification = (
                CLASS_DATA_DRIFT
            )

        elif scenario == SCENARIO_CONCEPT_DRIFT:

            classification = (
                CLASS_CONCEPT_DRIFT
            )

        elif scenario == SCENARIO_INFRASTRUCTURE_FAILURE:

            classification = (
                CLASS_INFRASTRUCTURE
            )

        else:

            classification = (
                CLASS_NORMAL
            )

        local_step = (
            step
            if cycle_step is None
            else int(cycle_step)
        )

        return StreamObservation(

            step=int(step),

            cycle=int(cycle),

            cycle_step=local_step,

            timestamp=float(
                local_step
                * self.interval_seconds
            ),

            date=float(
                row["date"]
            ),

            day=int(
                row["day"]
            ),

            period=float(
                row["period"]
            ),

            nswprice=float(
                row["nswprice"]
            ),

            nswdemand=float(
                row["nswdemand"]
            ),

            vicprice=float(
                row["vicprice"]
            ),

            vicdemand=float(
                row["vicdemand"]
            ),

            transfer=float(
                row["transfer"]
            ),

            target=target,

            target_original=target_original,

            concept_drift_target_changed=bool(
                concept_drift_target_changed
            ),

            original_nswprice=original_values[
                "nswprice"
            ],

            original_nswdemand=original_values[
                "nswdemand"
            ],

            original_vicprice=original_values[
                "vicprice"
            ],

            original_vicdemand=original_values[
                "vicdemand"
            ],

            original_transfer=original_values[
                "transfer"
            ],

            scenario=scenario,

            classification=int(
                classification
            ),

            system_state=int(
                system_state
            ),

            anomaly_type=anomaly_type,

            root_cause=root_cause,

            trigger_feature=trigger_feature,

            phase_progress=float(
                phase_progress
            ),

            infrastructure_cpu_load=float(
                cpu
            ),

            infrastructure_memory_load=float(
                memory
            ),

            infrastructure_restarts=int(
                restarts
            ),

            missing_value_injected=missing,

            extreme_value_injected=extreme,

            schema_error_injected=schema_error,
        )

    # =========================================================================
    # Schneller Datenstrom
    # =========================================================================

    def stream_fast(
        self,
        max_rows=None,
        continuous=False,
    ):
        """Erzeugt den vollständigen Datenstrom ohne Wartezeit."""

        emitted = 0
        cycle = 1

        while True:

            for step in range(
                self.total_rows
            ):

                if (
                    max_rows is not None
                    and emitted >= int(
                        max_rows
                    )
                ):
                    return

                yield self._create_observation(
                    step,
                    cycle,
                )

                emitted += 1

            if not continuous:
                return

            cycle += 1

    # =========================================================================
    # Kontinuierlicher Stream
    # =========================================================================

    def stream_continuous(
        self,
        observations_per_phase=100,
    ):
        """Erzeugt einen kompakten kontinuierlichen Stream über alle Phasen."""

        phases = [

            (
                0,
                self.data_drift_start,
            ),

            (
                self.data_drift_start,
                self.data_drift_end,
            ),

            (
                self.concept_drift_start,
                self.concept_drift_end,
            ),

            (
                self.concept_drift_end,
                self.infrastructure_start,
            ),

            (
                self.infrastructure_start,
                self.infrastructure_end,
            ),

            (
                self.infrastructure_end,
                self.total_rows,
            ),
        ]

        cycle = 1

        while True:

            cycle_step = 0

            for start, end in phases:

                if end <= start:
                    continue

                count = min(
                    int(
                        observations_per_phase
                    ),
                    end - start,
                )

                indices = np.linspace(
                    start,
                    end - 1,
                    num=count,
                    dtype=int,
                )

                for step in indices:

                    yield self._create_observation(
                        int(step),
                        cycle,
                        cycle_step,
                        include_concept_drift=True,
                        live_demo=True,
                    )

                    cycle_step += 1

                    if self.interval_seconds > 0:

                        time.sleep(
                            min(
                                self.interval_seconds,
                                0.1,
                            )
                        )

            cycle += 1

    # =========================================================================
    # Phasenzusammenfassung
    # =========================================================================

    def get_phase_summary(
        self,
    ) -> pd.DataFrame:
        """Gibt die zeitlichen Experimentphasen zurück."""

        return pd.DataFrame(
            [
                {
                    "scenario": SCENARIO_NORMAL,
                    "start": 0,
                    "end": self.data_drift_start,
                    "classification": CLASS_NORMAL,
                    "system_state": "NORMAL",
                    "anomaly_type": ANOMALY_NONE,
                },

                {
                    "scenario": SCENARIO_DATA_DRIFT,
                    "start": self.data_drift_start,
                    "end": self.data_drift_end,
                    "classification": CLASS_DATA_DRIFT,
                    "system_state": "NORMAL",
                    "anomaly_type": ANOMALY_NONE,
                },

                {
                    "scenario": SCENARIO_CONCEPT_DRIFT,
                    "start": self.concept_drift_start,
                    "end": self.concept_drift_end,
                    "classification": CLASS_CONCEPT_DRIFT,
                    "system_state": "ANOMALY",
                    "anomaly_type": ANOMALY_CONCEPT_DRIFT,
                },

                {
                    "scenario": SCENARIO_NORMAL,
                    "start": self.concept_drift_end,
                    "end": self.infrastructure_start,
                    "classification": CLASS_NORMAL,
                    "system_state": "NORMAL",
                    "anomaly_type": ANOMALY_NONE,
                },

                {
                    "scenario": SCENARIO_INFRASTRUCTURE_FAILURE,
                    "start": self.infrastructure_start,
                    "end": self.infrastructure_end,
                    "classification": CLASS_INFRASTRUCTURE,
                    "system_state": "ANOMALY",
                    "anomaly_type": ANOMALY_INFRASTRUCTURE,
                },

                {
                    "scenario": SCENARIO_NORMAL,
                    "start": self.infrastructure_end,
                    "end": self.total_rows,
                    "classification": CLASS_NORMAL,
                    "system_state": "NORMAL",
                    "anomaly_type": ANOMALY_NONE,
                },
            ]
        )


# ============================================================================
# Factory
# ============================================================================

def create_ingestor() -> Elec2StreamIngestor:
    """Erzeugt den standardisierten Experiment-Ingestor."""

    return Elec2StreamIngestor(
        data_path=DATA_PATH,
        interval_seconds=STREAM_INTERVAL_SECONDS,
        random_state=RANDOM_STATE,
    )


# ============================================================================
# Direkter Aufruf
# ============================================================================

if __name__ == "__main__":

    ingestor = create_ingestor()

    print(
        ingestor.get_phase_summary().to_string(
            index=False
        )
    )

    for observation in ingestor.stream_fast(
        max_rows=5
    ):
        print(
            asdict(
                observation
            )
        )