"""
Sliding Window für multivariates Elec2-Driftmonitoring.

Das Modul unterscheidet:
- historische Abweichung gegenüber einer stabilen Referenz;
- lokale Verteilungsänderung zwischen zwei aufeinanderfolgenden Fenstern;
- punktuelle Peaks werden nicht als Concept Drift klassifiziert.

PSI und KS werden featureweise für:
    nswprice, nswdemand, vicprice, vicdemand, transfer
berechnet.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Mapping, Optional

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import ks_2samp

from app.config import (
    FEATURE_COLUMNS,
    KS_ALPHA,
    MIN_WINDOW_SIZE,
    PEAK_SIGMA_FACTOR,
    PSI_THRESHOLD,
    PRICE_COLUMN,
    WINDOW_SIZE,
)


@dataclass
class WindowObservation:
    """Repräsentiert eine Beobachtung im Sliding Window."""

    step: int
    timestamp: float
    values: Dict[str, float]


class SlidingWindow:
    """
    Verwaltet das aktuelle Fenster und ein zusätzliches Vergleichsfenster.

    Das aktuelle Fenster enthält W Beobachtungen. Intern werden bis zu 2W
    Beobachtungen gehalten, damit zwei nicht überlappende lokale Fenster
    verglichen werden können.
    """

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        min_window_size: int = MIN_WINDOW_SIZE,
        reference_values: Optional[
            Mapping[str, np.ndarray]
        ] = None,
    ) -> None:
        if window_size <= 0:
            raise ValueError(
                "Die Fenstergröße muss größer als 0 sein."
            )

        if min_window_size <= 0:
            raise ValueError(
                "Die minimale Fenstergröße muss größer als 0 sein."
            )

        if min_window_size > window_size:
            raise ValueError(
                "Die minimale Fenstergröße darf nicht größer "
                "als die Fenstergröße sein."
            )

        self.window_size = int(window_size)
        self.min_window_size = int(min_window_size)

        self._observations: Deque[
            WindowObservation
        ] = deque(
            maxlen=self.window_size
        )

        self._comparison_history: Deque[
            WindowObservation
        ] = deque(
            maxlen=2 * self.window_size
        )

        self._reference_values: Dict[
            str,
            np.ndarray,
        ] = {}

        if reference_values is not None:
            self.set_reference(reference_values)

    # ========================================================================
    # Referenz
    # ========================================================================

    def set_reference(
        self,
        reference_values: Mapping[str, np.ndarray],
    ) -> None:
        """Setzt stabile Referenzverteilungen für alle verfügbaren Features."""

        if not isinstance(reference_values, Mapping):
            raise TypeError(
                "reference_values muss ein Mapping sein."
            )

        cleaned: Dict[str, np.ndarray] = {}

        for feature, values in reference_values.items():
            array = np.asarray(
                values,
                dtype=float,
            )
            array = array[np.isfinite(array)]

            if len(array) >= 2:
                cleaned[str(feature)] = array.copy()

        if not cleaned:
            raise ValueError(
                "Es konnte keine gültige Referenz gesetzt werden."
            )

        self._reference_values = cleaned

    def set_single_reference(
        self,
        feature: str,
        values: np.ndarray,
    ) -> None:
        """Setzt die Referenz für ein einzelnes Feature."""

        array = np.asarray(
            values,
            dtype=float,
        )
        array = array[np.isfinite(array)]

        if len(array) < 2:
            raise ValueError(
                "Eine Referenz benötigt mindestens zwei gültige Werte."
            )

        self._reference_values[
            str(feature)
        ] = array.copy()

    @property
    def has_reference(self) -> bool:
        """Gibt an, ob eine Referenz vorhanden ist."""

        return bool(self._reference_values)

    @property
    def reference_size(self) -> int:
        """Gibt die Größe der ersten Referenz zurück."""

        if not self._reference_values:
            return 0

        return len(
            next(
                iter(
                    self._reference_values.values()
                )
            )
        )

    # ========================================================================
    # Beobachtungen
    # ========================================================================

    def add_observation(
        self,
        values: Mapping[str, float],
        step: int,
        timestamp: float,
    ) -> None:
        """Fügt eine multivariate Beobachtung hinzu."""

        normalized: Dict[str, float] = {}

        for feature in FEATURE_COLUMNS:
            value = values.get(
                feature,
                np.nan,
            )

            try:
                normalized[feature] = float(value)
            except (TypeError, ValueError):
                normalized[feature] = np.nan

        observation = WindowObservation(
            step=int(step),
            timestamp=float(timestamp),
            values=normalized,
        )

        self._observations.append(observation)
        self._comparison_history.append(observation)

    def add(
        self,
        value: float,
        step: int,
        timestamp: float,
    ) -> None:
        """Kompatibilitätsmethode für nswprice."""

        self.add_observation(
            values={
                PRICE_COLUMN: value,
            },
            step=step,
            timestamp=timestamp,
        )

    @property
    def size(self) -> int:
        """Anzahl der Beobachtungen im aktuellen Fenster."""

        return len(self._observations)

    @property
    def comparison_size(self) -> int:
        """Anzahl der gespeicherten Vergleichsbeobachtungen."""

        return len(self._comparison_history)

    @property
    def is_ready(self) -> bool:
        """Prüft die Mindestgröße des aktuellen Fensters."""

        return self.size >= self.min_window_size

    @property
    def is_local_comparison_ready(self) -> bool:
        """Prüft, ob zwei vollständige Fenster vorhanden sind."""

        return len(
            self._comparison_history
        ) >= 2 * self.window_size

    @property
    def values(self) -> np.ndarray:
        """Gültige Werte von nswprice."""

        return self.feature_values(
            PRICE_COLUMN
        )

    @property
    def timestamps(self) -> np.ndarray:
        """Zeitstempel des aktuellen Fensters."""

        return np.asarray(
            [
                observation.timestamp
                for observation in self._observations
            ],
            dtype=float,
        )

    @property
    def observations(self) -> list:
        """Aktuelle Beobachtungen."""

        return list(
            self._observations
        )

    def feature_values(
        self,
        feature: str,
    ) -> np.ndarray:
        """Gibt gültige Werte eines Features im aktuellen Fenster zurück."""

        values = [
            observation.values.get(
                feature,
                np.nan,
            )
            for observation in self._observations
        ]

        array = np.asarray(
            values,
            dtype=float,
        )

        return array[
            np.isfinite(array)
        ]

    def latest_value(
        self,
        feature: str = PRICE_COLUMN,
    ) -> Optional[float]:
        """Gibt den letzten gültigen Featurewert zurück."""

        for observation in reversed(
            self._observations
        ):
            value = observation.values.get(
                feature,
                np.nan,
            )

            if np.isfinite(value):
                return float(value)

        return None

    def previous_window_values(
        self,
        feature: str,
    ) -> np.ndarray:
        """Gibt das unmittelbar vorhergehende, nicht überlappende Fenster zurück."""

        history = list(
            self._comparison_history
        )

        if len(history) < 2 * self.window_size:
            return np.array(
                [],
                dtype=float,
            )

        previous = history[
            -2 * self.window_size:
            -self.window_size
        ]

        values = [
            observation.values.get(
                feature,
                np.nan,
            )
            for observation in previous
        ]

        array = np.asarray(
            values,
            dtype=float,
        )

        return array[
            np.isfinite(array)
        ]

    def current_window_values(
        self,
        feature: str,
    ) -> np.ndarray:
        """Gibt das aktuelle lokale Vergleichsfenster zurück."""

        if not self.is_local_comparison_ready:
            return np.array(
                [],
                dtype=float,
            )

        return self.feature_values(
            feature
        )

    # ========================================================================
    # Rolling-Metriken
    # ========================================================================

    def rolling_mean(
        self,
        feature: str = PRICE_COLUMN,
    ) -> Optional[float]:
        """Berechnet den gleitenden Mittelwert."""

        if not self.is_ready:
            return None

        values = self.feature_values(
            feature
        )

        if len(values) == 0:
            return None

        return float(
            np.mean(values)
        )

    def rolling_std(
        self,
        feature: str = PRICE_COLUMN,
    ) -> Optional[float]:
        """Berechnet die gleitende Standardabweichung."""

        if not self.is_ready:
            return None

        values = self.feature_values(
            feature
        )

        if len(values) == 0:
            return None

        return float(
            np.std(
                values,
                ddof=0,
            )
        )

    def rate_of_change(
        self,
        feature: str = PRICE_COLUMN,
    ) -> Optional[float]:
        """Berechnet die Änderungsrate der letzten zwei gültigen Werte."""

        if len(self._observations) < 2:
            return None

        valid = []

        for observation in reversed(
            self._observations
        ):
            value = observation.values.get(
                feature,
                np.nan,
            )

            if np.isfinite(value):
                valid.append(
                    (
                        observation.timestamp,
                        float(value),
                    )
                )

            if len(valid) == 2:
                break

        if len(valid) < 2:
            return None

        current_time, current_value = valid[0]
        previous_time, previous_value = valid[1]

        delta_t = current_time - previous_time

        if delta_t <= 0:
            return None

        return float(
            (current_value - previous_value)
            / delta_t
        )

    def peak_density(
        self,
        feature: str = PRICE_COLUMN,
    ) -> Optional[int]:
        """Zählt Peaks oberhalb von μ + kσ."""

        if not self.is_ready:
            return None

        values = self.feature_values(
            feature
        )

        if len(values) < 3:
            return 0

        mean = float(
            np.mean(values)
        )
        std = float(
            np.std(
                values,
                ddof=0,
            )
        )

        if std <= 1e-12:
            return 0

        threshold = (
            mean
            + PEAK_SIGMA_FACTOR * std
        )

        peaks, _ = find_peaks(
            values
        )

        if len(peaks) == 0:
            return 0

        return int(
            np.sum(
                values[peaks] > threshold
            )
        )

    def has_double_peak(
        self,
        feature: str = PRICE_COLUMN,
    ) -> bool:
        """Prüft auf mindestens zwei relevante Peaks."""

        density = self.peak_density(
            feature
        )

        return bool(
            density is not None
            and density >= 2
        )

    # ========================================================================
    # Statistische Tests gegen historische Referenz
    # ========================================================================

    def ks_test(
        self,
        feature: str = PRICE_COLUMN,
    ) -> Optional[Dict[str, float]]:
        """Vergleicht das aktuelle Fenster mit der historischen Referenz."""

        if not self.is_ready:
            return None

        reference = self._reference_values.get(
            feature
        )

        if reference is None:
            return None

        current = self.feature_values(
            feature
        )

        if len(current) < self.min_window_size:
            return None

        try:
            result = ks_2samp(
                reference,
                current,
                alternative="two-sided",
                method="asymp",
            )
        except (ValueError, FloatingPointError):
            return None

        return {
            "statistic": float(
                result.statistic
            ),
            "pvalue": float(
                result.pvalue
            ),
        }

    @staticmethod
    def _psi(
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10,
    ) -> Optional[float]:
        """Berechnet PSI anhand von Bins aus der Referenz."""

        reference = np.asarray(
            reference,
            dtype=float,
        )
        current = np.asarray(
            current,
            dtype=float,
        )

        reference = reference[
            np.isfinite(reference)
        ]
        current = current[
            np.isfinite(current)
        ]

        if len(reference) < 2 or len(current) < 2:
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

        if len(edges) < 2:
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

        epsilon = 1e-6

        ref_ratio = np.clip(
            ref_counts.astype(float)
            / len(reference),
            epsilon,
            None,
        )
        cur_ratio = np.clip(
            cur_counts.astype(float)
            / len(current),
            epsilon,
            None,
        )

        value = np.sum(
            (cur_ratio - ref_ratio)
            * np.log(
                cur_ratio / ref_ratio
            )
        )

        if not np.isfinite(value):
            return None

        return float(value)

    def psi(
        self,
        feature: str,
    ) -> Optional[float]:
        """Berechnet historischen PSI."""

        if not self.is_ready:
            return None

        reference = self._reference_values.get(
            feature
        )

        if reference is None:
            return None

        return self._psi(
            reference,
            self.feature_values(
                feature
            ),
        )

    def psi_by_feature(
        self,
    ) -> Dict[str, Optional[float]]:
        """Berechnet historischen PSI für alle Features."""

        return {
            feature: self.psi(feature)
            for feature in self._reference_values
        }

    # ========================================================================
    # Lokale Tests
    # ========================================================================

    def local_ks_test(
        self,
        feature: str,
    ) -> Optional[Dict[str, float]]:
        """Vergleicht vorheriges und aktuelles Fenster."""

        previous = self.previous_window_values(
            feature
        )
        current = self.current_window_values(
            feature
        )

        if (
            len(previous) < self.min_window_size
            or len(current) < self.min_window_size
        ):
            return None

        try:
            result = ks_2samp(
                previous,
                current,
                alternative="two-sided",
                method="asymp",
            )
        except (ValueError, FloatingPointError):
            return None

        return {
            "statistic": float(
                result.statistic
            ),
            "pvalue": float(
                result.pvalue
            ),
        }

    def local_psi(
        self,
        feature: str,
    ) -> Optional[float]:
        """Berechnet lokalen PSI mit dem vorherigen Fenster als Referenz."""

        previous = self.previous_window_values(
            feature
        )
        current = self.current_window_values(
            feature
        )

        if (
            len(previous) < self.min_window_size
            or len(current) < self.min_window_size
        ):
            return None

        return self._psi(
            previous,
            current,
        )

    def local_drift_by_feature(
        self,
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Berechnet lokale PSI-/KS-Werte und die gemeinsame Driftbedingung.

        Ein Feature gilt lokal als Driftkandidat, wenn beide Tests
        gleichzeitig anschlagen:

            PSI > PSI_THRESHOLD
            UND
            KS-p < KS_ALPHA
        """

        features = (
            list(self._reference_values.keys())
            if self._reference_values
            else list(FEATURE_COLUMNS)
        )

        result = {}

        for feature in features:
            psi_value = self.local_psi(
                feature
            )
            ks_result = self.local_ks_test(
                feature
            )

            ks_pvalue = (
                None
                if ks_result is None
                else ks_result["pvalue"]
            )

            psi_drift = (
                None
                if psi_value is None
                else bool(
                    psi_value > PSI_THRESHOLD
                )
            )

            ks_drift = (
                None
                if ks_pvalue is None
                else bool(
                    ks_pvalue < KS_ALPHA
                )
            )

            result[feature] = {
                "psi": psi_value,
                "ks_statistic": (
                    None
                    if ks_result is None
                    else ks_result["statistic"]
                ),
                "ks_pvalue": ks_pvalue,
                "psi_drift": psi_drift,
                "ks_drift": ks_drift,
                "drift_candidate": (
                    None
                    if psi_drift is None
                    or ks_drift is None
                    else bool(
                        psi_drift
                        and ks_drift
                    )
                ),
            }

        return result

    def local_drift_trigger_feature(
        self,
    ) -> Optional[str]:
        """
        Liefert das stärkste Feature unter den lokalen Driftkandidaten.

        Die Auswahl erfolgt nach dem höchsten PSI. Damit kann die
        Diagnoseebene ein konkretes Triggerfeature ausweisen.
        """

        metrics = self.local_drift_by_feature()

        candidates = {
            feature: values["psi"]
            for feature, values in metrics.items()
            if values["drift_candidate"]
            and values["psi"] is not None
            and np.isfinite(values["psi"])
        }

        if not candidates:
            return None

        return max(
            candidates,
            key=candidates.get,
        )

    def local_drift_detected(self) -> bool:
        """Gibt an, ob mindestens ein Feature beide Tests erfüllt."""

        metrics = self.local_drift_by_feature()

        return any(
            values["drift_candidate"] is True
            for values in metrics.values()
        )

    # ========================================================================
    # Historische Featurediagnose
    # ========================================================================

    def drift_by_feature(
        self,
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """Kombiniert historischen PSI und KS je Feature."""

        result = {}

        for feature in self._reference_values:
            psi_value = self.psi(
                feature
            )
            ks_result = self.ks_test(
                feature
            )

            ks_pvalue = (
                None
                if ks_result is None
                else ks_result["pvalue"]
            )

            result[feature] = {
                "psi": psi_value,
                "ks_statistic": (
                    None
                    if ks_result is None
                    else ks_result["statistic"]
                ),
                "ks_pvalue": ks_pvalue,
                "psi_drift": (
                    None
                    if psi_value is None
                    else bool(
                        psi_value > PSI_THRESHOLD
                    )
                ),
                "ks_drift": (
                    None
                    if ks_pvalue is None
                    else bool(
                        ks_pvalue < KS_ALPHA
                    )
                ),
            }

        return result

    def drift_trigger_feature(
        self,
    ) -> Optional[str]:
        """Liefert das Feature mit dem höchsten historischen PSI."""

        psi_values = self.psi_by_feature()

        valid = {
            feature: value
            for feature, value in psi_values.items()
            if value is not None
            and np.isfinite(value)
        }

        if not valid:
            return None

        return max(
            valid,
            key=valid.get,
        )

    # ========================================================================
    # Z-Score und Gesamtmetriken
    # ========================================================================

    def _previous_values(
        self,
        feature: str = PRICE_COLUMN,
    ) -> np.ndarray:
        """Gibt gültige Werte ohne die letzte Beobachtung zurück."""

        observations = list(
            self._observations
        )

        if len(observations) <= 1:
            return np.array(
                [],
                dtype=float,
            )

        values = [
            observation.values.get(
                feature,
                np.nan,
            )
            for observation in observations[:-1]
        ]

        array = np.asarray(
            values,
            dtype=float,
        )

        return array[
            np.isfinite(array)
        ]

    def z_score(
        self,
        feature: str = PRICE_COLUMN,
    ) -> Optional[float]:
        """Berechnet den Z-Score der letzten gültigen Beobachtung."""

        if not self.is_ready:
            return None

        previous = self._previous_values(
            feature
        )
        current = self.latest_value(
            feature
        )

        if (
            len(previous) < 2
            or current is None
        ):
            return None

        mean = float(
            np.mean(previous)
        )
        std = float(
            np.std(
                previous,
                ddof=0,
            )
        )

        if std <= 1e-12:
            return (
                0.0
                if abs(current - mean) <= 1e-12
                else float(
                    "inf"
                    if current > mean
                    else "-inf"
                )
            )

        return float(
            (current - mean) / std
        )

    def calculate_metrics(
        self,
        feature: str = PRICE_COLUMN,
    ) -> Dict[str, Optional[float]]:
        """Berechnet die Einzelmetriken eines Features."""

        ks_result = self.ks_test(
            feature
        )
        psi_value = self.psi(
            feature
        )

        return {
            "window_size": self.size,
            "rolling_mean": self.rolling_mean(
                feature
            ),
            "rolling_std": self.rolling_std(
                feature
            ),
            "rate_of_change": self.rate_of_change(
                feature
            ),
            "peak_density": self.peak_density(
                feature
            ),
            "double_peak": int(
                self.has_double_peak(
                    feature
                )
            ),
            "ks_pvalue": (
                None
                if ks_result is None
                else ks_result["pvalue"]
            ),
            "ks_statistic": (
                None
                if ks_result is None
                else ks_result["statistic"]
            ),
            "z_score": self.z_score(
                feature
            ),
            "psi": psi_value,
            "psi_threshold": PSI_THRESHOLD,
            "psi_drift": (
                None
                if psi_value is None
                else bool(
                    psi_value > PSI_THRESHOLD
                )
            ),
        }

    def calculate_all_drift_metrics(
        self,
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """Berechnet historische Driftmetriken."""

        return self.drift_by_feature()

    def calculate_all_local_drift_metrics(
        self,
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """Berechnet lokale Driftmetriken."""

        return self.local_drift_by_feature()

    # ========================================================================
    # Reset
    # ========================================================================

    def clear(self) -> None:
        """Leert Beobachtungen und lokalen Verlauf; Referenz bleibt erhalten."""

        self._observations.clear()
        self._comparison_history.clear()


def create_price_window(
    reference_values: Optional[np.ndarray] = None,
) -> SlidingWindow:
    """Erstellt ein kompatibles nswprice-Fenster."""

    reference = None

    if reference_values is not None:
        reference = {
            PRICE_COLUMN: np.asarray(
                reference_values,
                dtype=float,
            )
        }

    return SlidingWindow(
        window_size=WINDOW_SIZE,
        min_window_size=MIN_WINDOW_SIZE,
        reference_values=reference,
    )


def create_feature_window(
    reference_values: Optional[
        Mapping[str, np.ndarray]
    ] = None,
) -> SlidingWindow:
    """Erstellt ein multivariates Elec2-Fenster."""

    return SlidingWindow(
        window_size=WINDOW_SIZE,
        min_window_size=MIN_WINDOW_SIZE,
        reference_values=reference_values,
    )