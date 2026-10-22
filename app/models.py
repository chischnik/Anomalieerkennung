"""
ML-basierte Anomaliedetektoren für die Echtzeit-Erkennung.

Dieses Modul implementiert:

- Isolation Forest
- One-Class SVM

Beide Modelle werden ausschließlich mit normalen Referenzdaten
des Elec2-Datenstroms trainiert.

Wichtig für die Methodik:
Die Modelle arbeiten ausschließlich auf den fachlichen Elec2-Features.
Kubernetes-Metriken wie CPU, RAM oder Pod-Restarts werden nicht als
Eingaben für die beiden ML-Detektoren verwendet. Diese Metriken gehören
zur separaten Infrastruktur-Diagnose.

Die sklearn-Decision-Function wird als Modellscore beibehalten:

- höhere Werte: eher normal
- niedrigere Werte: eher anomal

Die finale Anomalieentscheidung erfolgt über einen auf normalen
Kalibrierungsdaten bestimmten Quantil-Schwellenwert.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from app.config import (
    ISOLATION_FOREST_CONTAMINATION,
    ISOLATION_FOREST_ESTIMATORS,
    ISOLATION_FOREST_RANDOM_STATE,
    ONE_CLASS_SVM_GAMMA,
    ONE_CLASS_SVM_KERNEL,
    ONE_CLASS_SVM_NU,
)


# ============================================================================
# Feature-Konfiguration
# ============================================================================

ELEC2_FEATURES = (
    "nswprice",
    "nswdemand",
    "vicprice",
    "vicdemand",
    "transfer",
)


# ============================================================================
# Gemeinsames Ergebnisformat
# ============================================================================

@dataclass
class DetectorResult:
    """
    Ergebnis eines einzelnen Anomaliedetektors.

    Attributes:
        score:
            Sklearn-Decision-Function.

            Höhere Werte entsprechen eher normalem Verhalten.

        anomaly_score:
            Negierter Decision-Function-Wert.

            Höhere Werte entsprechen stärker anomalem Verhalten.

        prediction:
            Binäre Entscheidung:

                1  = normal
               -1  = anomal

        is_anomaly:
            Boolesche Anomalieentscheidung.
    """

    score: float
    anomaly_score: float
    prediction: int
    is_anomaly: bool


# ============================================================================
# Hilfsfunktionen
# ============================================================================

def _prepare_input(
    X: np.ndarray,
    expected_features: Optional[int] = None,
) -> np.ndarray:
    """
    Konvertiert die Eingabe in ein zweidimensionales numerisches Array.

    Args:
        X:
            Eingabedaten.

        expected_features:
            Erwartete Anzahl an Features.

    Returns:
        Validiertes NumPy-Array.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.ndim != 2:
        raise ValueError(
            "Die Eingabedaten müssen ein zweidimensionales Array sein."
        )

    if X.shape[0] == 0:
        raise ValueError(
            "Die Eingabedaten dürfen nicht leer sein."
        )

    if expected_features is not None and X.shape[1] != expected_features:
        raise ValueError(
            "Die Eingabedaten enthalten "
            f"{X.shape[1]} Features; erwartet werden "
            f"{expected_features}."
        )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "Die Eingabedaten dürfen keine NaN- oder unendlichen "
            "Werte enthalten."
        )

    return X


def _validate_training_data(
    X: np.ndarray,
) -> None:
    """Prüft Größe und numerische Gültigkeit der Trainingsdaten."""
    if len(X) < 2:
        raise ValueError(
            "Für das Training werden mindestens zwei Beobachtungen benötigt."
        )

    if X.shape[1] < 1:
        raise ValueError(
            "Die Trainingsdaten müssen mindestens ein Feature enthalten."
        )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "Die Trainingsdaten enthalten ungültige Werte."
        )


# ============================================================================
# Isolation Forest
# ============================================================================

class IsolationForestDetector:
    """
    Isolation-Forest-basierter Anomaliedetektor.

    Der Detector wird ausschließlich auf normalen Referenzdaten trainiert.
    Der öffentliche anomaly_score besitzt die Semantik:

        höher = stärker anomal.

    Der sklearn Decision-Score bleibt zusätzlich verfügbar, da die
    Schwellenwertkalibrierung auf dieser Skala erfolgt.
    """

    def __init__(
        self,
        contamination: float = ISOLATION_FOREST_CONTAMINATION,
        n_estimators: int = ISOLATION_FOREST_ESTIMATORS,
        random_state: int = ISOLATION_FOREST_RANDOM_STATE,
    ) -> None:
        """
        Initialisiert den Isolation Forest.

        Args:
            contamination:
                Dokumentierter erwarteter Anomalieanteil. Da das Modell
                ausschließlich mit normalen Referenzdaten trainiert wird,
                wird die sklearn-interne Entscheidung nicht daraus abgeleitet.

            n_estimators:
                Anzahl der Bäume.

            random_state:
                Zufallszustand für reproduzierbare Experimente.
        """
        self.requested_contamination = contamination

        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination="auto",
            random_state=random_state,
            n_jobs=-1,
        )

        self.is_fitted = False

    @property
    def n_features_in_(self) -> Optional[int]:
        """Gibt die beim Training verwendete Feature-Anzahl zurück."""
        if not self.is_fitted:
            return None
        return int(self.model.n_features_in_)

    def fit(self, X: np.ndarray) -> None:
        """Trainiert den Isolation Forest mit normalen Referenzdaten."""
        X = _prepare_input(X)
        _validate_training_data(X)

        self.model.fit(X)
        self.is_fitted = True

    def predict(
        self,
        X: np.ndarray,
        threshold: Optional[float] = None,
    ) -> DetectorResult:
        """
        Bewertet eine Beobachtung.

        Bei einem expliziten Threshold gilt:

            Decision-Score < Threshold -> Anomalie.
        """
        self._check_fitted()

        X = _prepare_input(
            X,
            expected_features=self.n_features_in_,
        )

        score = float(self.model.decision_function(X)[0])
        anomaly_score = -score

        if threshold is None:
            prediction = int(self.model.predict(X)[0])
            is_anomaly = prediction == -1
        else:
            is_anomaly = score < float(threshold)
            prediction = -1 if is_anomaly else 1

        return DetectorResult(
            score=score,
            anomaly_score=anomaly_score,
            prediction=prediction,
            is_anomaly=is_anomaly,
        )

    def decision_function(self, X: np.ndarray) -> float:
        """Gibt den unveränderten sklearn Decision-Score zurück."""
        self._check_fitted()

        X = _prepare_input(
            X,
            expected_features=self.n_features_in_,
        )

        return float(self.model.decision_function(X)[0])

    def _check_fitted(self) -> None:
        """Prüft, ob der Detector trainiert wurde."""
        if not self.is_fitted:
            raise RuntimeError(
                "Der Isolation Forest wurde noch nicht trainiert."
            )


# ============================================================================
# One-Class SVM
# ============================================================================

class OneClassSVMDetector:
    """
    One-Class-SVM-basierter Anomaliedetektor.

    Die Features werden ausschließlich anhand der normalen Trainingsdaten
    standardisiert. Der bereits trainierte Scaler wird anschließend für
    alle Live-Beobachtungen wiederverwendet.

    Öffentliche Score-Semantik:

        anomaly_score höher = stärker anomal.
    """

    def __init__(
        self,
        kernel: str = ONE_CLASS_SVM_KERNEL,
        nu: float = ONE_CLASS_SVM_NU,
        gamma: str = ONE_CLASS_SVM_GAMMA,
    ) -> None:
        """Initialisiert Scaler und One-Class SVM."""
        self.scaler = StandardScaler()

        self.model = OneClassSVM(
            kernel=kernel,
            nu=nu,
            gamma=gamma,
        )

        self.is_fitted = False

    @property
    def n_features_in_(self) -> Optional[int]:
        """Gibt die beim Training verwendete Feature-Anzahl zurück."""
        if not self.is_fitted:
            return None
        return int(self.model.n_features_in_)

    def fit(self, X: np.ndarray) -> None:
        """Trainiert Scaler und One-Class SVM mit normalen Referenzdaten."""
        X = _prepare_input(X)
        _validate_training_data(X)

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)

        self.is_fitted = True

    def predict(
        self,
        X: np.ndarray,
        threshold: Optional[float] = None,
    ) -> DetectorResult:
        """Bewertet eine Beobachtung anhand der Decision-Function."""
        self._check_fitted()

        X = _prepare_input(
            X,
            expected_features=self.n_features_in_,
        )

        X_scaled = self.scaler.transform(X)
        score = float(self.model.decision_function(X_scaled)[0])
        anomaly_score = -score

        if threshold is None:
            prediction = int(self.model.predict(X_scaled)[0])
            is_anomaly = prediction == -1
        else:
            is_anomaly = score < float(threshold)
            prediction = -1 if is_anomaly else 1

        return DetectorResult(
            score=score,
            anomaly_score=anomaly_score,
            prediction=prediction,
            is_anomaly=is_anomaly,
        )

    def decision_function(self, X: np.ndarray) -> float:
        """Gibt den unveränderten sklearn Decision-Score zurück."""
        self._check_fitted()

        X = _prepare_input(
            X,
            expected_features=self.n_features_in_,
        )

        X_scaled = self.scaler.transform(X)

        return float(
            self.model.decision_function(X_scaled)[0]
        )

    def _check_fitted(self) -> None:
        """Prüft, ob die One-Class SVM trainiert wurde."""
        if not self.is_fitted:
            raise RuntimeError(
                "Die One-Class SVM wurde noch nicht trainiert."
            )


# ============================================================================
# Überwachtes Prognosemodell für die Concept-Drift-Erkennung
# ============================================================================


@dataclass
class TargetPredictionResult:
    """
    Ergebnis des überwachten Zielmodells.

    Die Zielvorhersage wird ausschließlich zur Messung der Modellleistung
    verwendet. Das Target ist kein Eingabefeature für Isolation Forest oder
    One-Class SVM.
    """

    prediction: int
    probability: float


class TargetPredictor:
    """
    Überwachtes Referenzmodell für die Elec2-Zielvariable.

    Das Modell wird nur mit normalen Referenzdaten trainiert und während der
    Auswertung nicht nachtrainiert. Dadurch kann eine Änderung von P(Y|X)
    als Leistungsabfall des eingefrorenen Referenzmodells sichtbar werden.
    """

    def __init__(self) -> None:
        """Initialisiert den Random-Forest-Klassifikator."""
        self.model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Trainiert das Zielmodell ausschließlich auf normalen Daten."""
        X = _prepare_input(X)
        y = np.asarray(y).reshape(-1)

        if len(X) != len(y):
            raise ValueError(
                "X und y müssen dieselbe Anzahl an Beobachtungen besitzen."
            )

        if len(np.unique(y)) < 2:
            raise ValueError(
                "Das Zielmodell benötigt mindestens zwei Zielklassen."
            )

        # Die Zielvariable darf semantische Klassen wie UP/DOWN enthalten.
        # Nur für das interne Random-Forest-Modell werden diese Labels codiert.
        if pd.isna(y).any():
            raise ValueError(
                "Die Zielwerte dürfen keine fehlenden Werte enthalten."
            )

        y_encoded = self.label_encoder.fit_transform(y.astype(str))
        self.model.fit(X, y_encoded)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> TargetPredictionResult:
        """Erzeugt eine einzelne Zielvorhersage."""
        if not self.is_fitted:
            raise RuntimeError(
                "Das Zielmodell wurde noch nicht trainiert."
            )

        X = _prepare_input(X)
        encoded_prediction = int(self.model.predict(X)[0])
        prediction = self.label_encoder.inverse_transform(
            [encoded_prediction]
        )[0]

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)[0]
            probability = float(np.max(probabilities))
        else:
            probability = 1.0

        return TargetPredictionResult(
            prediction=prediction,
            probability=probability,
        )


# ============================================================================
# Gemeinsame Modellverwaltung
# ============================================================================

class AnomalyModels:
    """
    Gemeinsame Verwaltung von Isolation Forest und One-Class SVM.

    Beide Modelle erhalten exakt denselben Feature-Vektor und werden
    ausschließlich auf einem normalen Referenzblock trainiert.

    Die Schwellenwerte werden anschließend auf einem separaten,
    ebenfalls normalen Kalibrierungsblock bestimmt. Dadurch werden
    Trainings- und Kalibrierungsdaten methodisch getrennt.
    """

    def __init__(self) -> None:
        """Initialisiert beide Detektoren und den Kalibrierungszustand."""
        self.isolation_forest = IsolationForestDetector()
        self.one_class_svm = OneClassSVMDetector()
        self.target_predictor = TargetPredictor()

        self.is_fitted = False
        self.is_calibrated = False

        self.isolation_forest_threshold: Optional[float] = None
        self.one_class_svm_threshold: Optional[float] = None
        self.calibration_quantile: Optional[float] = None

        self.feature_names = ELEC2_FEATURES

    @property
    def n_features_in_(self) -> Optional[int]:
        """Gibt die erwartete Anzahl an Features zurück."""
        if not self.is_fitted:
            return None
        return len(self.feature_names)

    def fit(self, X: np.ndarray) -> None:
        """
        Trainiert beide Detektoren mit derselben normalen Referenz.

        Args:
            X:
                Normaler Referenzblock.
        """
        X = _prepare_input(
            X,
            expected_features=len(self.feature_names),
        )
        _validate_training_data(X)

        self.isolation_forest.fit(X)
        self.one_class_svm.fit(X)

        self.is_fitted = True
        self.is_calibrated = False

        self.isolation_forest_threshold = None
        self.one_class_svm_threshold = None
        self.calibration_quantile = None

    def calibrate(
        self,
        X_calibration: np.ndarray,
        quantile: float = 0.01,
    ) -> None:
        """
        Kalibriert die Schwellenwerte mit normalen Daten.

        Die Quantilregel lautet:

            Threshold = q-Quantil der normalen Decision-Scores.

        Für neue Beobachtungen gilt:

            score < Threshold -> Anomalie.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Die Modelle müssen vor der Kalibrierung trainiert werden."
            )

        X_calibration = _prepare_input(
            X_calibration,
            expected_features=len(self.feature_names),
        )

        _validate_training_data(X_calibration)

        if not 0.0 < quantile < 1.0:
            raise ValueError(
                "Das Kalibrierungsquantil muss zwischen 0 und 1 liegen."
            )

        isolation_scores = self.isolation_forest.model.decision_function(
            X_calibration
        )

        X_calibration_scaled = self.one_class_svm.scaler.transform(
            X_calibration
        )

        svm_scores = self.one_class_svm.model.decision_function(
            X_calibration_scaled
        )

        self.isolation_forest_threshold = float(
            np.quantile(
                isolation_scores,
                quantile,
            )
        )

        self.one_class_svm_threshold = float(
            np.quantile(
                svm_scores,
                quantile,
            )
        )

        self.calibration_quantile = float(quantile)
        self.is_calibrated = True

    def fit_target_predictor(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Trainiert das eingefrorene Zielmodell auf normalen Referenzdaten.

        Dieses Modell dient ausschließlich der Leistungsüberwachung und
        ersetzt weder Isolation Forest noch One-Class SVM.
        """
        self.target_predictor.fit(X, y)

    def predict_target(self, X: np.ndarray) -> TargetPredictionResult:
        """Erzeugt eine Zielvorhersage mit dem eingefrorenen Referenzmodell."""
        return self.target_predictor.predict(X)

    def predict(self, X: np.ndarray) -> Dict[str, DetectorResult]:
        """
        Führt die kalibrierte Live-Inferenz beider Detektoren durch.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Die Modelle wurden noch nicht trainiert."
            )

        if not self.is_calibrated:
            raise RuntimeError(
                "Die Modelle wurden noch nicht kalibriert."
            )

        X = _prepare_input(
            X,
            expected_features=len(self.feature_names),
        )

        isolation_result = self.isolation_forest.predict(
            X,
            threshold=self.isolation_forest_threshold,
        )

        svm_result = self.one_class_svm.predict(
            X,
            threshold=self.one_class_svm_threshold,
        )

        return {
            "isolation_forest": isolation_result,
            "one_class_svm": svm_result,
        }

    def predict_single(
        self,
        values: Sequence[float],
    ) -> Dict[str, DetectorResult]:
        """
        Führt die Live-Inferenz für eine einzelne Elec2-Beobachtung aus.
        """
        X = create_feature_vector(*values)
        return self.predict(X)


# ============================================================================
# Elec2 Feature Vector
# ============================================================================

def create_feature_vector(
    nswprice: float,
    nswdemand: float,
    vicprice: float,
    vicdemand: float,
    transfer: float,
) -> np.ndarray:
    """
    Erstellt den standardisierten Elec2-Feature-Vektor.

    Die Reihenfolge ist verbindlich:

        1. nswprice
        2. nswdemand
        3. vicprice
        4. vicdemand
        5. transfer
    """
    values = np.asarray(
        [
            nswprice,
            nswdemand,
            vicprice,
            vicdemand,
            transfer,
        ],
        dtype=float,
    )

    if values.shape != (5,):
        raise ValueError(
            "Der Elec2-Feature-Vektor muss exakt fünf Werte enthalten."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Der Feature-Vektor enthält ungültige Werte."
        )

    return values.reshape(1, 5)
