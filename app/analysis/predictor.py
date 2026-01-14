"""ML-based prediction engine for TOTO numbers."""

import random
from datetime import datetime
from typing import Optional
import numpy as np
from collections import Counter

from sqlalchemy.orm import Session

from app.database.models import Draw
from app.analysis.statistics import NumberAnalyzer
from app.analysis.patterns import PatternAnalyzer
from app.config import (
    TOTO_MIN_NUMBER, TOTO_MAX_NUMBER, TOTO_NUMBERS_DRAWN,
    LSTM_SEQUENCE_LENGTH, LSTM_EPOCHS, LSTM_BATCH_SIZE, MODEL_PATH
)


class PredictionStrategy:
    """Base class for prediction strategies."""

    def __init__(self, session: Session, max_draw_number: Optional[int] = None):
        self.session = session
        self.max_draw_number = max_draw_number
        self.number_analyzer = NumberAnalyzer(session, max_draw_number)
        self.pattern_analyzer = PatternAnalyzer(session, max_draw_number)
        self.all_numbers = list(range(TOTO_MIN_NUMBER, TOTO_MAX_NUMBER + 1))

    def generate(self, count: int = 1) -> list[dict]:
        """Generate predictions. Override in subclass."""
        raise NotImplementedError


class HotNumbersStrategy(PredictionStrategy):
    """Predict based on frequently drawn numbers."""

    def generate(self, count: int = 1) -> list[dict]:
        predictions = []
        frequency = self.number_analyzer.calculate_frequency()

        # Sort numbers by frequency
        sorted_nums = sorted(
            frequency.items(),
            key=lambda x: x[1]["total_appearances"],
            reverse=True
        )

        # Get top 20 hot numbers for selection pool
        hot_pool = [num for num, _ in sorted_nums[:20]]

        for _ in range(count):
            # Select 6 numbers weighted by frequency
            weights = [frequency[n]["total_appearances"] for n in hot_pool]
            total_weight = sum(weights)
            if total_weight == 0:
                weights = [1] * len(hot_pool)
                total_weight = len(hot_pool)

            probs = [w / total_weight for w in weights]

            selected = np.random.choice(
                hot_pool,
                size=min(6, len(hot_pool)),
                replace=False,
                p=probs
            ).tolist()

            # If not enough, fill from remaining numbers
            while len(selected) < 6:
                remaining = [n for n in self.all_numbers if n not in selected]
                selected.append(random.choice(remaining))

            predictions.append({
                "numbers": sorted(selected),
                "strategy": "hot",
                "confidence": 0.6,
                "reasoning": "Selected from most frequently drawn numbers"
            })

        return predictions


class ColdNumbersStrategy(PredictionStrategy):
    """Predict based on overdue (cold) numbers."""

    def generate(self, count: int = 1) -> list[dict]:
        predictions = []
        frequency = self.number_analyzer.calculate_frequency()

        # Sort numbers by gap (draws since last appearance)
        sorted_nums = sorted(
            frequency.items(),
            key=lambda x: x[1]["draws_since_last"],
            reverse=True
        )

        # Get top 20 overdue numbers
        cold_pool = [num for num, _ in sorted_nums[:20]]

        for _ in range(count):
            # Weight by gap size
            weights = [frequency[n]["draws_since_last"] + 1 for n in cold_pool]
            total_weight = sum(weights)
            probs = [w / total_weight for w in weights]

            selected = np.random.choice(
                cold_pool,
                size=min(6, len(cold_pool)),
                replace=False,
                p=probs
            ).tolist()

            while len(selected) < 6:
                remaining = [n for n in self.all_numbers if n not in selected]
                selected.append(random.choice(remaining))

            predictions.append({
                "numbers": sorted(selected),
                "strategy": "cold",
                "confidence": 0.5,
                "reasoning": "Selected from numbers overdue for appearance"
            })

        return predictions


class BalancedStrategy(PredictionStrategy):
    """Generate balanced combinations based on statistical distributions."""

    def generate(self, count: int = 1) -> list[dict]:
        predictions = []

        # Get statistical targets
        sum_stats = self.number_analyzer.calculate_sum_statistics()
        target_sum = sum_stats.get("mean", 150) if sum_stats else 150

        for _ in range(count):
            selected = self._generate_balanced_set(target_sum)

            predictions.append({
                "numbers": sorted(selected),
                "strategy": "balanced",
                "confidence": 0.55,
                "reasoning": f"Balanced odd/even, high/low with target sum ~{int(target_sum)}"
            })

        return predictions

    def _generate_balanced_set(self, target_sum: float, max_attempts: int = 100) -> list[int]:
        """Generate a balanced set of numbers."""
        best_set = None
        best_score = float('inf')

        for _ in range(max_attempts):
            # Target: 3 odd, 3 even and 3 low (1-25), 3 high (26-49)
            low_numbers = list(range(1, 26))
            high_numbers = list(range(26, 50))

            selected = []

            # Pick 3 from low, 3 from high
            selected.extend(random.sample(low_numbers, 3))
            selected.extend(random.sample(high_numbers, 3))

            # Check odd/even balance
            odd_count = sum(1 for n in selected if n % 2 == 1)
            even_count = 6 - odd_count

            # Score based on how close to ideal
            sum_diff = abs(sum(selected) - target_sum)
            balance_diff = abs(odd_count - 3)

            score = sum_diff + balance_diff * 20

            if score < best_score:
                best_score = score
                best_set = selected

            # Good enough
            if sum_diff < 20 and balance_diff <= 1:
                break

        return best_set or random.sample(self.all_numbers, 6)


class LSTMStrategy(PredictionStrategy):
    """LSTM-based sequence prediction."""

    def __init__(self, session: Session, max_draw_number: Optional[int] = None):
        super().__init__(session, max_draw_number)
        self.model = None
        self.model_path = MODEL_PATH / "lstm_model.keras"

    def prepare_data(self) -> tuple:
        """Prepare training data for LSTM."""
        draws = self.number_analyzer.get_all_draws()
        draws = sorted(draws, key=lambda d: d.draw_number)

        if len(draws) < LSTM_SEQUENCE_LENGTH + 1:
            return None, None

        # Create sequences
        sequences = []
        targets = []

        for i in range(len(draws) - LSTM_SEQUENCE_LENGTH):
            # Sequence of past draws (one-hot encoded)
            seq = []
            for j in range(LSTM_SEQUENCE_LENGTH):
                draw = draws[i + j]
                one_hot = [0] * 49
                for num in draw.winning_numbers:
                    one_hot[num - 1] = 1
                seq.append(one_hot)
            sequences.append(seq)

            # Target: next draw
            target_draw = draws[i + LSTM_SEQUENCE_LENGTH]
            target = [0] * 49
            for num in target_draw.winning_numbers:
                target[num - 1] = 1
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def train(self, epochs: int = LSTM_EPOCHS) -> dict:
        """Train the LSTM model."""
        try:
            from tensorflow import keras
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
        except ImportError:
            return {"success": False, "error": "TensorFlow not installed"}

        X, y = self.prepare_data()
        if X is None:
            return {"success": False, "error": "Not enough data for training"}

        # Build model
        model = Sequential([
            LSTM(128, input_shape=(LSTM_SEQUENCE_LENGTH, 49), return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(49, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Train
        history = model.fit(
            X, y,
            epochs=epochs,
            batch_size=LSTM_BATCH_SIZE,
            validation_split=0.2,
            verbose=0
        )

        # Save model
        model.save(self.model_path)
        self.model = model

        return {
            "success": True,
            "epochs": epochs,
            "final_loss": float(history.history['loss'][-1]),
            "final_accuracy": float(history.history['accuracy'][-1]),
        }

    def load_model(self) -> bool:
        """Load saved model if exists."""
        try:
            from tensorflow import keras
            if self.model_path.exists():
                self.model = keras.models.load_model(self.model_path)
                return True
        except Exception:
            pass
        return False

    def generate(self, count: int = 1) -> list[dict]:
        predictions = []

        # Try to load or train model
        if self.model is None:
            if not self.load_model():
                # Train new model
                result = self.train()
                if not result.get("success"):
                    # Fallback to balanced strategy
                    fallback = BalancedStrategy(self.session, self.max_draw_number)
                    preds = fallback.generate(count)
                    for p in preds:
                        p["strategy"] = "ml"
                        p["reasoning"] = "Fallback: ML model unavailable"
                    return preds

        # Get recent draws for prediction
        draws = self.number_analyzer.get_all_draws(limit=LSTM_SEQUENCE_LENGTH)
        draws = sorted(draws, key=lambda d: d.draw_number)

        if len(draws) < LSTM_SEQUENCE_LENGTH:
            fallback = BalancedStrategy(self.session, self.max_draw_number)
            return fallback.generate(count)

        # Prepare input sequence
        seq = []
        for draw in draws[-LSTM_SEQUENCE_LENGTH:]:
            one_hot = [0] * 49
            for num in draw.winning_numbers:
                one_hot[num - 1] = 1
            seq.append(one_hot)

        X = np.array([seq])

        for _ in range(count):
            # Predict
            pred = self.model.predict(X, verbose=0)[0]

            # Select top 6 numbers by probability
            indices = np.argsort(pred)[-6:]
            selected = [i + 1 for i in indices]

            # Add some randomness based on probability
            probs = pred[indices]
            confidence = float(np.mean(probs))

            predictions.append({
                "numbers": sorted(selected),
                "strategy": "ml",
                "confidence": min(confidence, 0.7),
                "reasoning": "LSTM neural network prediction"
            })

        return predictions


class EnsembleStrategy(PredictionStrategy):
    """Combine multiple strategies for ensemble prediction."""

    def generate(self, count: int = 1) -> list[dict]:
        predictions = []

        # Get predictions from all strategies
        hot_strategy = HotNumbersStrategy(self.session, self.max_draw_number)
        cold_strategy = ColdNumbersStrategy(self.session, self.max_draw_number)
        balanced_strategy = BalancedStrategy(self.session, self.max_draw_number)

        for _ in range(count):
            # Get one prediction from each
            hot_pred = hot_strategy.generate(1)[0]["numbers"]
            cold_pred = cold_strategy.generate(1)[0]["numbers"]
            balanced_pred = balanced_strategy.generate(1)[0]["numbers"]

            # Count votes for each number
            votes = Counter()
            for num in hot_pred:
                votes[num] += 1.2  # Slightly favor hot
            for num in cold_pred:
                votes[num] += 0.8
            for num in balanced_pred:
                votes[num] += 1.0

            # Select top 6 by votes
            selected = [num for num, _ in votes.most_common(6)]

            # If ties, add from frequency analysis
            if len(selected) < 6:
                freq = self.number_analyzer.calculate_frequency()
                remaining = sorted(
                    [n for n in self.all_numbers if n not in selected],
                    key=lambda n: freq[n]["total_appearances"],
                    reverse=True
                )
                selected.extend(remaining[:6 - len(selected)])

            predictions.append({
                "numbers": sorted(selected[:6]),
                "strategy": "ensemble",
                "confidence": 0.65,
                "reasoning": "Combined hot, cold, and balanced strategies"
            })

        return predictions


class TOTOPredictor:
    """Main predictor class that combines all strategies."""

    STRATEGIES = {
        "hot": HotNumbersStrategy,
        "cold": ColdNumbersStrategy,
        "balanced": BalancedStrategy,
        "ml": LSTMStrategy,
        "ensemble": EnsembleStrategy,
    }

    def __init__(self, session: Session, max_draw_number: Optional[int] = None):
        self.session = session
        self.max_draw_number = max_draw_number

    def predict(
        self,
        strategy: str = "ensemble",
        count: int = 1
    ) -> list[dict]:
        """Generate predictions using specified strategy."""
        strategy_lower = strategy.lower()

        if strategy_lower not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.STRATEGIES.keys())}")

        strategy_class = self.STRATEGIES[strategy_lower]
        predictor = strategy_class(self.session, self.max_draw_number)

        predictions = predictor.generate(count)

        # Add metadata
        for pred in predictions:
            pred["generated_at"] = datetime.now().isoformat()

        return predictions

    def predict_all_strategies(self, count_per_strategy: int = 1) -> dict:
        """Generate predictions from all strategies."""
        results = {}
        for strategy_name in self.STRATEGIES:
            try:
                results[strategy_name] = self.predict(strategy_name, count_per_strategy)
            except Exception as e:
                results[strategy_name] = [{"error": str(e)}]
        return results

    def train_ml_model(self, epochs: int = LSTM_EPOCHS) -> dict:
        """Train the ML model."""
        lstm_strategy = LSTMStrategy(self.session, self.max_draw_number)
        return lstm_strategy.train(epochs)
