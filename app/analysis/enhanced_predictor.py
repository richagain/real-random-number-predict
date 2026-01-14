"""Enhanced ML prediction with better feature engineering and model architecture."""

import numpy as np
from collections import Counter
from typing import Optional
from sqlalchemy import select, desc
from sqlalchemy.orm import Session

from app.database.models import Draw
from app.config import MODEL_PATH, TOTO_MIN_NUMBER, TOTO_MAX_NUMBER


class EnhancedPredictor:
    """Enhanced predictor with advanced feature engineering."""

    def __init__(self, session: Session):
        self.session = session
        self.model = None
        self.model_path = MODEL_PATH / "enhanced_model.keras"
        self.scaler = None
        self.sequence_length = 30  # Use more history

    def get_all_draws(self) -> list[Draw]:
        """Get all draws sorted by draw number."""
        result = self.session.execute(
            select(Draw).order_by(Draw.draw_number)
        )
        return list(result.scalars().all())

    def extract_features(self, draws: list[Draw], target_idx: int) -> np.ndarray:
        """Extract rich features from historical draws.

        Features per number (49 numbers):
        1. Frequency in last N draws (normalized)
        2. Gap since last appearance (normalized)
        3. Position frequency (which position it appeared in)
        4. Recent trend (frequency in last 10 vs last 50)

        Global features:
        5. Average sum of recent draws
        6. Odd/even ratio trend
        7. High/low ratio trend
        """
        # Get draws up to target_idx
        history = draws[:target_idx]
        if len(history) < self.sequence_length:
            return None

        recent = history[-self.sequence_length:]
        very_recent = history[-10:] if len(history) >= 10 else history

        features = []

        # Per-number features
        freq_recent = Counter()
        freq_very_recent = Counter()
        last_seen = {}
        position_freq = {i: Counter() for i in range(6)}

        for idx, draw in enumerate(recent):
            for pos, num in enumerate(draw.winning_numbers):
                freq_recent[num] += 1
                position_freq[pos][num] += 1
                last_seen[num] = idx

        for draw in very_recent:
            for num in draw.winning_numbers:
                freq_very_recent[num] += 1

        current_draw_idx = len(recent) - 1

        for num in range(1, 50):
            # Frequency (normalized)
            freq = freq_recent.get(num, 0) / (self.sequence_length * 6)
            features.append(freq)

            # Gap since last appearance (normalized)
            if num in last_seen:
                gap = (current_draw_idx - last_seen[num]) / self.sequence_length
            else:
                gap = 1.0  # Never appeared
            features.append(gap)

            # Recent trend (very recent vs recent)
            recent_freq = freq_very_recent.get(num, 0) / (len(very_recent) * 6)
            trend = recent_freq - freq
            features.append(trend)

            # Position preference (which position it appears most)
            pos_counts = [position_freq[p].get(num, 0) for p in range(6)]
            total_pos = sum(pos_counts)
            if total_pos > 0:
                pos_entropy = -sum((c/total_pos) * np.log(c/total_pos + 1e-10)
                                   for c in pos_counts if c > 0)
            else:
                pos_entropy = 0
            features.append(pos_entropy / 2.0)  # Normalize

        # Global features
        recent_sums = [sum(d.winning_numbers) for d in recent]
        avg_sum = np.mean(recent_sums) / 279  # Normalize by max possible sum
        features.append(avg_sum)

        # Odd/even ratio
        odd_counts = [sum(1 for n in d.winning_numbers if n % 2 == 1) for d in recent]
        avg_odd = np.mean(odd_counts) / 6
        features.append(avg_odd)

        # High/low ratio
        high_counts = [sum(1 for n in d.winning_numbers if n > 25) for d in recent]
        avg_high = np.mean(high_counts) / 6
        features.append(avg_high)

        # Sum trend
        if len(recent_sums) > 5:
            sum_trend = (np.mean(recent_sums[-5:]) - np.mean(recent_sums[:-5])) / 100
        else:
            sum_trend = 0
        features.append(sum_trend)

        return np.array(features)

    def prepare_training_data(self) -> tuple:
        """Prepare training data with enhanced features."""
        draws = self.get_all_draws()

        if len(draws) < self.sequence_length + 10:
            return None, None

        X = []
        y = []

        for i in range(self.sequence_length, len(draws)):
            features = self.extract_features(draws, i)
            if features is None:
                continue

            # Target: one-hot encoding of winning numbers
            target = np.zeros(49)
            for num in draws[i].winning_numbers:
                target[num - 1] = 1

            X.append(features)
            y.append(target)

        return np.array(X), np.array(y)

    def build_model(self, input_dim: int):
        """Build enhanced neural network with attention-like mechanism."""
        from tensorflow import keras
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input, Dense, Dropout, BatchNormalization,
            Concatenate, Multiply, Add
        )

        inputs = Input(shape=(input_dim,))

        # First branch - deep features
        x1 = Dense(256, activation='relu')(inputs)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.3)(x1)
        x1 = Dense(128, activation='relu')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.3)(x1)

        # Second branch - attention-like weights
        x2 = Dense(256, activation='relu')(inputs)
        x2 = Dense(128, activation='sigmoid')(x2)

        # Combine with attention
        attended = Multiply()([x1, x2])

        # Output layers
        x = Dense(98, activation='relu')(attended)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(49, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=x)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, epochs: int = 200) -> dict:
        """Train the enhanced model."""
        print("Preparing training data with enhanced features...")
        X, y = self.prepare_training_data()

        if X is None:
            return {"success": False, "error": "Not enough data"}

        print(f"Training data shape: X={X.shape}, y={y.shape}")
        print(f"Feature count: {X.shape[1]}")

        # Build model
        self.model = self.build_model(X.shape[1])
        self.model.summary()

        # Train with early stopping
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=30,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.0001
            )
        ]

        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )

        # Save model
        self.model.save(self.model_path)

        return {
            "success": True,
            "epochs_trained": len(history.history['loss']),
            "final_loss": float(history.history['loss'][-1]),
            "final_val_loss": float(history.history['val_loss'][-1]),
            "final_accuracy": float(history.history['accuracy'][-1]),
            "best_val_loss": float(min(history.history['val_loss'])),
        }

    def load_model(self) -> bool:
        """Load saved model."""
        try:
            from tensorflow import keras
            if self.model_path.exists():
                self.model = keras.models.load_model(self.model_path)
                return True
        except Exception as e:
            print(f"Error loading model: {e}")
        return False

    def predict(self, count: int = 1) -> list[dict]:
        """Generate predictions using the enhanced model."""
        if self.model is None:
            if not self.load_model():
                return [{"error": "Model not trained"}]

        draws = self.get_all_draws()
        features = self.extract_features(draws, len(draws))

        if features is None:
            return [{"error": "Not enough historical data"}]

        predictions = []

        for _ in range(count):
            # Predict probabilities
            probs = self.model.predict(features.reshape(1, -1), verbose=0)[0]

            # Get top candidates with some randomness
            # Use temperature-based sampling for variety
            temperature = 0.8
            scaled_probs = np.power(probs, 1/temperature)
            scaled_probs = scaled_probs / scaled_probs.sum()

            # Select 6 numbers weighted by probability
            selected_indices = np.random.choice(
                49, size=6, replace=False, p=scaled_probs
            )
            selected = sorted([i + 1 for i in selected_indices])

            # Calculate confidence from probabilities
            confidence = float(np.mean([probs[i-1] for i in selected]))

            predictions.append({
                "numbers": selected,
                "strategy": "enhanced_ml",
                "confidence": min(confidence, 0.7),
                "reasoning": "Enhanced neural network with attention mechanism",
                "top_probabilities": {
                    int(i+1): float(probs[i])
                    for i in np.argsort(probs)[-10:][::-1]
                }
            })

        return predictions
