import os, logging
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import Huber

from config import WINDOW_SIZE, HORIZONS, FEATURES, BATCH_SIZE, REG_MODEL_PATH, CLS_MODEL_PATH
from indicators import add_indicators


def create_classification_labels(df, horizon):
    """
    3-class 레이블 생성: 하락(0), 보합(1), 상승(2)
    """
    df = df.copy()
    df['ret'] = np.log(df['close'].shift(-horizon) / df['close'])

    lower = np.percentile(df['ret'].dropna(), 30)
    upper = np.percentile(df['ret'].dropna(), 70)

    def label_fn(r):
        if r <= lower:
            return 0  # 하락
        elif r >= upper:
            return 2  # 상승
        else:
            return 1  # 보합

    df['class'] = df['ret'].apply(label_fn)
    return df.dropna(subset=['class'])


class ModelManager:
    def __init__(self, df_hist, classification_horizon):
        self.scalers    = {f: StandardScaler() for f in FEATURES}
        self.label_mean = {}
        self.label_std  = {}
        self._fit_stats(df_hist)

        self.reg_models = {
            h: self._load_or_train_reg(h, df_hist)
            for h in HORIZONS
        }

        self.classifier = self._load_or_train_clf(df_hist, classification_horizon)

    def _fit_stats(self, df):
        df2 = add_indicators(df)
        for f in FEATURES:
            self.scalers[f].fit(df2[f].values.reshape(-1, 1))

        for h in HORIZONS:
            rets = np.log(df2['close'].shift(-h) / df2['close']).dropna()
            mean = rets.mean()
            std  = max(rets.std(), 0.0005)
            self.label_mean[h] = mean
            self.label_std[h]  = std
            logging.info(f"[H{h}] mean={mean:.6f}, std={std:.6f}")

    def _build_regressor(self, input_shape):
        m = Sequential([
            Input(shape=input_shape),
            LSTM(128, return_sequences=True), Dropout(0.2),
            LSTM(64),                        Dropout(0.2),
            Dense(1),
        ])
        m.compile(optimizer=Adam(1e-4), loss=Huber())
        return m

    def _load_or_train_reg(self, horizon, df):
        path = REG_MODEL_PATH.format(horizon=horizon)
        model = self._build_regressor((WINDOW_SIZE, len(FEATURES)))

        if os.path.exists(path):
            try:
                model.load_weights(path)
                logging.info(f"✅ Regressor H{horizon} loaded")
                return model
            except:
                logging.warning(f"⚠️ Load failed for regressor H{horizon}, retraining")

        df2 = add_indicators(df)
        feats = np.hstack([
            self.scalers[f].transform(df2[f].values.reshape(-1, 1))
            for f in FEATURES
        ])
        X, y, w = [], [], []
        vol = df2['close'].rolling(WINDOW_SIZE).std().shift(-horizon).fillna(0).values

        for i in range(len(feats) - WINDOW_SIZE - horizon + 1):
            X.append(feats[i : i+WINDOW_SIZE])
            prev = df2['close'].iloc[i+WINDOW_SIZE-1]
            fut  = df2['close'].iloc[i+WINDOW_SIZE+horizon-1]
            ret  = np.log(fut / prev)
            y.append(ret)
            w.append(vol[i+WINDOW_SIZE-1] or 1.0)

        X, y, w = map(np.array, (X, y, w))
        w /= w.mean()

        model.fit(
            X, y,
            sample_weight=w,
            epochs=50,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            callbacks=[
                EarlyStopping(patience=5, restore_best_weights=True),
                ModelCheckpoint(path, save_best_only=True),
                ReduceLROnPlateau(patience=2)
            ],
            verbose=1
        )
        model.save_weights(path)
        return model

    def _build_classifier(self, input_shape):
        m = Sequential([
            Input(shape=input_shape),
            LSTM(128, return_sequences=True), Dropout(0.3),
            LSTM(64),                        Dropout(0.3),
            Dense(3, activation='softmax'),  # <-- 3-class
        ])
        m.compile(
            optimizer=Adam(1e-4),
            loss='categorical_crossentropy',  # <-- categorical loss
            metrics=['accuracy']
        )
        return m

    def _load_or_train_clf(self, df, horizon):
        path = CLS_MODEL_PATH
        model = self._build_classifier((WINDOW_SIZE, len(FEATURES)))

        if os.path.exists(path):
            try:
                model.load_weights(path)
                logging.info("✅ Classifier loaded")
                return model
            except:
                logging.warning("⚠️ Classifier load failed, retraining")

        df2 = add_indicators(df)
        df2 = create_classification_labels(df2, horizon)

        feats = np.hstack([
            self.scalers[f].transform(df2[f].values.reshape(-1, 1))
            for f in FEATURES
        ])
        X, y = [], []

        for i in range(len(df2) - WINDOW_SIZE - horizon + 1):
            window = feats[i : i+WINDOW_SIZE]
            label  = df2['class'].iloc[i+WINDOW_SIZE-1]
            X.append(window)
            y.append(label)

        X = np.array(X)
        y = to_categorical(y, num_classes=3)

        model.fit(
            X, y,
            epochs=50,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            callbacks=[
                EarlyStopping(patience=5, restore_best_weights=True),
                ModelCheckpoint(path, save_best_only=True),
                ReduceLROnPlateau(patience=2)
            ],
            verbose=1
        )
        model.save_weights(path)
        return model
