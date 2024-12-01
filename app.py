import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import optuna
import joblib
import io

# Imposta il seed per la riproducibilità
np.random.seed(42)
tf.random.set_seed(42)

st.title("App di Previsione dei Prezzi XAUUSD con LSTM")

st.sidebar.header("Carica Dati")
uploaded_file = st.sidebar.file_uploader("Scegli un file CSV", type=["csv"])

if uploaded_file is not None:
    # Leggi il file caricato
    df = pd.read_csv(uploaded_file, names=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'], header=0)
    
    # Preprocessamento dei dati
    df['Time'] = pd.to_datetime(df['Time'], format='%Y.%m.%d %H:%M:%S')
    df.set_index('Time', inplace=True)
    
    st.write("### Dati Caricati")
    st.dataframe(df.head())

    # Aggiunta di indicatori tecnici
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def calculate_sma(series, period=20):
        return series.rolling(window=period).mean()

    def calculate_ema(series, period=20):
        return series.ewm(span=period, adjust=False).mean()

    def calculate_macd(series, slow=26, fast=12):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        return ema_fast - ema_slow

    # Calcolo degli indicatori tecnici
    df['RSI'] = calculate_rsi(df['Close'])
    df['SMA_20'] = calculate_sma(df['Close'], period=20)
    df['EMA_20'] = calculate_ema(df['Close'], period=20)
    df['MACD'] = calculate_macd(df['Close'])

    # Aggiunta di lag features
    lag_features = st.sidebar.multiselect("Seleziona i lag per il Close", options=[1,2,3], default=[1,2,3])
    for lag in lag_features:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)

    # Rimozione dei valori NaN
    df.dropna(inplace=True)

    # Seleziona le caratteristiche da utilizzare
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'SMA_20', 'EMA_20', 'MACD'] + [f'Close_lag_{lag}' for lag in lag_features]
    target_column = 'Close'

    st.write("### Caratteristiche Utilizzate")
    st.write(features)

    # Standardizzazione
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    df[features] = feature_scaler.fit_transform(df[features])
    df[target_column] = target_scaler.fit_transform(df[[target_column]])

    # Creazione delle sequenze
    def create_sequences(data, seq_length, target_column):
        X, y = [], []
        data_array = data[features].values
        target = data[target_column].values
        for i in range(seq_length, len(data_array) - 1):
            X.append(data_array[i - seq_length:i])
            y.append(target[i])
        return np.array(X), np.array(y)

    # Parametri di sequenza
    seq_length = st.sidebar.number_input("Lunghezza della Sequenza", min_value=10, max_value=200, value=60, step=10)

    X, y = create_sequences(df, seq_length, target_column)

    # Suddivisione dei dati
    train_size = int(len(X) * 0.8)
    X_train_full, X_test = X[:train_size], X[train_size:]
    y_train_full, y_test = y[:train_size], y[train_size:]

    # Ulteriore suddivisione per validazione
    val_size = int(len(X_train_full) * 0.1)
    X_train, X_val = X_train_full[:-val_size], X_train_full[-val_size:]
    y_train, y_val = y_train_full[:-val_size], y_train_full[-val_size:]

    st.write("### Dimensioni dei Dataset")
    st.write(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

    # Parametri del modello
    st.sidebar.header("Parametri del Modello")
    lstm_units = st.sidebar.selectbox("Unità LSTM", options=[32, 64, 128, 256], index=1)
    dropout_rate = st.sidebar.slider("Tasso di Dropout", min_value=0.0, max_value=0.5, value=0.2, step=0.05)
    learning_rate = st.sidebar.slider("Tasso di Apprendimento", min_value=1e-5, max_value=1e-3, value=1e-4, step=1e-5, format="%.5f")
    num_layers = st.sidebar.selectbox("Numero di Layer LSTM", options=[1, 2, 3], index=1)
    epochs = st.sidebar.number_input("Numero di Epoche", min_value=10, max_value=500, value=100, step=10)
    batch_size = st.sidebar.number_input("Dimensione del Batch", min_value=16, max_value=512, value=64, step=16)

    # Funzione per creare il modello LSTM parametrizzato
    def create_lstm_model(seq_length, num_features, lstm_units, dropout_rate, learning_rate, num_layers):
        inputs = Input(shape=(seq_length, num_features))
        x = inputs
        for i in range(num_layers):
            return_sequences = True if i < num_layers - 1 else False
            x = LSTM(units=lstm_units, return_sequences=return_sequences)(x)
            x = Dropout(dropout_rate)(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mae'])
        return model

    # Funzione obiettivo per Optuna
    def objective(trial):
        lstm_units = trial.suggest_categorical('lstm_units', [32, 64, 128, 256])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
        num_layers = trial.suggest_int('num_layers', 1, 3)

        model = create_lstm_model(seq_length, len(features), lstm_units, dropout_rate, learning_rate, num_layers)

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=64,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0
        )

        val_loss = min(history.history['val_loss'])
        return val_loss

    # Bottone per avviare l'addestramento
    if st.button("Avvia Addestramento"):
        with st.spinner('Addestramento in corso...'):
            # Creazione e addestramento del modello
            model = create_lstm_model(
                seq_length, 
                len(features), 
                lstm_units, 
                dropout_rate, 
                learning_rate, 
                num_layers
            )

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            history = model.fit(
                X_train_full, y_train_full,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=1
            )

            # Valutazione del modello
            test_loss, test_mae = model.evaluate(X_test, y_test)
            st.write(f"**Perdita sul test set:** {test_loss}")
            st.write(f"**MAE sul test set:** {test_mae}")

            # Previsioni
            y_pred = model.predict(X_test)

            # Denormalizzazione
            y_test_denorm = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_denorm = target_scaler.inverse_transform(y_pred).flatten()

            # Visualizzazione delle previsioni
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(y_test_denorm, label='Reale')
            ax1.plot(y_pred_denorm, label='Predetto', alpha=0.7)
            ax1.set_title('Previsioni dei Prezzi XAUUSD')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Prezzo')
            ax1.legend()
            st.pyplot(fig1)

            # Visualizzazione dell'andamento della perdita durante l'addestramento
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.plot(history.history['loss'], label='Perdita di Addestramento')
            ax2.plot(history.history['val_loss'], label='Perdita di Validazione')
            ax2.set_xlabel('Epoche')
            ax2.set_ylabel('Perdita')
            ax2.legend()
            st.pyplot(fig2)

            # Salvataggio del modello e degli scaler
            buffer_model = io.BytesIO()
            tf.keras.models.save_model(model, 'model.h5')
            with open('model.h5', 'rb') as f:
                buffer_model.write(f.read())

            buffer_scalers = io.BytesIO()
            joblib.dump(feature_scaler, 'feature_scaler.pkl')
            joblib.dump(target_scaler, 'target_scaler.pkl')
            with open('feature_scaler.pkl', 'rb') as f:
                buffer_scalers.write(f.read())
            with open('target_scaler.pkl', 'rb') as f:
                buffer_scalers.write(f.read())

            st.download_button(
                label="Scarica il Modello Addestrato",
                data=buffer_model.getvalue(),
                file_name="model.h5",
                mime="application/octet-stream"
            )

            st.download_button(
                label="Scarica gli Scaler",
                data=buffer_scalers.getvalue(),
                file_name="scalers.zip",
                mime="application/zip"
            )

    # Sezione per l'ottimizzazione degli iperparametri con Optuna
    st.sidebar.header("Ottimizzazione degli Iperparametri con Optuna")
    n_trials = st.sidebar.number_input("Numero di Prove Optuna", min_value=10, max_value=1000, value=50, step=10)
    
    if st.sidebar.button("Avvia Ottimizzazione Optuna"):
        with st.spinner('Ottimizzazione in corso...'):
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials)
            st.write("### Migliori Iperparametri Trovati")
            st.json(study.best_params)

            # Visualizzazione del grafico delle prestazioni
            fig3, ax3 = plt.subplots()
            optuna.visualization.plot_optimization_history(study)
            st.pyplot(fig3)

            # Addestramento finale con i migliori iperparametri
            best_params = study.best_params
            model_best = create_lstm_model(
                seq_length, 
                len(features), 
                best_params['lstm_units'], 
                best_params['dropout_rate'], 
                best_params['learning_rate'], 
                best_params['num_layers']
            )

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            history_best = model_best.fit(
                X_train_full, y_train_full,
                epochs=100,
                batch_size=64,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=1
            )

            # Valutazione del modello
            test_loss_best, test_mae_best = model_best.evaluate(X_test, y_test)
            st.write(f"**[Best] Perdita sul test set:** {test_loss_best}")
            st.write(f"**[Best] MAE sul test set:** {test_mae_best}")

            # Previsioni
            y_pred_best = model_best.predict(X_test)

            # Denormalizzazione
            y_test_denorm_best = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_denorm_best = target_scaler.inverse_transform(y_pred_best).flatten()

            # Visualizzazione delle previsioni
            fig4, ax4 = plt.subplots(figsize=(12, 6))
            ax4.plot(y_test_denorm_best, label='Reale')
            ax4.plot(y_pred_denorm_best, label='Predetto', alpha=0.7)
            ax4.set_title('Previsioni dei Prezzi XAUUSD (Best Model)')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Prezzo')
            ax4.legend()
            st.pyplot(fig4)

            # Visualizzazione dell'andamento della perdita durante l'addestramento
            fig5, ax5 = plt.subplots(figsize=(12, 6))
            ax5.plot(history_best.history['loss'], label='Perdita di Addestramento')
            ax5.plot(history_best.history['val_loss'], label='Perdita di Validazione')
            ax5.set_xlabel('Epoche')
            ax5.set_ylabel('Perdita')
            ax5.legend()
            st.pyplot(fig5)
