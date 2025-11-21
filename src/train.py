# src/train.py
from itertools import product
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN, LSTM
from tensorflow.keras.optimizers import Adam


# ======== MODELOS ========

def build_dense_model(input_dim, hidden_units=64, lr=1e-3):
    model = Sequential([
        Dense(hidden_units, activation='relu', input_dim=input_dim),
        Dense(hidden_units // 2, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_rnn_model(vocab_size, embedding_dim=100, rnn_units=64, lr=1e-3):
    model = Sequential([
        Embedding(vocab_size, embedding_dim),
        SimpleRNN(rnn_units),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm_model(vocab_size, embedding_dim=100, lstm_units=64, lr=1e-3):
    model = Sequential([
        Embedding(vocab_size, embedding_dim),
        LSTM(lstm_units),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ======== ENTRENAMIENTO SIMPLE ========
def train_model(model, X_train, y_train, X_val, y_val, epochs=5, batch_size=32):
    """Entrena un modelo y devuelve su historial."""
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    val_acc = history.history['val_accuracy'][-1]
    return model, history, val_acc


# ======== GRID SEARCH MANUAL ========
def grid_search_dense(X_train, y_train, X_val, y_val, input_dim):
    """Realiza un Grid Search sobre la red densa."""
    param_grid = {
        "hidden_units": [64, 128],
        "lr": [1e-3, 5e-4],
        "batch_size": [32, 64],
        "epochs": [5]
    }

    best_score = 0
    best_params = None
    best_model = None

    # producto cartesiano de parámetros
    for hidden_units, lr, batch_size, epochs in product(
        param_grid["hidden_units"],
        param_grid["lr"],
        param_grid["batch_size"],
        param_grid["epochs"]
    ):
        print(f"\n Entrenando Dense({hidden_units}) | lr={lr} | batch={batch_size} | epochs={epochs}")
        model = build_dense_model(input_dim=input_dim, hidden_units=hidden_units, lr=lr)
        model, history, val_acc = train_model(model, X_train, y_train, X_val, y_val,
                                              epochs=epochs, batch_size=batch_size)

        print(f"Val Accuracy: {val_acc:.4f}")

        if val_acc > best_score:
            best_score = val_acc
            best_params = (hidden_units, lr, batch_size, epochs)
            best_model = model

    print("\nMejor combinación encontrada:")
    print(f"Hidden Units: {best_params[0]} | LR: {best_params[1]} | Batch: {best_params[2]} | Epochs: {best_params[3]}")
    print(f"Mejor Val Accuracy: {best_score:.4f}")

    return best_model, best_params, best_score