from itertools import product
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN, LSTM
from tensorflow.keras.optimizers import Adam
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args


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


# ======== MANUAL GRID SEARCH ========
def grid_search_dense(X_train, y_train, X_val, y_val, input_dim):
    """Performs a grid search over the dense network."""
    param_grid = {
        "hidden_units": [64, 128],
        "lr": [1e-3, 5e-4],
        "batch_size": [32, 64],
        "epochs": [5]
    }

    best_score = 0
    best_params = None
    best_model = None
    best_history = None   # <-- NUEVO

    # cartesian product of parameters
    for hidden_units, lr, batch_size, epochs in product(
        param_grid["hidden_units"],
        param_grid["lr"],
        param_grid["batch_size"],
        param_grid["epochs"]
    ):
        print(f"\nTraining Dense({hidden_units}) | lr={lr} | batch={batch_size} | epochs={epochs}")
        
        model = build_dense_model(input_dim=input_dim, hidden_units=hidden_units, lr=lr)
        
        model, history, val_acc = train_model(
            model, X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size
        )

        print(f"Val Accuracy: {val_acc:.4f}")

        if val_acc > best_score:
            best_score = val_acc
            best_params = (hidden_units, lr, batch_size, epochs)
            best_model = model
            best_history = history

    print("\nBest combination found:")
    print(f"Hidden Units: {best_params[0]} | LR: {best_params[1]} | Batch: {best_params[2]} | Epochs: {best_params[3]}")
    print(f"Best Val Accuracy: {best_score:.4f}")
    return best_model, best_history, best_params, best_score


# ======== BAYESIAN OPTIMIZATION FOR VANILLA RNN ========
def bayes_opt_rnn(X_train, y_train, X_val, y_val, vocab_size):

    space = [
        Integer(100, 300, name='embedding_dim'),  # más capacidad
        Integer(64, 256, name='rnn_units'),       # más memoria
        Real(1e-4, 5e-3, "log-uniform", name='lr'), # rango más estable
        Integer(32, 128, name='batch_size')       # mayor estabilidad
    ]

    best_history = None
    best_model = None
    best_val_acc = -1

    @use_named_args(space)
    def objective(embedding_dim, rnn_units, lr, batch_size):
        nonlocal best_history, best_model, best_val_acc

        print(f"\n Testing RNN: emb={embedding_dim}, units={rnn_units}, lr={lr:.6f}, batch={batch_size}")

        model = build_rnn_model(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            rnn_units=rnn_units,
            lr=lr
        )

        model, history, val_acc = train_model(
            model, X_train, y_train, X_val, y_val,
            epochs=10,         # antes 5 → demasiado poco
            batch_size=batch_size
        )

        print(f"   → Val Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_history = history
            best_model = model

        return -val_acc

    res = gp_minimize(
        func=objective,
        dimensions=space,
        acq_func="EI",
        n_calls=35,
        n_random_starts=10,
        random_state=42
    )

    best_params = {
        "embedding_dim": res.x[0],
        "rnn_units": res.x[1],
        "lr": res.x[2],
        "batch_size": res.x[3],
    }

    best_score = -res.fun

    print(" Best hyperparameters found (Bayesian Optimization)")
    print(best_params)
    print(f" Best Val Accuracy: {best_score:.4f}")

    return best_model, best_history, best_params, best_score

# ======== BAYESIAN OPTIMIZATION FOR LSTM ========
def bayes_opt_lstm(
    X_train, y_train, 
    X_val, y_val, 
    vocab_size
):

    # Espacio de búsqueda (adaptado para LSTM)
    space = [
        Integer(100, 300, name='embedding_dim'),
        Integer(64, 256, name='lstm_units'),
        Real(1e-4, 5e-3, "log-uniform", name='lr'),
        Integer(32, 128, name='batch_size'),
    ]

    best_history = None
    best_model = None
    best_val_acc = -1

    @use_named_args(space)
    def objective(embedding_dim, lstm_units, lr, batch_size):
        nonlocal best_history, best_model, best_val_acc

        print(f"\n Testing LSTM: emb={embedding_dim}, units={lstm_units}, lr={lr:.6f}, batch={batch_size}")

        # construir modelo LSTM
        model = build_lstm_model(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            lstm_units=lstm_units,
            lr=lr,
        )

        # entrenar
        model, history, val_acc = train_model(
            model,
            X_train, y_train,
            X_val, y_val,
            epochs=10,
            batch_size=batch_size
        )

        print(f"   - Val Accuracy: {val_acc:.4f}")

        # guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_history = history
            best_model = model

        return -val_acc  # porque gp_minimize minimiza

    # BO con función de adquisición buena para deep learning → gp_hedge
    res = gp_minimize(
        func=objective,
        dimensions=space,
        acq_func="gp_hedge",
        n_calls=40,
        n_random_starts=10,
        random_state=42
    )

    best_params = {
        "embedding_dim": res.x[0],
        "lstm_units": res.x[1],
        "lr": res.x[2],
        "batch_size": res.x[3],
    }

    best_score = -res.fun

    print("\n Best hyperparameters found (Bayesian Optimization for LSTM):")
    print(best_params)
    print(f" Best Val Accuracy: {best_score:.4f}")

    return best_model, best_history, best_params, best_score