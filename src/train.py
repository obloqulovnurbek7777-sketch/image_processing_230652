import tensorflow as tf
from model import build_cnn_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def train_pipeline(train_ds, val_ds, epochs=30):
    """
    Compiles and executes the training process using standard Callbacks.
    """
    model = build_cnn_model()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks configuration
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

    print("Starting Model Training Pipeline...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    return model, history
