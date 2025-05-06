from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import json
import os

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Modelo global
model = None
loss_history = []
trained = False

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,))
    ])
    
    model.compile(
        optimizer='sgd',
        loss='mse'
    )
    
    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    global model, loss_history, trained
    loss_history = []
    data = request.get_json()
    x_train = np.array(data['x'], dtype=float)
    y_train = np.array(data['y'], dtype=float)

    model = create_model()
    
    class LossHistory(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            loss_history.append(float(logs.get('loss')))
    
    # Entrenar modelo
    epochs = int(data.get('epochs', 100))
    model.fit(
        x_train.reshape(-1, 1), y_train, 
        epochs=epochs, 
        verbose=0,
        callbacks=[LossHistory()]
    )
    
    trained = True
    weights = float(model.layers[0].get_weights()[0][0][0])
    bias = float(model.layers[0].get_weights()[1][0])
    
    return jsonify({
        'status': 'success',
        'weights': weights,
        'bias': bias,
        'loss_history': loss_history
    })

@app.route('/predict', methods=['POST'])
def predict():
    global model, trained
    
    if not trained or model is None:
        return jsonify({
            'status': 'error',
            'message': 'El modelo no ha sido entrenado'
        }), 400
    
    data = request.get_json()
    x_values = np.array(data['x_values'], dtype=float)
    
    # Realizar predicciones
    predictions = model.predict(x_values.reshape(-1, 1)).flatten().tolist()
    
    return jsonify({
        'status': 'success',
        'predictions': predictions,
        'x_values': x_values.tolist()
    })

@app.route('/loss_history', methods=['GET'])
def get_loss_history():
    global loss_history
    
    return jsonify({
        'status': 'success',
        'loss_history': loss_history
    })

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    app.run(debug=True)