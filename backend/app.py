"""
app.py — Backend Flask para predicción de Cáncer de Mama
Dataset: Breast Cancer Wisconsin (Diagnostic) - UCI
Modelos: Regresión Logística | Red Neuronal Artificial (MLP)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import os

from flask import render_template

app = Flask(
    __name__,
    static_folder='../frontend',
    template_folder='../frontend'
)
CORS(app)  # Permite peticiones desde el frontend (GitHub Pages)
@app.route("/")
def home():
    return "API funcionando correctamente "

# ── Rutas a los modelos guardados ────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), '..', 'modelos', 'guardados')

def cargar(nombre):
    with open(os.path.join(BASE, nombre), 'rb') as f:
        return pickle.load(f)

# Cargar todo al iniciar el servidor
modelo_logreg  = cargar('modelo_logreg.pkl')
scaler_logreg  = cargar('scaler_logreg.pkl')
modelo_mlp     = cargar('modelo_mlp.pkl')
scaler_mlp     = cargar('scaler_mlp.pkl')
columnas       = cargar('columnas.pkl')      # lista con los 30 nombres de features
label_encoder  = cargar('label_encoder.pkl') # B=0, M=1

print("✅ Modelos cargados correctamente")
print(f"   Columnas esperadas: {len(columnas)}")

# ── Helper: seleccionar modelo y scaler ──────────────────────
def get_modelo(nombre):
    """Retorna (modelo, scaler) según el nombre recibido del frontend."""
    nombre = nombre.lower()
    if nombre == 'logreg':
        return modelo_logreg, scaler_logreg
    elif nombre == 'mlp':
        return modelo_mlp, scaler_mlp
    else:
        raise ValueError(f"Modelo desconocido: {nombre}. Use 'logreg' o 'mlp'.")

# ── Helper: decodificar predicción ───────────────────────────
def decodificar(pred_num):
    """Convierte 0/1 → 'Benigno'/'Maligno'."""
    label = label_encoder.inverse_transform([pred_num])[0]
    return 'Benigno' if label == 'B' else 'Maligno'

# ════════════════════════════════════════════════════════════
# ENDPOINT 1 — Predicción Individual
# POST /predecir
# Body JSON: { "modelo": "logreg"|"mlp", "datos": { feature: valor, ... } }
# ════════════════════════════════════════════════════════════
@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        body   = request.get_json()
        nombre = body.get('modelo', 'logreg')
        datos  = body.get('datos', {})

        # Validar que vengan todas las columnas
        faltantes = [c for c in columnas if c not in datos]
        if faltantes:
            return jsonify({'error': f'Faltan campos: {faltantes}'}), 400

        # Construir array en el orden correcto
        X = np.array([[float(datos[c]) for c in columnas]])

        modelo, scaler = get_modelo(nombre)
        X_sc     = scaler.transform(X)
        pred_num = int(modelo.predict(X_sc)[0])
        proba    = modelo.predict_proba(X_sc)[0].tolist()  # [p_benigno, p_maligno]

        return jsonify({
            'prediccion'  : decodificar(pred_num),
            'clase'       : pred_num,          # 0=Benigno, 1=Maligno
            'probabilidad': {
                'Benigno': round(proba[0] * 100, 2),
                'Maligno': round(proba[1] * 100, 2)
            },
            'modelo_usado': nombre.upper()
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Error interno: {str(e)}'}), 500


# ════════════════════════════════════════════════════════════
# ENDPOINT 2 — Predicción por Lotes (CSV)
# POST /predecir-lotes
# Body: multipart/form-data con campo "archivo" (CSV) y "modelo"
# El CSV debe tener las 30 columnas de features + columna "Diagnosis" (opcional)
# ════════════════════════════════════════════════════════════
@app.route('/predecir-lotes', methods=['POST'])
def predecir_lotes():
    try:
        nombre  = request.form.get('modelo', 'logreg')
        archivo = request.files.get('archivo')

        if not archivo:
            return jsonify({'error': 'No se recibió ningún archivo CSV'}), 400

        df = pd.read_csv(archivo)

        # Verificar si trae columna de etiquetas reales (para métricas)
        tiene_etiquetas = 'Diagnosis' in df.columns

        # Separar features
        X_df = df[columnas] if all(c in df.columns for c in columnas) else None
        if X_df is None:
            faltantes = [c for c in columnas if c not in df.columns]
            return jsonify({'error': f'Columnas faltantes en CSV: {faltantes}'}), 400

        modelo, scaler = get_modelo(nombre)
        X_sc     = scaler.transform(X_df.values)
        preds    = modelo.predict(X_sc)
        probas   = modelo.predict_proba(X_sc)

        # Construir lista de resultados por fila
        resultados = []
        for i, (pred, proba) in enumerate(zip(preds, probas)):
            resultados.append({
                'fila'       : i + 1,
                'prediccion' : decodificar(int(pred)),
                'clase'      : int(pred),
                'prob_benigno': round(float(proba[0]) * 100, 2),
                'prob_maligno': round(float(proba[1]) * 100, 2)
            })

        respuesta = {
            'total'       : len(preds),
            'modelo_usado': nombre.upper(),
            'predicciones': resultados,
            'resumen': {
                'Benigno': int(np.sum(preds == 0)),
                'Maligno': int(np.sum(preds == 1))
            }
        }

        # Si el CSV tiene etiquetas reales → calcular métricas
        if tiene_etiquetas:
            y_real = label_encoder.transform(df['Diagnosis'].values)

            cm = confusion_matrix(y_real, preds).tolist()  # [[TN,FP],[FN,TP]]

            respuesta['metricas'] = {
                'accuracy'        : round(accuracy_score(y_real, preds), 4),
                'precision'       : round(precision_score(y_real, preds), 4),
                'recall'          : round(recall_score(y_real, preds), 4),
                'f1_score'        : round(f1_score(y_real, preds), 4),
                'confusion_matrix': cm
            }

        return jsonify(respuesta)

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Error interno: {str(e)}'}), 500


# ════════════════════════════════════════════════════════════
# ENDPOINT 3 — Info de columnas (para que el frontend genere el formulario)
# GET /columnas
# ════════════════════════════════════════════════════════════
@app.route('/columnas', methods=['GET'])
def get_columnas():
    return jsonify({'columnas': columnas, 'total': len(columnas)})


# ── Verificación de salud del servidor ───────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'modelos': ['logreg', 'mlp']})


if __name__ == '__main__':
    app.run(debug=True, port=5000)