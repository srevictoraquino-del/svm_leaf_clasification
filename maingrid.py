import cv2
import glob
import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def calcular_moda_numpy(imagen_array):
    valores, cuentas = np.unique(imagen_array, return_counts=True)
    indice_maximo = np.argmax(cuentas)
    return valores[indice_maximo]

def procesamiento(ruta_origen, clase):
    resultados = []
    # Verificar si hay imágenes en la ruta
    archivos = glob.glob(ruta_origen)
    if not archivos:
        print(f"Advertencia: No se encontraron imágenes en {ruta_origen}")
        return

    for ruta in archivos:
        img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        re_size = cv2.resize(img, (64,64))

        h, w = re_size.shape
        # Cuadrantes
        q1 = re_size[0:h//2, 0:w//2].mean() # Superior Izquierda
        q2 = re_size[0:h//2, w//2:w].mean() # Superior Derecha
        q3 = re_size[h//2:h, 0:w//2].mean() # Inferior Izquierda
        q4 = re_size[h//2:h, w//2:w].mean() # Inferior Derecha

        nueva_mediana = np.median(re_size)
        nueva_moda = calcular_moda_numpy(re_size)
        val_max = re_size.max()
        val_min = re_size.min()
        des_est = re_size.std()

        resultados.append([nueva_mediana, nueva_moda, val_max, val_min, des_est, q1, q2, q3, q4, clase])

    if resultados:
        df = pd.DataFrame(resultados)
        # Guardar en CSV (mode='a' anexa al archivo existente)
        df.to_csv("dataset.CSV", mode='a', index=False, header=False)
        print(f"Procesadas {len(resultados)} imágenes de la clase {clase}")

def entrenar_svm(ruta_dataset):
    print("\n--- Iniciando Entrenamiento SVM con Grid Search ---")
    
    # 1. Cargar datos
    if not os.path.exists(ruta_dataset):
        print("Error: No se encontró el archivo dataset.CSV")
        return

    data = pd.read_csv(ruta_dataset, header=None)
    X = data.iloc[:, :-1].values # Características
    y = data.iloc[:, -1].values  # Etiquetas

    print(f"Datos cargados: {X.shape[0]} instancias, {X.shape[1]} características.")

    # 2. Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

    # 3. ESCALAR DATOS (CRUCIAL para SVM Poly y velocidad)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. Definir parámetros para Grid Search
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.1, 0.01],
        'kernel': ['linear', 'rbf', 'poly','sigmoid'] 
    }

    # 5. Configurar y ejecutar Grid Search
    # n_jobs=-1 usa todos los núcleos del procesador
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=5, n_jobs=-1)
    
    print("Buscando los mejores hiperparámetros...")
    grid.fit(X_train, y_train)

    # 6. Resultados
    print(f"\nMejores parámetros encontrados: {grid.best_params_}")
    print(f"Mejor score en validación: {grid.best_score_:.4f}")

    # 7. Evaluar el mejor modelo
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("\nReporte de Clasificación en Test Set:")
    print(classification_report(y_test, y_pred))
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Exactitud del modelo SVM: {acc*100:.2f}%")

    return best_model

def main(): 
    start_time = time.perf_counter()
    
    # --- LIMPIEZA PREVIA ---
    # Eliminar dataset previo para evitar datos duplicados
    if os.path.exists("dataset.CSV"):
        os.remove("dataset.CSV")
        print("Archivo 'dataset.CSV' anterior eliminado.")
    
    # Procesar imágenes
    print("Procesando imágenes...")
    procesamiento('A/*.JPG','A')
    procesamiento('B/*.JPG','B')
    procesamiento('C/*.JPG','C')
    procesamiento('D/*.JPG','D')
    
    # Verificar si se creó el dataset
    if os.path.exists("dataset.CSV"):
        # Entrenar SVM con GridSearch
        entrenar_svm("dataset.CSV")
    else:
        print("No se creó el dataset. Revisa las rutas de las imágenes.")

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"\nTiempo total de ejecución: {elapsed:.2f} segundos")

if __name__ == '__main__':
    main()