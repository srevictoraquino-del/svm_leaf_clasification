import cv2
import glob
import pandas as pd
import numpy as np
import time
import os
import shutil
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Configuración
GUARDAR_DEBUG = False # Ponlo en False para que corra rápido
CARPETA_DEBUG = "debug_recortes"

def limpiar_carpeta_debug():
    if os.path.exists(CARPETA_DEBUG):
        shutil.rmtree(CARPETA_DEBUG)
    os.makedirs(CARPETA_DEBUG)

def segmentar_hoja_lab(img_bgr):
    # Segmentación basada en el canal A de LAB (Verde vs Rojo/Magenta)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    _, mask = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8) # Kernel más pequeño para no perder detalles
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask, l, a, b # Devolvemos los canales separados

def obtener_momentos_hu(mask):
    momentos = cv2.moments(mask)
    hu = cv2.HuMoments(momentos).flatten()
    processed_hu = []
    for val in hu:
        if val != 0:
            processed_hu.append(-1 * np.sign(val) * np.log10(abs(val)))
        else:
            processed_hu.append(0)
    return processed_hu

def obtener_datos_geometricos(mask):
    # Encuentra el contorno más grande (la hoja)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, 0, 0
    
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimetro = cv2.arcLength(cnt, True)
    
    # 1. Compacidad (Compactness): Relación área/perímetro^2 (Círculo es más compacto)
    if perimetro == 0: return 0, 0, 0, 0
    compacidad = (4 * np.pi * area) / (perimetro ** 2)
    
    # 2. Rectangularidad y Aspect Ratio
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h
    rect_area = w * h
    extent = float(area) / rect_area # Qué tanto llena su caja rectangular
    
    # 3. Solidez (Solidity): Área / Área del contorno convexo (sin huecos)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    return compacidad, aspect_ratio, extent, solidity

def procesamiento(ruta_origen, clase):
    resultados = []
    archivos = glob.glob(ruta_origen)
    
    if not archivos:
        print(f"No se encontraron imágenes en: {ruta_origen}")
        return

    print(f"Procesando {len(archivos)} imágenes de la clase {clase}...")

    for i, ruta in enumerate(archivos):
        try:
            img_color = cv2.imread(ruta)
            if img_color is None: continue
            
            # Segmentar y obtener canales individuales
            mask, l_channel, a_channel, b_channel = segmentar_hoja_lab(img_color)
            
            pixels = cv2.countNonZero(mask)
            if pixels < 500: continue # Basura / Máscara vacía

            # --- DEBUG ---
            if GUARDAR_DEBUG and i < 3:
                cv2.imwrite(f"{CARPETA_DEBUG}/{clase}_{os.path.basename(ruta)}_mask.jpg", mask)

            # 1. CARACTERÍSTICAS DE COLOR (Crucial: Canal A y B por separado)
            # Usamos la máscara para no contar el fondo negro
            mean_l, std_l = cv2.meanStdDev(l_channel, mask=mask)
            mean_a, std_a = cv2.meanStdDev(a_channel, mask=mask) # Verde-Rojo
            mean_b, std_b = cv2.meanStdDev(b_channel, mask=mask) # Azul-Amarillo
            
            # 2. CARACTERÍSTICAS DE TEXTURA SIMPLE
            # Varianza del canal L (Luminosidad) indica rugosidad
            textura = cv2.Laplacian(l_channel, cv2.CV_64F).var()

            # 3. CARACTERÍSTICAS GEOMÉTRICAS (Físicas)
            compacidad, aspect_ratio, extent, solidity = obtener_datos_geometricos(mask)

            # 4. MOMENTOS DE HU (Forma abstracta)
            hu = obtener_momentos_hu(mask)

            # Armar vector de características (Más robusto)
            # Flatten() es necesario porque meanStdDev devuelve matrices numpy
            features = [
                mean_l.flatten()[0], std_l.flatten()[0],
                mean_a.flatten()[0], std_a.flatten()[0],
                mean_b.flatten()[0], std_b.flatten()[0],
                textura,
                compacidad, aspect_ratio, extent, solidity
            ] + hu + [clase]
            
            resultados.append(features)
            
        except Exception as e:
            print(f"Error procesando {ruta}: {e}")

    if resultados:
        df = pd.DataFrame(resultados)
        df.to_csv("dataset.CSV", mode='a', index=False, header=False)

def entrenar_svm(ruta_dataset):
    print("\n--- Iniciando Entrenamiento Avanzado ---")
    if not os.path.exists(ruta_dataset): return

    data = pd.read_csv(ruta_dataset, header=None)
    data = data.dropna()
    
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    print(f"Dimensiones del dataset: {X.shape}") # Deberías ver más columnas ahora

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # MinMaxScaler a veces va mejor para características geométricas que StandardScaler
    # Pero StandardScaler es estándar para SVM. Probemos Standard.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Grid Search agresivo
    param_grid = {
        'C': [1, 10, 100, 1000], # Probamos valores más altos de regularización
        'gamma': ['scale', 0.1, 0.01, 0.001],
        'kernel': ['rbf'] # RBF es el rey casi siempre en estos datos
    }

    grid = GridSearchCV(SVC(class_weight='balanced'), param_grid, refit=True, verbose=1, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"\nMejores parámetros: {grid.best_params_}")
    print(f"Accuracy en Validación: {grid.best_score_:.4f}")

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("\n--- Reporte Final ---")
    print(classification_report(y_test, y_pred))
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Exactitud Final: {accuracy_score(y_test, y_pred)*100:.2f}%")

def main(): 
    if os.path.exists("dataset.CSV"): os.remove("dataset.CSV")
    if GUARDAR_DEBUG: limpiar_carpeta_debug()

    procesamiento('A/*.JPG','A')
    procesamiento('B/*.JPG','B')
    procesamiento('C/*.JPG','C')
    procesamiento('D/*.JPG','D')
    
    if os.path.exists("dataset.CSV"):
        entrenar_svm("dataset.CSV")

if __name__ == '__main__':
    main()