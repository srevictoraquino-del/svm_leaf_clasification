import cv2
import glob
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import time
import numpy as np

def entrenar_svm(ruta_dataset):
    # Cargar datos con pandas
    data = pd.read_csv(ruta_dataset, header=None)
    X = data.iloc[:, :-1].values #Píxeles en el eje X
    y = data.iloc[:, -1].values #Etiqueta o clase

    #Obtiene instancias, píxeles, y número de clases
    #print("Dataset cargado:", X.shape, len(set(y)), "clases")

    #Divide los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

    #Se crea y entrena el modelo SVM
    svm_model = SVC(kernel='rbf', C=1.0, gamma='auto')  # kernel: 'linear', 'rbf', 'poly', 'sigmoid'
    #svm_model = SVC(kernel='linear', C=1.0, gamma=0.01)  # kernel: 'linear', 'rbf', 'poly', 'sigmoid'
    #gamma='scale' o gamma='auto' = nivel de zoom del SVM
    # C = Controla cuánto cuesta cometer un error.
    #C: [0.1, 1, 10, 100],
    #gamma: [0.001, 0.01, 0.1, 1]                  gamma='scale'
    #kernel='linear', C=1.0, gamma=0.01   61.66%     61.66%
    #kernel='rbf', C=1.0, gamma=0.01      38.34%     77.08%
    #kernel='poly', C=1.0, gamma=0.01     67.67%     68.21%
    #kernel='sigmoid', C=1.0, gamma=0.01  38.34%     38.34%
    svm_model.fit(X_train, y_train)

    #Evalua el modelo
    y_pred = svm_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Exactitud del modelo SVM: {acc*100:.2f}%")
    return svm_model

def procesamiento(ruta_origen, clase):
    resultados = []
    for ruta in glob.glob(ruta_origen):
        img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        re_size = cv2.resize(img, (64,64))

        h, w = re_size.shape
        # Superior Izquierda
        q1 = re_size[0:h//2, 0:w//2].mean()
        # Superior Derecha
        q2 = re_size[0:h//2, w//2:w].mean()
        # Inferior Izquierda
        q3 = re_size[h//2:h, 0:w//2].mean()
        # Inferior Derecha
        q4 = re_size[h//2:h, w//2:w].mean() 

        #nueva_media = re_size.mean()
        nueva_mediana = np.median(re_size)
        nueva_moda = calcular_moda_numpy(re_size)
        val_max = re_size.max()
        val_min = re_size.min()
        des_est = re_size.std()

        resultados.append([nueva_mediana,nueva_moda,val_max,val_min,des_est,q1,q2,q3,q4,clase])

    df = pd.DataFrame(resultados)
    df.to_csv("dataset.CSV", mode='a', index = False, header = False)
        
        #print(f"Imagen:{m}")
        #print("Dimensiones:", re_size.shape)

        #cv2.imshow("Imagen", re_size)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

def calcular_moda_numpy(imagen_array):
    # valores: los píxeles únicos encontrados
    # cuentas: cuántas veces aparece cada uno
    valores, cuentas = np.unique(imagen_array, return_counts=True)
    
    # Buscamos el índice del valor que tiene la cuenta más alta
    indice_maximo = np.argmax(cuentas)
    
    return valores[indice_maximo]

# def entrenar_knn(ruta_dataset):
#     # Cargar datos con pandas
#     data = pd.read_csv(ruta_dataset, header=None)
#     X = data.iloc[:, :-1].values   #Píxeles en el eje X
#     y = data.iloc[:, -1].values    #Etiqueta o clase
    
#     #Obtiene instancias, píxeles, y número de clases
#     print("Dataset cargado:", X.shape, len(set(y)), "clases")

#     #Divide los datos en entrenamiento y prueba
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

#     #Se crea y entrena el modelo KNN
#     knn = KNeighborsClassifier(n_neighbors=6)
#     knn.fit(X_train, y_train)

#     #Evalua el modelo
#     y_pred = knn.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     print(f"Exactitud del modelo KNN: {acc*100:.2f}%")
#     return knn

def encontrar_mejor_k(X_train, X_test, y_train, y_test, cov_matrix):
    best_k = 1
    max_acc = 0
    print("Buscando el mejor K (1 a 20)...")
    
    for k in range(1, 21):
        # Entrenamos con un k específico
        knn = KNeighborsClassifier(
            n_neighbors=k, 
            metric='mahalanobis', 
            metric_params={'V': cov_matrix}
        )
        knn.fit(X_train, y_train)
        
        # Evaluamos
        pred = knn.predict(X_test)
        acc = accuracy_score(y_test, pred)
        
        if acc > max_acc:
            max_acc = acc
            best_k = k
            
    return best_k

def entrenar_knn(ruta_dataset):
    # Cargar datos con pandas
    data = pd.read_csv(ruta_dataset, header=None)
    X = data.iloc[:, :-1].values   
    y = data.iloc[:, -1].values    
    
    print("Dataset cargado:", X.shape, len(set(y)), "clases")

    # Divide los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

    # Calcular matriz de covarianza para Mahalanobis
    cov_matrix = np.cov(X_train, rowvar=False)
    cov_matrix += np.eye(cov_matrix.shape[0]) * 0.1

    # Llamada a la función de optimización
    k_optimo = encontrar_mejor_k(X_train, X_test, y_train, y_test, cov_matrix)
    print(f"El mejor valor de K encontrado es: {k_optimo}")

    # Modelo final con el mejor K
    knn = KNeighborsClassifier(
        n_neighbors=k_optimo,
        metric='mahalanobis',
        metric_params={'V': cov_matrix}
    )
    knn.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Exactitud del modelo KNN (Mahalanobis): {acc*100:.2f}%")
    return knn

def main(): 
    start_time = time.perf_counter()
    procesamiento('A/*.JPG','A')
    procesamiento('B/*.JPG','B')
    procesamiento('C/*.JPG','C')
    procesamiento('D/*.JPG','D')
    
    entrenar_svm("dataset.CSV")
    end_time = time.perf_counter()
    elapsed = end_time - start_time

    print(f"Execution took {elapsed:.2f} seconds")

if __name__ == '__main__':
    main()