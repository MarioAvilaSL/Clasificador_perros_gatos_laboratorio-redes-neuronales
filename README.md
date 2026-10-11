# Clasificación de imágenes de perros y gatos con redes neuronales

Comparación de tres arquitecturas (red densa, CNN y CNN con Dropout), cada una entrenada **sin y con aumento de datos**, para clasificar imágenes de perros y gatos del conjunto `cats_vs_dogs`. El proyecto incluye un notebook reproducible que entrena los modelos y genera automáticamente las tablas, figuras y métricas del informe.

## Resultados principales

Evaluación en 3.490 imágenes de prueba que los modelos no vieron durante el entrenamiento (una corrida por modelo, semilla 42).

| Modelo | Accuracy de prueba (IC 95 %) | F1 macro | AUC |
|---|---|---|---|
| **CNN + aumento de datos** | **87,7 % (±1,1)** | 0,877 | 0,953 |
| CNN2 (Dropout 0,5) + aumento de datos | 85,8 % (±1,2) | 0,858 | 0,935 |
| CNN2 (Dropout 0,5) | 84,7 % (±1,2) | 0,847 | 0,923 |
| CNN | 80,2 % (±1,3) | 0,801 | 0,885 |
| Denso + aumento de datos | 56,2 % (±1,6) | 0,562 | 0,577 |
| Denso | 49,8 % (±1,7) | 0,338 | 0,503 |

Conclusiones:

- Las convoluciones fueron decisivas: la red densa quedó cerca del azar.
- Sin aumento de datos las CNN sobreajustaron (99,5 % o más de accuracy de entrenamiento frente a 79 % y 84 % en validación).
- El aumento de datos fue la técnica más eficaz (+7,5 puntos en la CNN), a costa de unas 3 veces más tiempo de entrenamiento.
- El Dropout ayudó sin aumento de datos (+4,5 puntos), pero no con aumento de datos.
- Las diferencias se contrastaron con la prueba de McNemar; no se midió la variación entre semillas.

## Modelos

| Modelo | Arquitectura | Parámetros |
|---|---|---|
| Denso | Aplanado (10.000) → Dense 150 → Dense 150 → Dense 1 (sigmoide) | 1.522.951 |
| CNN | 3 × (Conv 3x3 + MaxPool 2x2) con 32, 64 y 128 filtros → Aplanado → Dense 100 → Dense 1 | 1.372.873 |
| CNN2 | Igual que la CNN, con Dropout 0,5 antes del aplanado y Dense 250 | 3.293.173 |

## Datos y configuración

- **Datos:** [TensorFlow Datasets `cats_vs_dogs`](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs), 23.262 imágenes.
- **Preprocesamiento:** redimensionado a 100x100, escala de grises, normalización a 0-1.
- **División:** 70 % entrenamiento (16.283), 15 % validación (3.489) y 15 % prueba (3.490), con semilla fija 42.
- **Entrenamiento:** Adam (lr 0,001), entropía cruzada binaria, lote de 32, 100 épocas, sin parada anticipada.
- **Aumento de datos** (solo en entrenamiento): giros de hasta 30°, desplazamientos de hasta 20 %, cizalla de 15°, zoom 0,7–1,4 y volteos horizontal y vertical.

## Estructura del repositorio

```
.
├── Clasificador_perros_gatos_laboratorio.ipynb   # entrena, evalúa y genera el informe
├── informe/                                      # se crea al ejecutar el notebook
│   ├── figuras/          # gráficos en PNG
│   ├── tablas/           # tablas en CSV y Excel
│   ├── resultados/       # resultados de cada modelo (caché .pkl)
│   └── RESUMEN_PARA_CLAUDE.md
├── informe.zip                                   # todo lo anterior comprimido
└── README.md
```

## Cómo ejecutarlo

1. Abre el notebook en [Google Colab](https://colab.research.google.com/) y activa una GPU (*Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU*).
2. Si falla la descarga del dataset por `importlib_resources`, ejecuta primero:
   ```python
   !pip install importlib_resources
   ```
3. Ejecuta todas las celdas. Con la configuración por defecto se entrenan los seis modelos (unas 2 horas en la GPU de Colab).
4. Al terminar se genera `informe.zip` con figuras, tablas y métricas.

Los resultados de cada modelo se guardan en caché: si se interrumpe la ejecución, al repetirla se reutilizan los modelos ya entrenados.

### Configurar más experimentos

Los experimentos se definen en la lista `EXPERIMENTOS` del notebook. Cada uno parte de la configuración por defecto (`DEFAULTS`) y solo cambia lo que se indique, por ejemplo:

```python
dict(grupo='General', tipo='cnn', filtros=(32, 64, 128), neuronas=(100,),
     dropout=0.5, aumento='original', epochs=100)
```

Parámetros disponibles: tipo de red, resolución, color, filtros, neuronas, Dropout, BatchNormalization, optimizador, tasa de aprendizaje, tamaño de lote, épocas, aumento de datos y parada anticipada.

## Limitaciones

- Una sola corrida por modelo: no se midió la variación entre semillas.
- Imágenes de 100x100 en escala de grises y con la proporción deformada.
- Se evalúa con los pesos de la última época, sin parada anticipada.
- Los modelos con aumento de datos seguían mejorando en la época 100.
- La CNN2 cambia dos cosas a la vez respecto a la CNN (Dropout y tamaño de la capa densa).
- Hiperparámetros sin ajustar y volteo vertical en el aumento de datos.

## Tecnologías

Python, TensorFlow / Keras 2.21, TensorFlow Datasets, NumPy, OpenCV, scikit-learn, Matplotlib y pandas.

## Autor

Proyecto académico, Universidad Técnica de Machala.
