# turing_rag

## Apartado 1

Para levantar el servicio del apartado 1, es necesario tener instalado la requierement del fichero ./requierement.txt y luego dirigirse a la carpeta src y ejecutar

```bash
cd src
python main.py
```
## Apartado 2

Tarea 1: Diferencias entre 'completion' y 'chat' models

Como en el apartado 1, al guardar los mensajes o las conversaciones, se realiza de la format
{"role": "assistant", "content": "Hola, soy un asistente que te ayudará en tus consultas sobre los PDFs, ¿Cómo te puedo ayudar?"}
Esto se realiza asi por el tipo de modelo chat, en la que entiende entre el tipo de input si es usuario, servidor o asistente. 

prompt: {"role": "assistant", "content": "Tu tarea es identificar el expediente del usuario"}
        {"role": "user", "content": "Quiero consutlar el expediente 312345"}


En cambio los modelos completion, tienen una estructura de input en forma plana, sin tener este tipo de conversacion.

prompt: "Identifica el numero de expediente del siguiente texto..."

Tarea 2: ¿Cómo forzar a que el chatbot responda 'si' o 'no'?¿Cómo parsear la salida para que siga un formato determinado?

Principalmente la formas que conozco son, indicandolo en el prompt, de la forma, "solo responde con un si o no" tambien podemos mostrarle ejemplos
mediante few shots para que vea como debe de devolver la respuesta, podemos añadir, reducir la temperatura para que tenga menos inventiva

Para parsear el output del modelo, podemos expresarlo en el prompt, tecnicas como postprocesado para formatear el output o librerias como jsonformer

Tarea 3: Ventajas e inconvenientes de RAG vs fine-tunning

RAG contamos con los documento fisicos, los que puede consultar el modelo en cualquier momento, y si añadimos un modelo nuevo no haria falta reentrenar, 
al igual que ocurriria con una red siamesa, donde no necesitar reentrenar el modelos con nuevas imagenes, simplemente se añaden. Por otro lado el problema
es que puede estar limitado a la hora de entener el contexto. 

Fine-tunning, al entrenar el modelo, identifica conexiones entre los documentos, entendiendo mejor el contexto de la conversacion, pero es necesario 
entrenar el modelo y no vale como con el RAG que se puede añadir documentos nuevo.

Tarea 4: ¿Cómo evaluar el desempaño de un bot de Q&A? ¿Cómo evaluar el desempeño de un RAG?

Para evaluar tanto el bot como con el RAG, se puede hacer evaluación humana, si la respuesta es la esperada. Para el bot, podria medirse la similitud de la 
respuesta a lo esperado, si es un json, se puede medir numericamente, mientras que con el RAG, seria necesario ver si hace referencia al documento indicado,
si la respuesta es la esperada, si es correcta a lo esperado o si la información es real, de la información recuperada de los documento.

## Apartado 3

Para levantar el apartado 3, es necesario ir a la carpeta detector y ahi ejecutar

```bash
cd detector
docker build -t yolo-detector .
docker run -p 8000:8000 yolo-detector
```

Y como ejemplo se puede ejecutar el archivo prueba.py

```bash
cd detector
python prueba.py
```

## Apartado 4

Pasos necesarios a seguir.
Descripción de posibles problemas que puedan surgir y medidas para reducir el riesgo.
Estimación de cantidad de datos necesarios así como de los resultados, métricas, esperadas.
Enumeración y pequeña descripción (2-3 frases) de técnicas que se pueden utilizar para mejorar el desempeño, las métricas del modelo en tiempo de entrenamiento y las métricas del modelo en tiempo de inferencia.


Para el entrenamiento de un modelo, lo primero de todo seria crear un dataset, o encontralos en kaggle o en algun repositorio ya que es lo mas importante,
si no existiera, necesitarios usar alguna herramiento del estilo labelme para etiquetarla, ya sea por deteccion o clasificacion, una vez tenemos el dataset
deberemos de crear el modelo para entrenar, creando el dataloader, si es necesario data augmentation. Una vez tenemos el modelo, podria ser necesario 
tener tanto preprocesado como postprocesado, si tenemos que realizar division de la imagen para detectarla mejor etc. Una vez entrenado, necesitmos evaluar
el modelo con accuracy, recall, precision etc.

Los problemas, como he dicho antes, data augmentation si no tenemos suficientes imagenes, etiquetar a mano mas datos, si tenemos overfitting, ya que ha aprendido
demasiado bien las imagenes de entrenamiento etc.

Para la cantidad de datos, normalmente cuantas mas mejor y depende de la dificultar del tipo de objeto, no es lo mismo detectar un avion a un coche que detectar
una persona y el tipo de animal. Pero unas 1000 imagenes por clase nueva.

Data augmentation, modificar imagenes que ya tenemos en el dataset, para que el modelo tenga mas ejemplos como rotacion, cambio de gama de colores etc
Tiling, division de la imagen para hacer detecciones sobre trozos de imagen para juntarlas despues en el postprocesado.
No Max Supression, si dos detecciones se solapan mas de un umbral solo seleccionamos una.