import cv2
import numpy as np


## Referencias de configuracion: 
    #https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg
    #https://github.com/AlexeyAB/darknet/blob/master/data/coco.names

#  archivos de configuración, pesos y nombres de clases
config_path = 'yolo/yolov4.cfg'
weights_path = 'yolo/yolov4.weights'
names_path = 'yolo/coco.names'


with open(names_path, 'r') as f:
    classes = f.read().strip().split('\n')

# Configurar la red YOLO
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Obtener las capas de salida de YOLO
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def detect_person(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: La imagen no se pudo cargar.")
        return 0, None

    height, width = image.shape[:2]

    # Crear un blob a partir de la imagen
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Ejecutar la detección
    detections = net.forward(output_layers)

    # Inicializar listas para las detecciones
    boxes = []
    confidences = []
    class_ids = []

    # Procesar cada detección
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filtrar por la clase 'persona' y una confianza mínima
            if class_id == 0 and confidence > 0.5:
                box = detection[:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype('int')

                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar Non-Maxima Suppression (NMS) para eliminar duplicados
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    # Verificar si indices no está vacío y es una lista
    if len(indices) > 0 and isinstance(indices[0], list):
        indices = [i[0] for i in indices]

    # Dibujar los recuadros alrededor de las detecciones
    for i in indices:
        box = boxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f'{classes[class_ids[i]]}: {confidences[i]:.2f}'
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return len(indices), image

# Ejemplo de cómo usar la función
image_path = 'pictures/49.jpeg'
num_detections, processed_image = detect_person(image_path)

if processed_image is not None:
    output_image_path = 'resultados/imagen_procesada8.jpg'
    cv2.imwrite(output_image_path, processed_image)
    print(f"Se detectaron {num_detections} personas.")
else:
    print("No se pudo procesar la imagen.")
