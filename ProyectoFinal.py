#Importar librerrias
import warnings
import urllib.request
import zipfile
from PIL import Image
import torch
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

#desactivar warnings
warnings.filterwarnings('ignore')

# URL del archivo zip
url = 'https://github.com/RubenDHerrera/Proyecto-Final-Samsung-Innovation-Campus/blob/cfee0136a7f6bb24182c66e56385769ca9a5f5ec/Imagenes/Imagenes.zip?raw=true'
extract_dir = './Imagenes'

# Descargar y extraer el archivo zip
zip_path, _ = urllib.request.urlretrieve(url)
with zipfile.ZipFile(zip_path, "r") as f:
    f.extractall(extract_dir)
folder_path = os.path.join(extract_dir)


# Listar todas las imágenes dentro del directorio extraído
path_imagenes = glob.glob(os.path.join(folder_path, "*.jpg"))
path_imagenes.extend(glob.glob(os.path.join(folder_path, "*.jpeg")))
path_imagenes.extend(glob.glob(os.path.join(folder_path, "*.png")))
print(f'Total imágenes encontradas: {len(path_imagenes)}')

#guardar imagenes
imagenes = []
for path_imagen in path_imagenes:
    imagen = Image.open(path_imagen)
    # Si la imagen es RGBA se pasa a RGB
    if np.array(imagen).shape[2] == 4:
        imagen = np.array(imagen)[:, :, :3]
        imagen = Image.fromarray(imagen)
    imagenes.append(imagen)

# Detectar si se dispone de una GPU compatible con CUDA y configurar el dispositivo a usar
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Imprimir el dispositivo que se está utilizando (GPU o CPU)
print('Running on device: {}'.format(device))

# Crear una instancia del detector de rostros MTCNN con parámetros específicos
mtcnn = MTCNN(
    keep_all=True,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    post_process=False,
    image_size=160,
    device=device
)

# Detección de bounding boxes y landmarks
# ==============================================================================
all_boxes = []
for imagen in imagenes:
    boxes, probs, landmarks = mtcnn.detect(imagen, landmarks=True)
    all_boxes.append((imagen, boxes))

# Cargar la imagen censored
censored = Image.open('/Users/Gaming/Documents/Clases samsung innovation campus/Semana 20/censored.jpg')


# Representación con matplotlib
# ==============================================================================
for idx, (imagen, boxes) in enumerate(all_boxes):
    fig, ax = plt.subplots(figsize=(10, 8))
    # Obtener las dimensiones de la imagen
    width, height = imagen.size

    # Mostrar la imagen en los ejes
    ax.imshow(imagen)

    # Iterar sobre todas las cajas detectadas
    if boxes is not None:
        for box in boxes:
            # Mostrar la imagen censored sobre la cara detectada
            ax.imshow(censored, extent=[box[0], box[2], box[1], box[3]], aspect='auto', cmap='gray', origin='lower')

    # Ajustar los límites de los ejes al tamaño de la imagen
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Invertir el eje y para que el origen esté en la parte superior izquierda
    ax.axis('off')

    # Mostrar la imagen
    plt.show()