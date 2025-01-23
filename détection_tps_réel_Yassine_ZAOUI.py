import sys
import subprocess
# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install','-r', 'requirements.txt'])



import cv2  # OpenCV ---> la capture vidéo, le traitement d'images et l'affichage
import numpy as np  # NumPy ---> les manipulations de matrices
import mediapipe as mp  # MediaPipe ---> ajuster le format des images à l'entrée du modèle
from mediapipe.tasks import python  # Importation des options de base 
from mediapipe.tasks.python import vision  # Module vision ---> détection des objets
import time  # Mesurer la latence du traitement en temps réel


###################################### Fonction pour visualiser les détections d'objets sur l'image capturée ######################################################
def visualize(image, detection_result, target_classes):
    """
    Dessine les cadres de détection et affiche les informations sur l'image.

    Args:
        image (np.ndarray): Image sur laquelle dessiner les détections.
        detection_result: Résultats de la détection fournis par MediaPipe.
        target_classes (list): Liste des classes d'objets à détecter (ex. "cup", "cell phone").

    Returns:
        np.ndarray: Image annotée avec les cadres des objets détectés.
    """
    # Boucler sur chaque détection identifiée dans l'image
    for detection in detection_result.detections:
        # Extraire le label et le score de probabilité
        label = detection.categories[0].category_name
        score = detection.categories[0].score
        
        # Si le label n'est pas dans la liste des objets ciblés, on passe
        if label not in target_classes:
            continue

        # Récupérer les coordonnées du bounding box
        bbox = detection.bounding_box
        start_point = (int(bbox.origin_x), int(bbox.origin_y))  # Point supérieur gauche
        end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))  # Point inférieur droit

        # Dessiner un rectangle autour de l'objet détecté en bleu (couleur (255, 0, 0))
        cv2.rectangle(image, start_point, end_point, (255,0,0), 2) 

        # Afficher le label et le score de probabilité au-dessus du cadre
        cv2.putText(
            image, 
            f"{label}: {score:.2f}",  
            (start_point[0], start_point[1] - 10),  # Position du texte juste au-dessus du cadre
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,  # Police et taille du texte
            (255,0,0), 2  # Couleur du texte (bleu) et épaisseur
        )


############################################# Fonction pour charger le modèle de détection d'objets MediaPipe #########################################################
def load_model(model_path):
    """
    Charge un modèle de détection d'objets MediaPipe à partir d'un fichier .tflite. ( dans notre cas le efficientDetlite0)

    Args:
        model_path (str): Chemin du fichier modèle (.tflite).

    Returns:
        ObjectDetector: Modèle de détection d'objets prêt à être utilisé.
    """
    # Définir les options de base avec le chemin du modèle
    base_options = python.BaseOptions(model_asset_path=model_path)
    
    # Configurer l'ObjectDetector avec un seuil de confiance minimum pour les détections (0.5)
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
    
    # Créer l'instance du modèle de détection d'objets
    return vision.ObjectDetector.create_from_options(options)

#################################################################Code principal#########################################################################################

# Initialisation du modèle pré-entrainé et quantifié en int8
model_path = "efficientdet_lite0.tflite"
#model_path="efficientdet_lite2.tflite"
#model_path="ssd_mobilenet_v2.tflite"
model = load_model(model_path)  # récupérer l'instance de détection associé au model en question

# Initialisation de la capture vidéo depuis la caméra du pc
cap = cv2.VideoCapture(0)  # Utiliser l'ID de la caméra par défaut (0)
# Configuration pour enregistrer la vidéo de sortie 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Vidéo en  format MP4
vid = cv2.VideoWriter('output_detection.mp4', fourcc, 20.0, (640, 480))  # Initialiser l'enregistrement vidéo avec 20 fps et de dimensions 640x480

# Définir le petit ensemble d'objets à détecter
target_classes = ["cup", "cell phone", "bottle","fork","spoon"]

# Boucle principale pour la capture vidéo et le traitement en temps réel
while cap.isOpened():
    ret, frame = cap.read() # Lire un frame de la webcam
    if not ret:
        break  # Si la capture échoue, quitter la boucle

    # Convertir l'image capturée de BGR à RGB pour le traitement avec MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Créer une image MediaPipe à partir du frame capturé
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Mesurer le temps de détection pour évaluer la latence
    detection_start_time = time.time()
    detection_result = model.detect(mp_image)  # Effectuer la détection d'objets
    detection_latency = (time.time() - detection_start_time) * 1000  # Calculer la latence en millisecondes

    # Annoter l'image avec les détections d'objets
    visualize(frame, detection_result, target_classes)

    # Afficher la latence sur l'image annotée
    cv2.putText(
       frame, 
        f"Latency: {detection_latency:.2f} ms",  # Afficher le temps de latence
        (10, 30),  # Position du texte
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,  # Police et taille du texte
        (0, 0, 255), 2  # Couleur du texte (rouge) et épaisseur
    )

    # Afficher le flux vidéo avec les détections en temps réel dans une fenêtre
    cv2.imshow("EfficientDet Lite0 Detection", frame)

    # Enregistrer l'image annotée dans le fichier vidéo de sortie
    vid.write(frame)

    # Attendre 1 ms pour vérifier si l'utilisateur appuie sur 'q' pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources après la capture vidéo
cap.release()  # Libérer la caméra
vid.release()  # Libérer l'enregistrement vidéo
cv2.destroyAllWindows()  # Fermer toutes les fenêtres ouvertes
