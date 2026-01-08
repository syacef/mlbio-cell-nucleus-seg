import os
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.models import Model

# Fix CUDA libdevice error by disabling XLA JIT compilation
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'

# --- CONFIGURATION ---
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = 'stage1_train/' # Chemin vers le dossier dézippé du Data Science Bowl
TEST_PATH = 'stage1_test/'   # Chemin vers le test

# 1. DÉFINITION DU MODÈLE U-NET
def build_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    inputs = Input(input_shape)
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs) # Normalisation

    # Contraction (Encoder)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Expansion (Decoder)
    u6 = UpSampling2D(size=(2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = UpSampling2D(size=(2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = UpSampling2D(size=(2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = UpSampling2D(size=(2, 2))(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    # Output layer (Binary segmentation)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 2. FONCTION DE CHARGEMENT DE DONNÉES (Adaptée au format DS Bowl 2018)
def load_data(path, height, width, channels):
    if not os.path.exists(path):
        print(f"Attention: Le chemin {path} n'existe pas. Créez des données factices pour tester.")
        # Retourne des données aléatoires si le dossier n'existe pas (pour test immédiat)
        return np.random.rand(10, height, width, channels), np.random.randint(0, 2, (10, height, width, 1))

    ids = next(os.walk(path))[1]
    X = np.zeros((len(ids), height, width, channels), dtype=np.uint8)
    Y = np.zeros((len(ids), height, width, 1), dtype=bool)

    print('Chargement des images et masques...')
    for n, id_ in enumerate(ids):
        path_img = path + id_
        img = imread(path_img + '/images/' + id_ + '.png')[:,:,:channels]
        img = resize(img, (height, width), mode='constant', preserve_range=True)
        X[n] = img
        
        mask = np.zeros((height, width, 1), dtype=bool)
        for mask_file in next(os.walk(path_img + '/masks/'))[2]:
            mask_ = imread(path_img + '/masks/' + mask_file)
            mask_ = resize(mask_, (height, width), mode='constant', preserve_range=True)
            mask_ = np.expand_dims(mask_, axis=-1)
            mask = np.maximum(mask, mask_) # Combiner les masques individuels
        Y[n] = mask
    return X, Y

# 3. ANALYSE ET CALCUL DE TAILLE
def analyze_nuclei_size(original_image, predicted_mask, pixel_to_micron_ratio=1.0, threshold=0.5):
    """
    Prend un masque binaire, identifie les noyaux individuels et calcule leur aire.
    """
    # Seuil pour binariser la prédiction (0 à 1 -> 0 ou 1)
    mask_binary = (predicted_mask > threshold).astype(np.uint8)
    
    # Étiquetage des régions connexes (Labeling)
    # Chaque noyau distinct reçoit un ID unique (1, 2, 3...)
    label_img = label(mask_binary)
    regions = regionprops(label_img)
    
    nuclei_data = []
    
    print(f"--- Analyse : {len(regions)} noyaux détectés ---")
    
    for props in regions:
        # Aire en pixels
        area_pixels = props.area
        
        # Conversion en microns carrés (si ratio connu)
        # Formule: Aire physique = Aire pixels * (microns par pixel)²
        area_microns = area_pixels * (pixel_to_micron_ratio ** 2)
        
        # Diamètre équivalent (si le noyau était un cercle parfait)
        diameter = props.equivalent_diameter * pixel_to_micron_ratio
        
        nuclei_data.append({
            'label_id': props.label,
            'area_px': area_pixels,
            'area_sq_microns': round(area_microns, 2),
            'diameter_microns': round(diameter, 2),
            'centroid': props.centroid
        })

    return label_img, nuclei_data

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Charger les données
    # Note: Assurez-vous d'avoir téléchargé le dataset Kaggle dans le dossier 'stage1_train'
    X_train, Y_train = load_data(TRAIN_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    # 2. Construire et entraîner le modèle
    model = build_unet()
    model.summary()
    
    # Checkpoint pour sauvegarder le modèle à chaque epoch
    checkpointer = tf.keras.callbacks.ModelCheckpoint('model_dsbowl_epoch_{epoch:02d}.keras', verbose=1, save_best_only=False)
    
    # Entraînement (réduit à 5 epochs pour la démo, augmentez à 50+ pour la prod)
    results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=5, callbacks=[checkpointer])

    # 3. Prédiction sur une image de test (ici on reprend une image d'entrainement pour l'exemple)
    idx = random.randint(0, len(X_train)-1)
    test_img = X_train[idx]
    test_input = np.expand_dims(test_img, axis=0) # Ajout dimension batch
    
    preds_test = model.predict(test_input, verbose=0)
    pred_mask = preds_test[0, :, :, 0] # Retirer dimension batch et channel

    # 4. Extraction de la taille des noyaux
    # Supposons que 1 pixel = 0.5 microns (à calibrer avec votre microscope)
    PIXEL_CALIBRATION = 0.5 
    labeled_nuclei, stats = analyze_nuclei_size(test_img, pred_mask, pixel_to_micron_ratio=PIXEL_CALIBRATION)

    # 5. Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Image originale
    axes[0].imshow(test_img)
    axes[0].set_title("Biopsie Originale")
    
    # Masque prédit (Probabilités)
    axes[1].imshow(pred_mask, cmap='jet')
    axes[1].set_title("Segmentation U-Net")
    
    # Noyaux identifiés et colorés
    image_label_overlay = label2rgb(labeled_nuclei, image=test_img, bg_label=0)
    axes[2].imshow(image_label_overlay)
    axes[2].set_title(f"Noyaux Identifiés ({len(stats)} détectés)")
    
    plt.tight_layout()
    plt.show()
    plt.save_fig("nuclei_segmentation_result.png")

    # Affichage des stats des 5 premiers noyaux
    print("\nExemple de données extraites (5 premiers noyaux):")
    for nucleus in stats[:5]:
        print(nucleus)
