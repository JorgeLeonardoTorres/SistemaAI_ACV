import os, torch, torchvision, pydicom
import numpy as np
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomRotation, ColorJitter
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ----------------------------------- *** ---------------------------------------
#                                   FUNCIÓN 1: 

# Configuración de semillas para reproducibilidad
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------------- *** ---------------------------------------
#                                   FUNCIÓN 2: 

# Función para calcular Intersection over Union (IoU)
def calculate_iou(boxes1, boxes2):
    """
    Calcula el Intersection over Union (IoU) entre dos conjuntos de cajas delimitadoras.
    """
    inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area

# ----------------------------------- *** ---------------------------------------
#                                   FUNCIÓN 3: 

# Clase para Dataset de imágenes médicas
class BrainStrokeValidationDatasetAdjusted(Dataset):
    def __init__(self, images, transform, scale_factor=1.0):
        self.images = images
        self.transform = transform
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        dicom_data = pydicom.dcmread(img_path)
        image = dicom_data.pixel_array

        # Convertir la imagen a RGB y aplicar transformaciones
        image = Image.fromarray(image).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Etiquetas
        if "hemorrhagic" in img_path.lower():
            label = 1  # Hemorrhagic
        elif "ischaemic" in img_path.lower():
            label = 2  # Ischaemic
        else:
            raise ValueError(f"No se pudo determinar la etiqueta para el archivo: {img_path}")
        
        # Escalar las cajas como en el entrenamiento
        scaled_box = [
            int(30 * self.scale_factor),  # Escalar coordenada x1
            int(30 * self.scale_factor),  # Escalar coordenada y1
            int(650 * self.scale_factor), # Escalar coordenada x2
            int(650 * self.scale_factor)  # Escalar coordenada y2
        ]

        # Escalar las cajas
        target = {
            "labels": torch.tensor([label], dtype=torch.int64),
            "boxes": torch.tensor([scaled_box], dtype=torch.float32)  # Caja ajustada
        }

        return image, target

# ----------------------------------- *** ---------------------------------------
#                                   FUNCIÓN 4: 

# Collate function para DataLoader
def custom_collate_fn(batch):
    return tuple(zip(*batch))

# ----------------------------------- *** ---------------------------------------
#                                   FUNCIÓN 5: 

# Función para calcular media y desviación estándar
def calculate_mean_std(image_paths):
    all_pixels = []
    for path in tqdm(image_paths, desc="Calculando mean y std"):
        dicom_data = pydicom.dcmread(path)
        image = dicom_data.pixel_array.flatten()
        all_pixels.extend(image)

    all_pixels = np.array(all_pixels, dtype=np.float32)
    mean = np.mean(all_pixels) / 255.0  # Escalar a [0, 1]
    std = np.std(all_pixels) / 255.0
    return mean, std

# ----------------------------------- *** ---------------------------------------
#                                   FUNCIÓN 6: 

# Transformaciones ajustadas para entrenamiento
def get_transform_training(mean, std):
    return Compose([
        Resize((700, 700)),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=15),  # Aumenta el rango de rotaciones
        ColorJitter(brightness=0.2, contrast=0.2),  # Aumenta el rango de ajustes
        ToTensor(),
        Normalize(mean=[mean], std=[std])
    ])


# Transformaciones ajustadas para validación
def get_transform_validation(mean, std):
    return Compose([
        Resize((700, 700)),
        ToTensor(),
        Normalize(mean=[mean], std=[std])
    ])


# ----------------------------------- *** ---------------------------------------
#                                   FUNCIÓN 7: 

# Crear backbone EfficientNet-B0
def create_model(backbone_type="efficientnet"):
    if backbone_type == "efficientnet":
        weights = EfficientNet_B0_Weights.DEFAULT
        backbone = efficientnet_b0(weights=weights).features
        backbone = torch.nn.Sequential(
            backbone,
            torch.nn.Dropout(p=0.5)  # Regularización adicional
        )
        backbone.out_channels = 1280  # Último canal de EfficientNet_B0
    else:
        raise NotImplementedError("Solo se soporta EfficientNet-B0 por ahora.")
    return backbone

# ----------------------------------- *** ---------------------------------------
#                                   FUNCIÓN 8: 

# Crear modelo Faster R-CNN
def create_faster_rcnn(backbone, anchor_sizes, num_classes=3):
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=((0.5, 1.0, 2.0),) * len(anchor_sizes)
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0"], output_size=7, sampling_ratio=2
    )
    model = FasterRCNN(
        backbone, num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    return model

# ----------------------------------- *** ---------------------------------------
#                                   FUNCIÓN 9: 

# Calcular tamaños de anchors
def calculate_anchor_sizes(image_paths):
    all_boxes = []
    for path in tqdm(image_paths, desc="Calculando tamaño de anchors"):
        dicom_data = pydicom.dcmread(path)
        image = dicom_data.pixel_array
        height, width = image.shape
        boxes = [[0.1 * width, 0.1 * height, 0.9 * width, 0.9 * height]]
        for box in boxes:
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            avg_size = (box_width + box_height) / 2
            all_boxes.append(avg_size)

    all_boxes = np.array(all_boxes, dtype=np.float32)
    n5 = int(np.percentile(all_boxes, 95))
    anchor_sizes = ((24, 60, 130, 260, n5),)
    return anchor_sizes


# ----------------------------------- *** ---------------------------------------
#                                   FUNCIÓN 10:
# Función para calcular métricas
def calculate_metrics(true_labels, pred_labels, pred_scores=None, average="binary"):
    """
    Calcula métricas clave utilizando scikit-learn.
    Args:
        true_labels (list or array): Etiquetas reales.
        pred_labels (list or array): Etiquetas predichas.
        pred_scores (list or array, optional): Puntajes de predicción (para mAP).
        average (str): Tipo de promedio ('binary', 'micro', 'macro', 'weighted').
    
    Returns:
        dict: Diccionario con las métricas calculadas.
    """
    metrics = {}

    # Precisión, Recall y F1-Score
    metrics["Precision"] = precision_score(true_labels, pred_labels, average=average, zero_division=0)
    metrics["Recall"] = recall_score(true_labels, pred_labels, average=average, zero_division=0)
    metrics["F1-Score"] = f1_score(true_labels, pred_labels, average=average, zero_division=0)

    # Specificity (Especificidad)
    tn = sum((t == 0 and p == 0) for t, p in zip(true_labels, pred_labels))
    fp = sum((t == 0 and p == 1) for t, p in zip(true_labels, pred_labels))
    metrics["Specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0

    # False Positive Rate (FPR)
    metrics["FPR"] = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Mean Average Precision (mAP)
    if pred_scores is not None:
        metrics["mAP"] = average_precision_score(true_labels, pred_scores, average="macro")
    else:
        metrics["mAP"] = None  # Opcional si no hay puntajes

    return metrics

# ----------------------------------- *** ---------------------------------------
#                                   FUNCIÓN 11:
# Función para traducir los valores categóricos a texto
def map_category(value, mapping):
    return mapping.get(int(value), "Valor no categorizado")

# Mapas para traducción de categorías
stroke_map = {0: "se descarta ACV", 1: "diagnostico medico de ACV"}
gender_map = {1: "masculino", 2: "femenino"}
age_map = {1: "joven", 2: "adulto", 3: "persona mayor"}
# race_map = {1: "latino", 2: "caucasico", 3: "negro no hispanico", 4: "blanco no hispanico", 5: "raza no especificada"}
marital_map = {1: "casado", 2: "soltero", 3: "divorciado/separado", 4: "viudo", 5: "en union libre", 6: "sin informacion de estado marital"}
# general_health_map = {
    # 1: "aparente buen estado general de salud",
    # 2: "aceptable estado general de salud",
    # 3: "regular estado general de salud",
    # 4: "mal estado general de salud",
    # 5: "delicado estado general de salud"
# }
diabetes_map = {0: "sin antecedentes de diabetes", 1: "con antecedentes de diabetes"}
hypertension_map = {0: "sin hipertension arterial", 1: "hipertension arterial"}
cholesterol_map = {0: "sin colesterol alto", 1: "hipercolesterolemia"}
smoke_map = {0: "no fuma", 1: "fumador activo"}
alcohol_map = {0: "no consume alcohol", 1: "consumidor de alcohol"}
bmi_map = {1: "bajo", 2: "normal", 3: "con sobrepeso", 4: "con obesidad"}

# ----------------------------------- *** ---------------------------------------
#                                   FUNCIÓN 12:
# Función para validar textos clínicos generados
def validate_text_mapping_enhanced(original_data, clinical_data):
    print("\n# Validación Detallada Mejorada:\n")
    total_mismatches = 0
    total_categories = 0
    record_precisions = []

    for i, row in original_data.iterrows():
        generated_text = clinical_data.iloc[i]['Clinical Text']

        # Categorías según diagnóstico de ACV
        if int(row['stroke']) == 1:  # Paciente con ACV
            categories = {
                "stroke": map_category(row['stroke'], stroke_map),
                "age": map_category(row['age'], age_map),
                "gender": map_category(row['gender'], gender_map),
                # "race": map_category(row['Race'], race_map),
                "marital_status": map_category(row['Marital status'], marital_map),
                # "general_health": map_category(row['General health condition'], general_health_map),
                "hypertension": map_category(row['hypertension'], hypertension_map),
                "systolic_bp": row.get('Systolic blood pressure', "N/A"),
                "diastolic_bp": row.get('Diastolic blood pressure', "N/A"),
                "diabetes": map_category(row['diabetes'], diabetes_map),
                "cholesterol": map_category(row['high cholesterol'], cholesterol_map),
                "smoke": map_category(row['smoke'], smoke_map),
                "alcohol": map_category(row['alcohol'], alcohol_map)
            }
        else:  # Paciente sin ACV
            categories = {
                "stroke": map_category(row['stroke'], stroke_map),
                "age": map_category(row['age'], age_map),
                "gender": map_category(row['gender'], gender_map),
                # "race": map_category(row['Race'], race_map),
                "marital_status": map_category(row['Marital status'], marital_map),
                # "general_health": map_category(row['General health condition'], general_health_map)
            }

        # Validar la presencia de cada categoría en el texto generado
        mismatches = [key for key, value in categories.items() if str(value) not in generated_text]

        # Calcular precisión por registro
        num_categories = len(categories)
        correct_categories = num_categories - len(mismatches)
        precision = (correct_categories / num_categories) * 100
        record_precisions.append(precision)

        # Reportar inconsistencias
        if mismatches:
            total_mismatches += len(mismatches)
            print(f"- Registro {i+1}: Precisión: {precision:.2f}%")
            print(f"Texto generado: {generated_text}")
            print(f"Inconsistencias detectadas: {', '.join(mismatches)}\n")

        total_categories += num_categories

    # Resumen global
    overall_precision = (sum(record_precisions) / len(record_precisions)) if record_precisions else 0
    print("\n# Resumen Global de Validación:\n")
    print(f"- Precisión promedio: {overall_precision:.2f}%")
    print(f"- Total de categorías evaluadas: {total_categories}")
    print(f"- Total de fallos: {total_mismatches}")
    print("\n✔️ Validación completada.\n")

# ----------------------------------- *** ---------------------------------------
#                                   FUNCIÓN 13:
# Función para crear DataLoaders a partir de datasets de PyTorch
def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=16):
    """
    Crea DataLoaders para conjuntos de datos de entrenamiento, validación y prueba.

    Args:
        train_dataset (Dataset): Dataset de entrenamiento.
        val_dataset (Dataset): Dataset de validación.
        test_dataset (Dataset): Dataset de prueba.
        batch_size (int): Tamaño del batch para los DataLoaders.

    Returns:
        tuple: DataLoaders para entrenamiento, validación y prueba.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# ----------------------------------- *** ---------------------------------------
#                                   FUNCIÓN 14:
# Función para crear DataLoaders a partir de datasets de PyTorch
# Crear clase personalizada para los datasets
class ClinicalTextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

# ----------------------------------- *** ---------------------------------------
#                                   FUNCIÓN 15:
# Funciones y clases para entrenamiento y evaluación
class ClinicalTextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for batch in loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), accuracy

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
    return total_loss / len(loader), accuracy, precision, recall, f1
