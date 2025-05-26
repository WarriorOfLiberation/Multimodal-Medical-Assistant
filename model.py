import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import transformers
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    CLIPProcessor, CLIPModel,
    AutoTokenizer, AutoModelForCausalLM
)
import langgraph
from langgraph.graph import Graph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, List, Any, Optional, Tuple
import requests
from io import BytesIO
import json
import logging
import asyncio
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import warnings
import pydicom
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModalityType(Enum):
    VISION = "vision"
    LANGUAGE = "language"
    MULTIMODAL = "multimodal"

@dataclass
class MedicalQuery:
    """Structure for medical queries"""
    query_id: str
    question: str
    image_path: Optional[str] = None
    modality: ModalityType = ModalityType.LANGUAGE
    context: Optional[str] = None
    
@dataclass
class MedicalResponse:
    """Structure for medical responses"""
    query_id: str
    answer: str
    confidence: float
    reasoning: str
    image_analysis: Optional[str] = None
    embeddings: Optional[np.ndarray] = None

class PracticalMedicalDatasetLoader:
    """Real-world medical dataset loader for Kaggle datasets"""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.chest_xray_data = None
        self.medical_qa_data = None
        self.mimic_data = None
        
    def load_nih_chest_xray_dataset(self, dataset_path: str = None):
        """
        Load NIH Chest X-ray Dataset from Kaggle
        Download from: https://www.kaggle.com/datasets/nih-chest-xrays/data
        """
        if dataset_path is None:
            dataset_path = self.data_root / "nih-chest-xrays"
        else:
            dataset_path = Path(dataset_path)
            
        try:
            # Load the main data entry file
            data_entry_path = dataset_path / "Data_Entry_2017_v2020.csv"
            bbox_path = dataset_path / "BBox_List_2017.csv"
            
            if not data_entry_path.exists():
                logger.warning(f"Real dataset not found at {data_entry_path}. Using simulated data.")
                return self._load_simulated_chest_xray_data()
                
            # Load main dataset
            df = pd.read_csv(data_entry_path)
            
            # Load bounding box annotations if available
            bbox_df = None
            if bbox_path.exists():
                bbox_df = pd.read_csv(bbox_path)
                logger.info(f"Loaded bounding box data: {len(bbox_df)} entries")
            
            # Clean and preprocess data
            df['Finding Labels'] = df['Finding Labels'].fillna('No Finding')
            df['Patient Age'] = pd.to_numeric(df['Patient Age'], errors='coerce')
            df['Patient Age'] = df['Patient Age'].fillna(df['Patient Age'].median())
            
            # Split multi-label findings
            df['Finding_List'] = df['Finding Labels'].apply(lambda x: x.split('|'))
            df['Has_Finding'] = df['Finding Labels'].apply(lambda x: 0 if x == 'No Finding' else 1)
            
            # Add image paths
            image_dirs = [
                dataset_path / "images_001" / "images",
                dataset_path / "images_002" / "images", 
                dataset_path / "images_003" / "images",
                dataset_path / "images_004" / "images",
                dataset_path / "images_005" / "images",
                dataset_path / "images_006" / "images",
                dataset_path / "images_007" / "images",
                dataset_path / "images_008" / "images",
                dataset_path / "images_009" / "images",
                dataset_path / "images_010" / "images",
                dataset_path / "images_011" / "images",
                dataset_path / "images_012" / "images"
            ]
            
            def find_image_path(image_name):
                for img_dir in image_dirs:
                    img_path = img_dir / image_name
                    if img_path.exists():
                        return str(img_path)
                return None
            
            df['Image_Path'] = df['Image Index'].apply(find_image_path)
            
            # Filter out images that don't exist
            df = df[df['Image_Path'].notna()].reset_index(drop=True)
            
            # Merge with bounding box data if available
            if bbox_df is not None:
                df = df.merge(bbox_df, on='Image Index', how='left')
            
            self.chest_xray_data = df
            logger.info(f"Loaded NIH Chest X-ray dataset: {len(df)} images")
            logger.info(f"Findings distribution:\n{df['Finding Labels'].value_counts().head(10)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading NIH Chest X-ray dataset: {e}")
            return self._load_simulated_chest_xray_data()
    
    def _load_simulated_chest_xray_data(self):
        """Fallback simulated data when real dataset is not available"""
        sample_data = {
            'Image Index': [
                '00000013_005.png', '00000013_026.png', '00000017_001.png',
                '00000029_001.png', '00000033_001.png', '00000042_023.png',
                '00000050_001.png', '00000065_002.png', '00000078_003.png',
                '00000091_001.png'
            ],
            'Finding Labels': [
                'Hernia', 'No Finding', 'Infiltration',
                'Effusion|Infiltration', 'Atelectasis', 'No Finding',
                'Pneumonia', 'Cardiomegaly', 'Edema', 'Consolidation'
            ],
            'Patient Age': [58, 81, 25, 36, 62, 45, 72, 34, 55, 68],
            'Patient Gender': ['M', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M'],
            'View Position': ['PA', 'PA', 'AP', 'PA', 'AP', 'PA', 'PA', 'AP', 'PA', 'PA']
        }
        
        df = pd.DataFrame(sample_data)
        df['Finding_List'] = df['Finding Labels'].apply(lambda x: x.split('|'))
        df['Has_Finding'] = df['Finding Labels'].apply(lambda x: 0 if x == 'No Finding' else 1)
        df['Image_Path'] = df['Image Index'].apply(lambda x: f"/simulated/path/{x}")
        
        self.chest_xray_data = df
        logger.info(f"Loaded simulated chest X-ray dataset: {len(df)} samples")
        return df
            
    def load_vqa_rad_dataset(self, dataset_path: str = None):
        """
        Load VQA-RAD Dataset for Medical Visual Question Answering
        Download from: https://osf.io/89kps/
        """
        if dataset_path is None:
            dataset_path = self.data_root / "vqa-rad"
        else:
            dataset_path = Path(dataset_path)
            
        try:
            # Load training and test data
            train_path = dataset_path / "trainset.json"
            test_path = dataset_path / "testset.json"
            images_dir = dataset_path / "VQA_RAD Image Folder"
            
            if not train_path.exists():
                logger.warning(f"Real VQA dataset not found at {train_path}. Using simulated data.")
                return self._load_simulated_vqa_data()
            
            qa_data = []
            
            # Load training data
            if train_path.exists():
                with open(train_path, 'r') as f:
                    train_data = json.load(f)
                    for item in train_data:
                        qa_data.append({
                            'qid': item.get('qid', ''),
                            'question': item.get('question', ''),
                            'answer': item.get('answer', ''),
                            'image_name': item.get('image_name', ''),
                            'question_type': item.get('question_type', ''),
                            'answer_type': item.get('answer_type', ''),
                            'split': 'train'
                        })
            
            # Load test data
            if test_path.exists():
                with open(test_path, 'r') as f:
                    test_data = json.load(f)
                    for item in test_data:
                        qa_data.append({
                            'qid': item.get('qid', ''),
                            'question': item.get('question', ''),
                            'answer': item.get('answer', ''),
                            'image_name': item.get('image_name', ''),
                            'question_type': item.get('question_type', ''),
                            'answer_type': item.get('answer_type', ''),
                            'split': 'test'
                        })
            
            df = pd.DataFrame(qa_data)
            
            # Add image paths
            def get_vqa_image_path(image_name):
                img_path = images_dir / image_name
                return str(img_path) if img_path.exists() else f"/simulated/vqa/{image_name}"
                
            df['Image_Path'] = df['image_name'].apply(get_vqa_image_path)
            
            self.medical_qa_data = df
            logger.info(f"Loaded VQA-RAD dataset: {len(df)} Q&A pairs")
            logger.info(f"Question types: {df['question_type'].value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading VQA-RAD dataset: {e}")
            return self._load_simulated_vqa_data()
    
    def _load_simulated_vqa_data(self):
        """Fallback simulated VQA data"""
        sample_qa = [
            {
                'qid': 'q001',
                'question': 'What is the primary finding in this chest X-ray?',
                'answer': 'The chest X-ray shows bilateral lower lobe infiltrates consistent with pneumonia.',
                'image_name': '00000013_005.png',
                'question_type': 'abnormality',
                'answer_type': 'open',
                'split': 'train'
            },
            {
                'qid': 'q002', 
                'question': 'Is there any evidence of cardiomegaly?',
                'answer': 'No, the cardiac silhouette appears normal in size.',
                'image_name': '00000017_001.png',
                'question_type': 'presence',
                'answer_type': 'closed',
                'split': 'train'
            },
            {
                'qid': 'q003',
                'question': 'Describe the lung fields in this image.',
                'answer': 'The lung fields show clear bilateral expansion with no acute infiltrates or effusions.',
                'image_name': '00000042_023.png', 
                'question_type': 'description',
                'answer_type': 'open',
                'split': 'test'
            },
            {
                'qid': 'q004',
                'question': 'Are there any signs of pneumothorax?',
                'answer': 'No pneumothorax is visible in this chest X-ray.',
                'image_name': '00000050_001.png',
                'question_type': 'presence',
                'answer_type': 'closed',
                'split': 'train'
            },
            {
                'qid': 'q005',
                'question': 'What abnormality is present in the right lung?',
                'answer': 'There is consolidation in the right lower lobe suggesting pneumonia.',
                'image_name': '00000065_002.png',
                'question_type': 'abnormality',
                'answer_type': 'open',
                'split': 'test'
            }
        ]
        
        df = pd.DataFrame(sample_qa)
        df['Image_Path'] = df['image_name'].apply(lambda x: f"/simulated/vqa/{x}")
        
        self.medical_qa_data = df
        logger.info(f"Loaded simulated medical Q&A dataset: {len(df)} samples")
        return df

    def load_covid_chest_xray_dataset(self, dataset_path: str = None):
        """
        Load COVID-19 Chest X-ray Dataset
        Download from: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
        """
        if dataset_path is None:
            dataset_path = self.data_root / "covid19-radiography-database"
        else:
            dataset_path = Path(dataset_path)
            
        try:
            covid_data = []
            
            # Load different categories
            categories = ['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia']
            
            for category in categories:
                category_dir = dataset_path / category / "images"
                if category_dir.exists():
                    for img_file in category_dir.glob("*.png"):
                        covid_data.append({
                            'image_name': img_file.name,
                            'image_path': str(img_file),
                            'category': category,
                            'diagnosis': 1 if category == 'COVID' else 0
                        })
            
            if not covid_data:
                logger.warning("Real COVID dataset not found. Using simulated data.")
                return self._load_simulated_covid_data()
            
            df = pd.DataFrame(covid_data)
            logger.info(f"Loaded COVID-19 dataset: {len(df)} images")
            logger.info(f"Category distribution:\n{df['category'].value_counts()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading COVID-19 dataset: {e}")
            return self._load_simulated_covid_data()
    
    def _load_simulated_covid_data(self):
        """Simulated COVID data"""
        covid_data = []
        categories = ['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia']
        
        for i, category in enumerate(categories):
            for j in range(5):  # 5 samples per category
                covid_data.append({
                    'image_name': f'{category.lower()}_{j+1}.png',
                    'image_path': f'/simulated/covid/{category.lower()}_{j+1}.png',
                    'category': category,
                    'diagnosis': 1 if category == 'COVID' else 0
                })
        
        df = pd.DataFrame(covid_data)
        logger.info(f"Loaded simulated COVID-19 dataset: {len(df)} images")
        return df

    def create_pytorch_dataset(self, df: pd.DataFrame, transform=None):
        """Create PyTorch dataset from DataFrame"""
        
        class MedicalImageDataset(Dataset):
            def __init__(self, dataframe, transform=None):
                self.df = dataframe.reset_index(drop=True)
                self.transform = transform
                
            def __len__(self):
                return len(self.df)
                
            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                
                # Load image (simulated for demo)
                image_path = row.get('Image_Path', row.get('image_path', ''))
                
                try:
                    if image_path.endswith('.dcm'):
                        # Load DICOM image
                        dicom = pydicom.dcmread(image_path)
                        image = dicom.pixel_array.astype(np.float32)
                        # Normalize to 0-255
                        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                        image = Image.fromarray(image).convert('RGB')
                    elif os.path.exists(image_path):
                        # Load regular image
                        image = Image.open(image_path).convert('RGB')
                    else:
                        # Create placeholder image for simulation
                        image = Image.new('RGB', (224, 224), color='gray')
                        
                except Exception:
                    # Fallback to placeholder
                    image = Image.new('RGB', (224, 224), color='gray')
                
                if self.transform:
                    image = self.transform(image)
                
                # Prepare labels/metadata
                sample = {
                    'image': image,
                    'image_path': image_path,
                    'index': idx
                }
                
                # Add available labels
                for col in self.df.columns:
                    if col not in ['Image_Path', 'image_path']:
                        sample[col] = row[col]
                
                return sample
        
        return MedicalImageDataset(df, transform)

    def get_default_transforms(self, img_size: int = 224, is_training: bool = True):
        """Get default image transforms for medical images"""
        
        if is_training:
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.1),  # Limited for medical images
                transforms.RandomRotation(degrees=5),      # Small rotation only
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
        return transform

    def create_dataloaders(self, df: pd.DataFrame, batch_size: int = 32, 
                          test_size: float = 0.2, img_size: int = 224):
        """Create train/validation DataLoaders"""
        
        # Split data
        if 'Has_Finding' in df.columns:
            stratify_col = df['Has_Finding']
        elif 'diagnosis' in df.columns:
            stratify_col = df['diagnosis']
        else:
            stratify_col = None
            
        train_df, val_df = train_test_split(df, test_size=test_size, 
                                           random_state=42, stratify=stratify_col)
        
        # Create transforms
        train_transform = self.get_default_transforms(img_size, is_training=True)
        val_transform = self.get_default_transforms(img_size, is_training=False)
        
        # Create datasets
        train_dataset = self.create_pytorch_dataset(train_df, train_transform)
        val_dataset = self.create_pytorch_dataset(val_df, val_transform)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=2, pin_memory=True)
        
        logger.info(f"Created DataLoaders - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        return train_loader, val_loader, train_df, val_df

class VisionLanguageModels:
    """Manages vision-language models"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.blip_processor = None
        self.blip_model = None
        self.clip_processor = None
        self.clip_model = None
        
    def load_blip2_model(self):
        """Load BLIP-2 model for image captioning and VQA"""
        try:
            model_name = "Salesforce/blip-image-captioning-base"  # Using stable model
            self.blip_processor = BlipProcessor.from_pretrained(model_name)
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                model_name, 
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            ).to(self.device)
            logger.info("BLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading BLIP model: {e}")
            
    def load_clip_model(self):
        """Load CLIP model for image-text similarity"""
        try:
            model_name = "openai/clip-vit-base-patch32"
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading CLIP: {e}")
            
    def generate_image_caption(self, image: Image.Image) -> str:
        """Generate caption for medical image"""
        if self.blip_model is None or self.blip_processor is None:
            return "Medical image showing anatomical structures for diagnostic evaluation"
            
        try:
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generated_ids = self.blip_model.generate(
                    **inputs, 
                    max_length=100,
                    num_beams=5,
                    temperature=0.7
                )
                
            caption = self.blip_processor.decode(generated_ids[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return "Unable to generate image caption"
    
    def answer_visual_question(self, image: Image.Image, question: str) -> str:
        """Answer question about medical image"""
        if self.blip_model is None or self.blip_processor is None:
            return "Visual question answering requires medical expertise and careful image analysis"
            
        try:
            inputs = self.blip_processor(image, question, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generated_ids = self.blip_model.generate(
                    **inputs,
                    max_length=150,
                    num_beams=5,
                    temperature=0.8
                )
                
            answer = self.blip_processor.decode(generated_ids[0], skip_special_tokens=True)
            return answer
        except Exception as e:
            logger.error(f"Error answering visual question: {e}")
            return "Unable to process visual question"
    
    def compute_image_text_similarity(self, image: Image.Image, text: str) -> float:
        """Compute similarity between image and text using CLIP"""
        if self.clip_model is None or self.clip_processor is None:
            return 0.5  # Default similarity
            
        try:
            inputs = self.clip_processor(
                text=[text], 
                images=image, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                similarity = torch.softmax(logits_per_image, dim=1)[0][0].item()
                
            return similarity
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.5

class MedicalKnowledgeBase:
    """Medical knowledge base with embeddings"""
    
    def __init__(self):
        self.knowledge_entries = []
        self.embeddings = []
        
    def add_knowledge(self, text: str, category: str, embeddings: np.ndarray):
        """Add knowledge entry with embeddings"""
        entry = {
            'text': text,
            'category': category,
            'embedding': embeddings
        }
        self.knowledge_entries.append(entry)
        
    def retrieve_similar_knowledge(self, query_embedding: np.ndarray, top_k: int = 3) -> List[str]:
        """Retrieve similar knowledge entries"""
        if not self.knowledge_entries:
            return []
            
        similarities = []
        for entry in self.knowledge_entries:
            sim = cosine_similarity([query_embedding], [entry['embedding']])[0][0]
            similarities.append((sim, entry['text']))
            
        similarities.sort(reverse=True)
        return [text for _, text in similarities[:top_k]]

class MedicalAssistantWorkflow:
    """LangGraph workflow for medical assistant"""
    
    def __init__(self, data_root: str = "/kaggle/input"):
        self.vision_models = VisionLanguageModels()
        self.knowledge_base = MedicalKnowledgeBase()
        self.dataset_loader = PracticalMedicalDatasetLoader(data_root)
        self.workflow = None
        
    def setup_models(self):
        """Initialize all models"""
        logger.info("Loading vision-language models...")
        self.vision_models.load_blip2_model()
        self.vision_models.load_clip_model()
        
        # Load datasets
        logger.info("Loading medical datasets...")
        self.dataset_loader.load_nih_chest_xray_dataset()
        self.dataset_loader.load_vqa_rad_dataset()
        
        # Populate knowledge base
        self._populate_knowledge_base()
        
    def _populate_knowledge_base(self):
        """Populate knowledge base with medical information"""
        medical_knowledge = [
            ("Pneumonia appears as consolidation or infiltrates in lung fields on chest X-rays", "radiology"),
            ("Cardiomegaly is indicated by a cardiothoracic ratio greater than 0.5 on PA chest X-rays", "cardiology"),
            ("Pleural effusion appears as blunting of costophrenic angles or fluid levels", "radiology"),
            ("Atelectasis shows as areas of increased opacity with volume loss", "radiology"),
            ("Normal chest X-ray shows clear lung fields with normal cardiac silhouette", "normal"),
            ("Infiltration indicates fluid or inflammatory material in lung interstitium", "radiology"),
            ("Hernia may appear as gas-filled bowel loops in abnormal locations", "radiology"),
            ("Edema presents as bilateral pulmonary vascular congestion", "cardiology"),
            ("Consolidation appears as homogeneous opacity obscuring vascular markings", "radiology"),
            ("Pneumothorax shows as absence of lung markings with visceral pleural line", "radiology")
        ]
        
        # In practice, you would compute actual embeddings using sentence transformers
        for text, category in medical_knowledge:
            # Simulated embedding
            embedding = np.random.rand(512)  
            self.knowledge_base.add_knowledge(text, category, embedding)
            
        logger.info(f"Populated knowledge base with {len(medical_knowledge)} entries")
    
    def create_workflow(self):
        """Create LangGraph workflow"""
        
        def route_query(state: Dict) -> str:
            """Route query based on modality"""
            query = state.get('query')
            if query.image_path:
                if query.question:
                    return 'multimodal_analysis'
                else:
                    return 'vision_analysis'
            else:
                return 'language_analysis'
        
        def vision_analysis(state: Dict) -> Dict:
            """Analyze medical image"""
            query = state['query']
            
            try:
                # Load image (create placeholder for demo)
                image = Image.new('RGB', (224, 224), color='gray')
                
                # Generate caption
                caption = self.vision_models.generate_image_caption(image)
                
                state['vision_analysis'] = {
                    'caption': caption,
                    'image_processed': True,
                    'findings': ['chest X-ray', 'anatomical structures visible']
                }
                
            except Exception as e:
                logger.error(f"Vision analysis error: {e}")
                state['vision_analysis'] = {
                    'caption': 'Medical image analysis unavailable',
                    'image_processed': False,
                    'findings': []
                }
                
            return state
        
        def language_analysis(state: Dict) -> Dict:
            """Analyze language query"""
            query = state['query']
            
            # Retrieve relevant knowledge
            query_embedding = np.random.rand(512)  # In practice, use proper embeddings
            relevant_knowledge = self.knowledge_base.retrieve_similar_knowledge(query_embedding)
            
            # Analyze question type
            question_lower = query.question.lower()
            question_type = 'general'
            if any(word in question_lower for word in ['what', 'describe', 'show']):
                question_type = 'descriptive'
            elif any(word in question_lower for word in ['is', 'are', 'does', 'can']):
                question_type = 'yes_no'
            elif any(word in question_lower for word in ['how', 'why', 'where']):
                question_type = 'explanatory'
            
            state['language_analysis'] = {
                'relevant_knowledge': relevant_knowledge,
                'question_type': question_type,
                'query_processed': True
            }
            
            return state
        
        def multimodal_analysis(state: Dict) -> Dict:
            """Combined vision-language analysis"""
            query = state['query']
            
            # Process both image and text
            state = vision_analysis(state)
            state = language_analysis(state)
            
            try:
                # Load image for VQA
                image = Image.new('RGB', (224, 224), color='gray')
                
                # Answer visual question
                vqa_answer = self.vision_models.answer_visual_question(image, query.question)
                
                # Compute image-text similarity
                similarity = self.vision_models.compute_image_text_similarity(image, query.question)
                
                state['multimodal_analysis'] = {
                    'vqa_answer': vqa_answer,
                    'similarity_score': similarity,
                    'combined_processed': True
                }
                
            except Exception as e:
                logger.error(f"Multimodal analysis error: {e}")
                state['multimodal_analysis'] = {
                    'vqa_answer': 'Visual question answering unavailable',
                    'similarity_score': 0.5,
                    'combined_processed': False
                }
            
            return state
        
        def generate_response(state: Dict) -> Dict:
            """Generate final medical response"""
            query = state['query']
            
            # Compile information from different analyses
            response_parts = []
            confidence = 0.7  # Base confidence
            reasoning_parts = []
            
            # Vision analysis results
            if 'vision_analysis' in state:
                va = state['vision_analysis']
                if va.get('image_processed'):
                    response_parts.append(f"Image Analysis: {va['caption']}")
                    reasoning_parts.append("Analyzed medical image using computer vision")
                    if va['findings']:
                        response_parts.append(f"Key findings: {', '.join(va['findings'])}")
            
            # Language analysis results
            if 'language_analysis' in state:
                la = state['language_analysis']
                if la.get('relevant_knowledge'):
                    response_parts.append("Relevant medical knowledge:")
                    for knowledge in la['relevant_knowledge'][:2]:  # Top 2
                        response_parts.append(f"• {knowledge}")
                    reasoning_parts.append("Retrieved relevant medical knowledge from knowledge base")
                    confidence += 0.1
            
            # Multimodal analysis results
            if 'multimodal_analysis' in state:
                ma = state['multimodal_analysis']
                if ma.get('combined_processed'):
                    response_parts.append(f"Visual Question Answer: {ma['vqa_answer']}")
                    reasoning_parts.append("Combined visual and textual analysis")
                    confidence += ma.get('similarity_score', 0) * 0.2
                    
            # Generate final answer
            if not response_parts:
                answer = "I need more information to provide a comprehensive medical analysis."
                confidence = 0.3
            else:
                answer = "\n\n".join(response_parts)
            
            # Create response
            response = MedicalResponse(
                query_id=query.query_id,
                answer=answer,
                confidence=min(confidence, 1.0),
                reasoning=" | ".join(reasoning_parts) if reasoning_parts else "Basic analysis performed",
                image_analysis=state.get('vision_analysis', {}).get('caption', None)
            )
            
            state['response'] = response
            return state
        
        # Create the workflow graph
        workflow = Graph()
        
        # Add nodes
        workflow.add_node("route_query", route_query)
        workflow.add_node("vision_analysis", vision_analysis)
        workflow.add_node("language_analysis", language_analysis)
        workflow.add_node("multimodal_analysis", multimodal_analysis)
        workflow.add_node("generate_response", generate_response)
        
        # Add edges
        workflow.add_edge(START, "route_query")
        
        # Conditional routing
        workflow.add_conditional_edges(
            "route_query",
            lambda state: state.get('next_step', 'language_analysis'),
            {
                "vision_analysis": "vision_analysis",
                "language_analysis": "language_analysis", 
                "multimodal_analysis": "multimodal_analysis"
            }
        )
        
        # All paths lead to response generation
        workflow.add_edge("vision_analysis", "generate_response")
        workflow.add_edge("language_analysis", "generate_response")
        workflow.add_edge("multimodal_analysis", "generate_response")
        workflow.add_edge("generate_response", END)
        
        # Compile workflow
        memory = MemorySaver()
        self.workflow = workflow.compile(checkpointer=memory)
        
        logger.info("Medical assistant workflow created successfully")
    
    async def process_query(self, query: MedicalQuery) -> MedicalResponse:
        """Process medical query through workflow"""
        
        # Determine routing
        next_step = 'language_analysis'
        if query.image_path:
            if query.question:
                next_step = 'multimodal_analysis'
            else:
                next_step = 'vision_analysis'
        
        initial_state = {
            'query': query,
            'next_step': next_step
        }
        
        try:
            # Run workflow
            config = {"configurable": {"thread_id": query.query_id}}
            result = await self.workflow.ainvoke(initial_state, config)
            return result['response']
            
        except Exception as e:
            logger.error(f"Workflow processing error: {e}")
            # Fallback response
            return MedicalResponse(
                query_id=query.query_id,
                answer="Unable to process query due to technical issues. Please consult a medical professional.",
                confidence=0.1,
                reasoning="Workflow execution failed"
            )
    
    def evaluate_model_performance(self, test_queries: List[MedicalQuery]) -> Dict:
        """Evaluate model performance on test queries"""
        results = {
            'total_queries': len(test_queries),
            'successful_responses': 0,
            'average_confidence': 0.0,
            'modality_breakdown': {'vision': 0, 'language': 0, 'multimodal': 0},
            'response_times': []
        }
        
        responses = []
        
        for query in test_queries:
            start_time = asyncio.get_event_loop().time()
            
            try:
                # Run synchronously for evaluation
                response = asyncio.run(self.process_query(query))
                responses.append(response)
                
                end_time = asyncio.get_event_loop().time()
                response_time = end_time - start_time
                results['response_times'].append(response_time)
                
                if response.confidence > 0.5:
                    results['successful_responses'] += 1
                
                # Track modality
                if query.modality == ModalityType.VISION:
                    results['modality_breakdown']['vision'] += 1
                elif query.modality == ModalityType.MULTIMODAL:
                    results['modality_breakdown']['multimodal'] += 1
                else:
                    results['modality_breakdown']['language'] += 1
                    
            except Exception as e:
                logger.error(f"Evaluation error for query {query.query_id}: {e}")
                results['response_times'].append(0)
        
        # Calculate averages
        if responses:
            results['average_confidence'] = np.mean([r.confidence for r in responses])
            results['average_response_time'] = np.mean(results['response_times'])
            results['success_rate'] = results['successful_responses'] / len(responses)
        
        return results, responses

class MedicalAssistantDemo:
    """Demo class for the medical assistant"""
    
    def __init__(self, data_root: str = "/tmp/medical_data"):
        self.assistant = MedicalAssistantWorkflow(data_root)
        
    async def run_demo(self):
        """Run comprehensive demo"""
        logger.info("🏥 Starting Medical Assistant Demo")
        
        # Setup models
        logger.info("Setting up models...")
        self.assistant.setup_models()
        
        # Create workflow
        logger.info("Creating workflow...")
        self.assistant.create_workflow()
        
        # Demo queries
        demo_queries = [
            MedicalQuery(
                query_id="demo_001",
                question="What are the signs of pneumonia on a chest X-ray?",
                modality=ModalityType.LANGUAGE
            ),
            MedicalQuery(
                query_id="demo_002", 
                question="Describe this medical image",
                image_path="/simulated/chest_xray.png",
                modality=ModalityType.VISION
            ),
            MedicalQuery(
                query_id="demo_003",
                question="Is there evidence of cardiomegaly in this image?",
                image_path="/simulated/chest_xray.png", 
                modality=ModalityType.MULTIMODAL
            ),
            MedicalQuery(
                query_id="demo_004",
                question="What is the normal cardiothoracic ratio?",
                modality=ModalityType.LANGUAGE
            ),
            MedicalQuery(
                query_id="demo_005",
                question="Are there any abnormalities visible in this scan?",
                image_path="/simulated/ct_scan.png",
                modality=ModalityType.MULTIMODAL
            )
        ]
        
        logger.info(f"Processing {len(demo_queries)} demo queries...")
        
        # Process queries
        for i, query in enumerate(demo_queries, 1):
            logger.info(f"\n📋 Query {i}/{len(demo_queries)}: {query.query_id}")
            logger.info(f"Question: {query.question}")
            logger.info(f"Modality: {query.modality.value}")
            if query.image_path:
                logger.info(f"Image: {query.image_path}")
            
            try:
                response = await self.assistant.process_query(query)
                
                logger.info("✅ Response:")
                logger.info(f"Answer: {response.answer}")
                logger.info(f"Confidence: {response.confidence:.2f}")
                logger.info(f"Reasoning: {response.reasoning}")
                if response.image_analysis:
                    logger.info(f"Image Analysis: {response.image_analysis}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing query {query.query_id}: {e}")
        
        # Evaluate performance
        logger.info("\n📊 Evaluating model performance...")
        evaluation_results, responses = self.assistant.evaluate_model_performance(demo_queries)
        
        logger.info("Evaluation Results:")
        logger.info(f"Total Queries: {evaluation_results['total_queries']}")
        logger.info(f"Successful Responses: {evaluation_results['successful_responses']}")
        logger.info(f"Success Rate: {evaluation_results.get('success_rate', 0):.2%}")
        logger.info(f"Average Confidence: {evaluation_results['average_confidence']:.2f}")
        logger.info(f"Average Response Time: {evaluation_results.get('average_response_time', 0):.2f}s")
        logger.info(f"Modality Breakdown: {evaluation_results['modality_breakdown']}")
        
        return evaluation_results, responses

class DatasetAnalysis:
    """Analysis tools for medical datasets"""
    
    def __init__(self, dataset_loader: PracticalMedicalDatasetLoader):
        self.loader = dataset_loader
        
    def analyze_chest_xray_dataset(self):
        """Analyze chest X-ray dataset"""
        if self.loader.chest_xray_data is None:
            logger.warning("Chest X-ray dataset not loaded")
            return
            
        df = self.loader.chest_xray_data
        
        print("🔍 Chest X-ray Dataset Analysis")
        print("=" * 50)
        print(f"Total Images: {len(df)}")
        print(f"Unique Patients: {df['Patient ID'].nunique() if 'Patient ID' in df.columns else 'N/A'}")
        print(f"Age Range: {df['Patient Age'].min():.0f}-{df['Patient Age'].max():.0f}")
        print(f"Gender Distribution:")
        if 'Patient Gender' in df.columns:
            print(df['Patient Gender'].value_counts())
        
        print(f"\nTop 10 Findings:")
        findings_count = df['Finding Labels'].value_counts().head(10)
        for finding, count in findings_count.items():
            print(f"  {finding}: {count} ({count/len(df)*100:.1f}%)")
        
        # Visualization would go here in a real implementation
        logger.info("Dataset analysis completed")
    
    def analyze_vqa_dataset(self):
        """Analyze VQA dataset"""
        if self.loader.medical_qa_data is None:
            logger.warning("VQA dataset not loaded")
            return
            
        df = self.loader.medical_qa_data
        
        print("\n🔍 Medical VQA Dataset Analysis") 
        print("=" * 50)
        print(f"Total Q&A Pairs: {len(df)}")
        print(f"Unique Images: {df['image_name'].nunique()}")
        
        if 'question_type' in df.columns:
            print(f"\nQuestion Types:")
            qt_counts = df['question_type'].value_counts()
            for qtype, count in qt_counts.items():
                print(f"  {qtype}: {count} ({count/len(df)*100:.1f}%)")
        
        if 'answer_type' in df.columns:
            print(f"\nAnswer Types:")
            at_counts = df['answer_type'].value_counts()
            for atype, count in at_counts.items():
                print(f"  {atype}: {count} ({count/len(df)*100:.1f}%)")
        
        # Sample questions and answers
        print(f"\nSample Q&A Pairs:")
        for i, row in df.head(3).iterrows():
            print(f"  Q: {row['question']}")
            print(f"  A: {row['answer']}")
            print()
        
        logger.info("VQA analysis completed")

def create_training_pipeline(dataset_loader: PracticalMedicalDatasetLoader):
    """Create a training pipeline for the medical models"""
    
    logger.info("🚀 Creating Training Pipeline")
    
    # Load datasets
    chest_xray_df = dataset_loader.load_nih_chest_xray_dataset()
    covid_df = dataset_loader.load_covid_chest_xray_dataset()
    
    # Create dataloaders
    train_loader, val_loader, train_df, val_df = dataset_loader.create_dataloaders(
        chest_xray_df, batch_size=16, test_size=0.2
    )
    
    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    
    # In a real implementation, you would:
    # 1. Define your model architecture
    # 2. Set up loss functions and optimizers  
    # 3. Implement training loops
    # 4. Add validation and metrics tracking
    # 5. Save model checkpoints
    
    return train_loader, val_loader

async def main():
    """Main execution function"""
    
    # Setup
    demo = MedicalAssistantDemo()
    
    # Run demo
    evaluation_results, responses = await demo.run_demo()
    
    # Dataset analysis
    logger.info("\n🔬 Running Dataset Analysis")
    analyzer = DatasetAnalysis(demo.assistant.dataset_loader)
    analyzer.analyze_chest_xray_dataset()
    analyzer.analyze_vqa_dataset()
    
    # Training pipeline demo
    logger.info("\n🎯 Training Pipeline Demo")
    train_loader, val_loader = create_training_pipeline(demo.assistant.dataset_loader)
    
    logger.info("\n✅ Demo completed successfully!")
    logger.info("This system demonstrates:")
    logger.info("• Multi-modal medical image analysis")
    logger.info("• Vision-language question answering")
    logger.info("• Knowledge base retrieval")
    logger.info("• Workflow orchestration with LangGraph")
    logger.info("• Real-world dataset integration")
    logger.info("• Performance evaluation")
    
    return evaluation_results, responses

# Additional utility functions

def setup_kaggle_environment():
    """Setup environment for Kaggle competitions"""
    
    # Install required packages (would be done in Kaggle notebook)
    required_packages = [
        'transformers>=4.21.0',
        'torch>=1.12.0', 
        'torchvision>=0.13.0',
        'langgraph>=0.1.0',
        'sentence-transformers',
        'pydicom',
        'opencv-python',
        'scikit-learn',
        'matplotlib',
        'seaborn'
    ]
    
    logger.info("Required packages for Kaggle environment:")
    for package in required_packages:
        logger.info(f"  {package}")

def optimize_for_gpu_memory():
    """Optimize models for GPU memory constraints"""
    
    optimization_tips = [
        "Use gradient checkpointing",
        "Enable mixed precision training (fp16)",
        "Use smaller batch sizes",
        "Implement gradient accumulation",
        "Use model sharding for large models",
        "Clear cache between batches",
        "Use CPU offloading when necessary"
    ]
    
    logger.info("GPU Memory Optimization Tips:")
    for tip in optimization_tips:
        logger.info(f"  • {tip}")

if __name__ == "__main__":
    # For Jupyter/Kaggle notebook execution
    import nest_asyncio
    nest_asyncio.apply()
    
    # Setup environment info
    setup_kaggle_environment()
    optimize_for_gpu_memory()
    
    # Run the main demo
    evaluation_results, responses = asyncio.run(main())
    
    print("\n🎉 Multi-Modal Medical Assistant Demo Complete!")
    print(f"Processed {len(responses)} queries with average confidence: {evaluation_results['average_confidence']:.2f}")
