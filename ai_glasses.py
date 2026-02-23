#!/usr/bin/env python3
"""
AI Glasses Pro System v9.0 - Complete Feature Set
Fixed detection accuracy, clean structure, direct audio playback, and all requested features
"""

import cv2
import time
import threading
import queue
import speech_recognition as sr
from ultralytics import YOLO
from gtts import gTTS
import pygame
import os
import numpy as np
from datetime import datetime
import json
import logging
import signal
import sys
from pathlib import Path
import random
from collections import deque
import requests
import base64
from io import BytesIO
import re
import torch
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle
from functools import lru_cache
import tempfile

# Optional imports with error handling
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: Tesseract not available. OCR features will be disabled.")

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("Warning: Face recognition not available. Face features will be disabled.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Some features may be limited.")

try:
    import sklearn
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Color detection may be limited.")

# Production logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ai_glasses_pro.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProAIGlasses:
    """Professional AI Glasses with complete feature set and improved accuracy"""
    
    def __init__(self):
        """Initialize AI Glasses Pro system with visual interface"""
        # System state
        self.running = False
        self.active = False
        self.initialized = False
        
        # Meta glasses interface
        self.display_active = True
        self.overlay_mode = True  # For AR-style display
        
        # Performance metrics
        self.performance_stats = {
            'detection_times': deque(maxlen=100),
            'voice_times': deque(maxlen=50),
            'llm_times': deque(maxlen=30),
            'fps': deque(maxlen=30),
            'uptime': time.time()
        }
        
        # Latency tracking
        self.latency_metrics = {
            'capture': deque(maxlen=100),
            'detection': deque(maxlen=100),
            'voice': deque(maxlen=100),
            'llm': deque(maxlen=50),
            'total_response': deque(maxlen=50)
        }
        
        # Memory and caching
        self.llm_cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.conversation_history = deque(maxlen=10)
        
        # Face recognition
        self.face_encodings = {}
        self.known_faces = {}
        self.face_confidence_threshold = 0.6
        
        # Advanced features
        self.batch_images = []
        self.scene_memory = deque(maxlen=5)
        self.tracking_mode = False
        self.current_focus = None
        
        # Multi-threading
        self.processing_thread_pool = ThreadPoolExecutor(max_workers=4)
        self.voice_queue = queue.Queue(maxsize=3)  # Fixed: only initialize once
        self.detection_queue = queue.Queue()
        
        # Load configuration
        self.config = self.load_config()
        
        # Hardware interfaces
        self.camera = None
        self.model = None
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # LLM features
        self.ocr_enabled = True
        self.description_mode = "detailed"
        self.learning_mode = True
        self.wake_word = self.config['wake_word']
        
        # OCR modes
        self.ocr_mode = "instant"  # instant, scan, batch
        self.scan_active = False
        
        # Performance optimization
        self.frame_buffer = deque(maxlen=2)  # Reduced for lower latency
        self.detection_cache = {}
        self.response_cache = {}
        self.frame_skip = 0
        
        # Performance settings - ULTRA OPTIMIZED FOR 16GB RAM CHIPSET
        self.target_fps = 15  # Reduced from 20 for smoother performance
        self.detection_interval = 15  # Detect every 15 frames to reduce load
        self.confidence_threshold = 0.7  # Higher to reduce false positives and processing
        self.nms_threshold = 0.6  # Higher for faster processing
        self.model_size = "yolov8n"  # Keep nano for speed
        
        # Memory optimization for 16GB RAM
        self.max_detections = 3  # Limit to top 3 detections
        self.frame_skip_threshold = 2  # Skip frames for smoothness
        self.processing_batch_size = 1  # Process one frame at a time
        
        # Model storage and loading for 16GB RAM chipset
        self.model_path = "models/yolov8n_optimized.pt"  # Single optimized model
        self.model_loaded = False
        self.offline_mode = True  # Use offline models only
        
        # Hardware compatibility
        self.camera_available = False
        self.microphone_available = False
        self.audio_available = False
        self.llm_available = False
        
        # Initialize system
        self.setup_directories()
        
        print("🎯 Initialization complete!")
        logger.info("AI Glasses Pro System v9.0 initialized")
        
        # Print system status
        print("🔮 AI Glasses Pro v9.0 - Meta Glasses Ready")
        print("🎤 Voice commands: 'glasses' to activate")
        print("👁️ Visual interface: Active")
        
        # Initialize components
        self.initialize_components()
        
        # Start background threads
        self.start_background_threads()
    
    def load_config(self):
        """Load professional configuration"""
        default_config = {
            'wake_word': 'glasses',
            'llm_provider': 'ollama',
            'model_name': 'llama3.2:1b',
            'ollama_url': 'http://localhost:11434',
            'max_tokens': 300,
            'temperature': 0.5,
            'ocr_enabled': True,
            'description_mode': 'detailed',
            'learning_mode': True,
            'conversation_memory': 5,
            'cache_enabled': True,
            'cache_ttl': 300,
            'detection_interval': 2,
            'confidence_threshold': 0.5,
            'target_fps': 30,
            'model_size': 'yolov8n'
        }
        
        try:
            config_path = Path('ai_glasses_config.json')
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    return {**default_config, **config}
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
        
        return default_config
    
    def setup_directories(self):
        """Setup required directories"""
        directories = ['logs', 'models', 'data', 'exports', 'faces']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def load_known_faces(self):
        """Load known faces for recognition"""
        try:
            faces_file = Path('data/known_faces.json')
            if faces_file.exists():
                with open(faces_file, 'r') as f:
                    self.known_faces = json.load(f)
                logger.info(f"Loaded {len(self.known_faces)} known faces")
        except Exception as e:
            logger.warning(f"Could not load known faces: {e}")
            self.known_faces = {}
    
    def save_known_faces(self):
        """Save known faces"""
        try:
            faces_file = Path('data/known_faces.json')
            with open(faces_file, 'w') as f:
                json.dump(self.known_faces, f, indent=2)
            logger.info("Saved known faces")
        except Exception as e:
            logger.error(f"Could not save known faces: {e}")
    
    def initialize(self):
        """Initialize professional system"""
        try:
            logger.info("Starting professional initialization...")
            
            # Initialize hardware
            self.initialize_camera_pro()
            self.initialize_voice_pro()
            self.initialize_models_pro()
            self.initialize_audio_pro()
            
            # Initialize LLM
            self.initialize_llm()
            
            # Initialize OCR
            self.initialize_ocr()
            
            self.initialized = True
            logger.info("Professional initialization complete")
            
            self.speak("AI Glasses Pro system ready. All features activated.")
            return True
            
        except Exception as e:
            logger.error(f"Professional initialization failed: {e}")
            return False
    
    def setup_visual_interface(self):
        """Setup visual interface for Meta glasses"""
        try:
            print("📹 Creating display window...")
            # Create display window with proper flags
            cv2.namedWindow('AI Glasses Pro - Meta Interface', cv2.WINDOW_AUTOSIZE)
            cv2.resizeWindow('AI Glasses Pro - Meta Interface', 640, 480)
            
            # Move window to visible position
            cv2.moveWindow('AI Glasses Pro - Meta Interface', 100, 100)
            
            # Set window to stay on top
            cv2.setWindowProperty('AI Glasses Pro - Meta Interface', cv2.WND_PROP_TOPMOST, 1)
            
            # Force window to show immediately
            cv2.waitKey(1)
            
            print("✅ Visual interface initialized for Meta glasses")
            return True
        except Exception as e:
            print(f"❌ Visual interface setup failed: {e}")
            logger.error(f"Visual interface setup failed: {e}")
            return False
    
    def initialize_components(self):
        """Initialize all system components"""
        try:
            logger.info("Starting professional initialization...")
            
            # Initialize camera
            self.initialize_camera_pro()
            
            # Initialize voice recognition
            self.initialize_voice()
            
            # Initialize AI model
            self.initialize_model()
            
            # Initialize audio
            self.initialize_audio()
            
            # Initialize LLM
            self.initialize_llm()
            
            # Initialize OCR
            self.initialize_ocr()
            
            self.initialized = True
            logger.info("Professional initialization complete")
            
            self.speak("AI Glasses Pro system ready. All features activated.")
            return True
            
        except Exception as e:
            logger.error(f"Professional initialization failed: {e}")
            return False
    
    def start_background_threads(self):
        """Start background processing threads"""
        # This can be implemented for background processing
        pass
        """Setup visual interface for Meta glasses"""
        try:
            # Create display window
            cv2.namedWindow('AI Glasses Pro - Meta Interface', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('AI Glasses Pro - Meta Interface', 800, 600)
            
            # Set window position for Meta glasses
            cv2.moveWindow('AI Glasses Pro - Meta Interface', 0, 0)
            
            logger.info("Visual interface initialized for Meta glasses")
            return True
        except Exception as e:
            logger.error(f"Visual interface setup failed: {e}")
            return False
    
    def draw_overlay(self, frame, detections, status_text=""):
        """Draw lightweight AR-style overlay optimized for 16GB RAM"""
        if not self.overlay_mode:
            return frame
        
        # Create lightweight overlay
        overlay = frame.copy()
        h, w = overlay.shape[:2]
        
        # Minimal header for 16GB RAM
        cv2.rectangle(overlay, (0, 0), (w, 35), (0, 0, 0), -1)
        
        # Add status text only
        cv2.putText(overlay, f"AI Glasses: {status_text}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw only top 3 detections for 16GB RAM
        for detection in detections[:self.max_detections]:
            x, y, w_box, h_box = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw lightweight bounding box
            cv2.rectangle(overlay, (x, y), (x + w_box, y + h_box), (0, 255, 0), 1)
            
            # Draw label only if confidence is high
            if confidence > 0.8:
                label = f"{class_name}"
                cv2.putText(overlay, label, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        return overlay
    
    def initialize_voice(self):
        """Initialize voice recognition"""
        try:
            self.microphone_available = True
            device_name = getattr(self.microphone, 'device_name', 'Default Microphone')
            logger.info(f"Professional voice recognition using: {device_name}")
            return True
        except Exception as e:
            logger.error(f"Voice initialization failed: {e}")
            self.microphone_available = False
            return False
    
    def initialize_audio(self):
        """Initialize audio system for direct playback"""
        try:
            # Initialize pygame mixer for direct audio playback
            pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=128)
            pygame.mixer.init()
            self.audio_available = True
            logger.info("Professional audio system ready - direct playback mode")
            return True
        except Exception as e:
            logger.warning(f"Audio initialization failed: {e}")
            self.audio_available = False
            return False
    
    def initialize_model(self):
        """Initialize AI model with ULTRA optimization for 16GB RAM chipset"""
        try:
            print("🧠 Loading optimized AI model...")
            
            # Check if optimized model exists
            if os.path.exists(self.model_path):
                print(f"✅ Loading optimized model: {self.model_path}")
                self.model = YOLO(self.model_path)
                self.model_loaded = True
            else:
                print("⚠️ Optimized model not found, creating...")
                # Load base model and optimize it
                base_model = YOLO("yolov8n.pt")
                
                # Optimize model for 16GB RAM
                print("🔧 Optimizing model for 16GB RAM...")
                
                # Create models directory
                os.makedirs("models", exist_ok=True)
                
                # Export optimized model
                base_model.export(format='torchscript', imgsz=[256, 256])  # Smaller input size
                optimized_path = self.model_path
                base_model.save(optimized_path)
                
                print(f"✅ Optimized model saved: {optimized_path}")
                self.model = base_model
                self.model_loaded = True
            
            # Configure for maximum efficiency
            if self.device == 'cuda':
                self.model.to(self.device)
                if self.half_precision:
                    self.model.half()
            
            print(f"✅ Ultra optimized YOLO model ready on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Model initialization failed: {e}")
            logger.error(f"Model initialization failed: {e}")
            return False
    
    def detect_objects(self, frame):
        """Detect objects with ULTRA optimization for 16GB RAM chipset and loaded model"""
        if not self.model or not self.model_loaded or frame is None:
            return []
        
        try:
            start_time = time.time()
            
            # Skip frames for smoothness on 16GB RAM
            self.frame_skip += 1
            if self.frame_skip < self.frame_skip_threshold:
                return []  # Skip detection for smoothness
            
            self.frame_skip = 0  # Reset counter
            
            # Ultra optimized processing for 16GB RAM with loaded model
            processed_frame = frame  # Use frame directly - no preprocessing
            
            # Run detection with minimal settings for maximum speed
            results = self.model(
                processed_frame, 
                verbose=False, 
                imgsz=224,  # Further reduced for 16GB RAM
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                max_det=self.max_detections,  # Limit to 3 detections
                half=self.half_precision,  # Use half precision if available
                device=self.device  # Ensure correct device
            )
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get detection info
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[class_id]
                        
                        # Filter by higher confidence for 16GB RAM
                        if confidence >= self.confidence_threshold:
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                                'confidence': float(confidence),
                                'class_name': class_name,
                                'class_id': class_id
                            })
            
            # Sort by confidence and limit for memory
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Update metrics
            detection_time = time.time() - start_time
            self.latency_metrics['detection'].append(detection_time)
            
            return detections[:self.max_detections]  # Return only top 3 for 16GB RAM
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []
    
    def initialize_camera_pro(self):
        """Initialize camera with ULTRA optimization for Meta glasses chipset"""
        try:
            print("📹 Initializing camera...")
            
            # Try different camera indices
            camera_found = False
            for camera_index in range(3):  # Try camera 0, 1, 2
                print(f"🔄 Trying camera {camera_index}...")
                self.camera = cv2.VideoCapture(camera_index)
                
                if self.camera.isOpened():
                    print(f"✅ Camera {camera_index} opened successfully")
                    
                    # Test camera with multiple attempts
                    print("🔄 Testing camera capture...")
                    for attempt in range(3):
                        ret, test_frame = self.camera.read()
                        if ret and test_frame is not None:
                            print(f"✅ Camera test successful on attempt {attempt + 1}")
                            print(f"📏 Frame size: {test_frame.shape}")
                            camera_found = True
                            break
                        else:
                            print(f"⚠️ Camera test attempt {attempt + 1} failed")
                            time.sleep(0.2)
                    
                    if camera_found:
                        break
                    else:
                        self.camera.release()
                        print(f"❌ Camera {camera_index} failed all tests")
                else:
                    print(f"❌ Camera {camera_index} not available")
            
            if not camera_found:
                print("❌ No working camera found - running in simulation mode")
                self.camera_available = False
                return False
            
            # ULTRA optimized settings for 16GB RAM Meta glasses chipset
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)   # Further reduced for memory
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)  # Further reduced for memory
            self.camera.set(cv2.CAP_PROP_FPS, 15)  # Match target FPS
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus for speed
            
            # Additional chipset optimizations
            self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            self.camera_available = True
            print("✅ Ultra optimized camera initialized at 640x480 for Meta glasses chipset")
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            logger.error(f"Camera initialization failed: {e}")
            self.camera_available = False
            return False
    
    def preprocess_frame(self, frame):
        """Preprocess frame for better detection accuracy"""
        try:
            # Enhance contrast and brightness
            frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=5)
            
            # Apply slight blur to reduce noise
            frame = cv2.GaussianBlur(frame, (3, 3), 0)
            
            return frame
        except Exception as e:
            logger.error(f"Frame preprocessing failed: {e}")
            return frame
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Higher resolution
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.camera.set(cv2.CAP_PROP_EXPOSURE, -6)  # Better exposure
            
            # Test camera with quality check
            ret, frame = self.camera.read()
            if not ret or frame is None:
                logger.warning("Camera test failed - running in simulation mode")
                self.camera_available = False
                return
            
            # Check frame quality
            if frame.shape[0] < 720 or frame.shape[1] < 1280:
                logger.warning("Low resolution camera detected")
            
            self.camera_available = True
            logger.info(f"Professional camera initialized at 1920x1080")
            
        except Exception as e:
            logger.warning(f"Camera initialization failed: {e}")
            self.camera_available = False
    
    def initialize_voice_pro(self):
        """Initialize voice recognition with professional accuracy"""
        try:
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.energy_threshold = 100  # More sensitive
            self.recognizer.pause_threshold = 0.3  # Faster response
            self.recognizer.operation_timeout = None
            self.recognizer.non_speaking_duration = 0.2  # Quicker detection
            
            # Find best microphone
            mics = sr.Microphone.list_microphone_names()
            best_mic = 0
            
            for i, mic_name in enumerate(mics):
                if any(keyword in mic_name.lower() for keyword in ['usb', 'external', 'hdmi', 'blue']):
                    best_mic = i
                    break
            
            self.microphone = sr.Microphone(device_index=best_mic)
            
            try:
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                self.microphone_available = True
                logger.info(f"Professional voice recognition using: {mics[best_mic]}")
            except Exception as e:
                logger.warning(f"Microphone initialization failed: {e}")
                self.microphone_available = False
            
        except Exception as e:
            logger.warning(f"Voice initialization failed: {e}")
            self.microphone_available = False
    
    def initialize_models_pro(self):
        """Initialize AI models with professional accuracy"""
        try:
            # Use appropriate model size based on config
            model_path = f'models/{self.model_size}.pt'
            
            if not Path(model_path).exists():
                logger.info(f"Downloading {self.model_size} model...")
                self.model = YOLO(f'{self.model_size}.pt')
                self.model.save(model_path)
            else:
                self.model = YOLO(model_path)
            
            # Professional model optimization
            self.model.to(self.device)
            if self.half_precision:
                self.model.fuse()
            
            # Configure for maximum accuracy
            self.model.conf = self.confidence_threshold
            self.model.iou = self.nms_threshold
            self.model.max_det = 50  # Increased for better coverage
            
            logger.info(f"Professional YOLO {self.model_size} model on {self.device}")
            
        except Exception as e:
            logger.warning(f"Model initialization failed: {e}")
            self.model = None
    
    def initialize_audio_pro(self):
        """Initialize audio output with direct playback (no cache)"""
        try:
            pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=128)
            pygame.mixer.init()
            self.audio_available = True
            logger.info("Professional audio system ready - direct playback mode")
        except Exception as e:
            logger.warning(f"Audio initialization failed: {e}")
            self.audio_available = False
    
    def initialize_ocr(self):
        """Initialize OCR engine"""
        if not TESSERACT_AVAILABLE:
            logger.warning("OCR engine not available")
            self.ocr_enabled = False
            return False
            
        try:
            # Configure Tesseract for better accuracy
            pytesseract.pytesseract.tesseract_cmd = r'tesseract'
            self.ocr_enabled = True
            logger.info("OCR engine initialized")
            return True
        except Exception as e:
            logger.warning(f"OCR initialization failed: {e}")
            self.ocr_enabled = False
            return False
    
    def initialize_llm(self):
        """Initialize LLM connection"""
        try:
            # Test Ollama connection
            response = requests.get(f"{self.config['ollama_url']}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("LLM connection established")
                self.llm_available = True
                return True
            else:
                logger.warning("LLM connection failed")
                self.llm_available = False
                return False
        except Exception as e:
            logger.warning(f"LLM initialization failed: {e}")
            self.llm_available = False
            return False
    
    def speak(self, text):
        """Direct text-to-speech playback without caching"""
        if not self.audio_available:
            logger.info(f"Audio unavailable: {text}")
            return
        
        try:
            # Create temporary file for TTS
            with tempfile.NamedTemporaryFile(delete=True, suffix='.mp3') as temp_file:
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(temp_file.name)
                
                # Load and play directly
                pygame.mixer.music.load(temp_file.name)
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                # Clean up
                pygame.mixer.music.unload()
                
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
    
    def detect_objects_pro(self, frame):
        """Professional object detection with improved accuracy"""
        if self.model is None:
            return []
        
        try:
            # Preprocess frame for better detection
            processed_frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
            
            # Run detection
            results = self.model(processed_frame, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        if conf > self.confidence_threshold:
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(conf),
                                'class_id': cls,
                                'class_name': self.model.names[cls]
                            })
            
            # Sort by confidence
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            return detections[:20]  # Return top 20 detections
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        if not FACE_RECOGNITION_AVAILABLE:
            return []
            
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            faces = []
            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                faces.append({
                    'bbox': [left, top, right-left, bottom-top],
                    'encoding': encoding
                })
            
            return faces
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    def recognize_faces(self, faces):
        """Recognize known faces"""
        recognized = []
        for face in faces:
            name = "Unknown"
            min_distance = float('inf')
            
            for known_name, known_encoding in self.face_encodings.items():
                distance = face_recognition.face_distance([known_encoding], face['encoding'])[0]
                if distance < 0.6 and distance < min_distance:
                    name = known_name
                    min_distance = distance
            
            recognized.append({
                'bbox': face['bbox'],
                'name': name,
                'confidence': 1 - min_distance if name != "Unknown" else 0
            })
        
        return recognized
    
    def perform_ocr(self, frame, mode="instant"):
        """Perform OCR with different modes"""
        if not self.ocr_enabled or not TESSERACT_AVAILABLE:
            return "OCR not available"
        
        try:
            if mode == "instant":
                # Quick OCR on center region
                h, w = frame.shape[:2]
                center_frame = frame[h//4:3*h//4, w//4:3*w//4]
                text = pytesseract.image_to_string(center_frame, config='--psm 6')
                return text.strip()
            
            elif mode == "scan":
                # Full frame OCR
                text = pytesseract.image_to_string(frame, config='--psm 6')
                return text.strip()
            
            elif mode == "batch":
                # Add to batch for later processing
                self.batch_images.append(frame)
                return "Added to batch"
            
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""
    
    def process_batch_ocr(self):
        """Process batch OCR images"""
        if not self.batch_images or not TESSERACT_AVAILABLE:
            return "No images in batch or OCR not available"
        
        all_text = []
        for i, frame in enumerate(self.batch_images):
            try:
                text = pytesseract.image_to_string(frame, config='--psm 6')
                all_text.append(f"Image {i+1}: {text.strip()}")
            except Exception as e:
                logger.error(f"Batch OCR failed for image {i}: {e}")
                all_text.append(f"Image {i+1}: Error")
        
        self.batch_images.clear()
        return "\n".join(all_text) if all_text else "No text found"
    
    def detect_light_level(self, frame):
        """Detect ambient light level"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            if brightness < 50:
                return "Very dark"
            elif brightness < 100:
                return "Dark"
            elif brightness < 150:
                return "Normal indoor lighting"
            elif brightness < 200:
                return "Bright indoor"
            else:
                return "Very bright"
        except Exception as e:
            logger.error(f"Light detection failed: {e}")
            return "Unknown"
    
    def detect_colors(self, frame):
        """Detect dominant colors in frame"""
        if not SKLEARN_AVAILABLE or not PIL_AVAILABLE:
            return []
            
        try:
            # Resize for faster processing
            small_frame = cv2.resize(frame, (50, 50))
            small_frame = small_frame.reshape((-1, 3))
            
            # K-means clustering for dominant colors
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(small_frame)
            
            colors = []
            for color in kmeans.cluster_centers_:
                b, g, r = color.astype(int)
                colors.append(f"RGB({r},{g},{b})")
            
            return colors
        except Exception as e:
            logger.error(f"Color detection failed: {e}")
            return []
    
    def recognize_cash(self, frame):
        """Recognize cash currency"""
        try:
            # This is a simplified implementation
            # In production, you'd use a specialized currency detection model
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Look for currency-like features (simplified)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            cash_detected = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Filter small contours
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Check for bill-like aspect ratios
                    if 2.0 < aspect_ratio < 3.0:
                        cash_detected.append({
                            'bbox': [x, y, w, h],
                            'type': 'Bill',
                            'confidence': min(0.8, area / 10000)
                        })
            
            return cash_detected
        except Exception as e:
            logger.error(f"Cash recognition failed: {e}")
            return []
    
    def find_people(self, frame):
        """Find and count people"""
        detections = self.detect_objects_pro(frame)
        people = [d for d in detections if d['class_name'] == 'person']
        return people
    
    def find_objects(self, frame, target_object=None):
        """Find specific objects or all objects"""
        detections = self.detect_objects_pro(frame)
        
        if target_object:
            found = [d for d in detections if target_object.lower() in d['class_name']]
            return found
        else:
            return detections
    
    def describe_scene(self, frame):
        """Generate comprehensive scene description"""
        try:
            # Get all detections
            objects = self.detect_objects_pro(frame)
            people = self.find_people(frame)
            faces = self.recognize_faces(self.detect_faces(frame))
            light_level = self.detect_light_level(frame)
            colors = self.detect_colors(frame)
            cash = self.recognize_cash(frame)
            
            # Build description
            description = []
            
            # People and faces
            if people:
                description.append(f"I see {len(people)} people")
            
            known_faces = [f for f in faces if f['name'] != 'Unknown']
            if known_faces:
                names = [f['name'] for f in known_faces]
                description.append(f"Recognized: {', '.join(names)}")
            
            # Objects
            if objects:
                object_counts = {}
                for obj in objects:
                    name = obj['class_name']
                    if name not in object_counts:
                        object_counts[name] = 0
                    object_counts[name] += 1
                
                obj_desc = []
                for name, count in object_counts.items():
                    if count == 1:
                        obj_desc.append(f"a {name}")
                    else:
                        obj_desc.append(f"{count} {name}s")
                
                if obj_desc:
                    description.append(f"Also: {', '.join(obj_desc)}")
            
            # Cash
            if cash:
                description.append(f"I see {len(cash)} bills")
            
            # Lighting
            description.append(f"Lighting is {light_level}")
            
            # Colors
            if colors:
                description.append(f"Dominant colors: {', '.join(colors[:2])}")
            
            return ". ".join(description) + "." if description else "I don't see much in this scene."
            
        except Exception as e:
            logger.error(f"Scene description failed: {e}")
            return "I'm having trouble describing this scene."
    
    def teach_face(self, frame, name):
        """Teach the system a new face"""
        if not FACE_RECOGNITION_AVAILABLE:
            return "Face recognition not available"
            
        try:
            faces = self.detect_faces(frame)
            if not faces:
                return "No face detected to teach"
            
            if len(faces) > 1:
                return "Multiple faces detected. Please show only one face."
            
            face_encoding = faces[0]['encoding']
            self.face_encodings[name] = face_encoding.tolist()
            self.known_faces[name] = {
                'encoding': face_encoding.tolist(),
                'date_added': datetime.now().isoformat()
            }
            
            self.save_known_faces()
            return f"Face taught successfully. Hello {name}!"
            
        except Exception as e:
            logger.error(f"Face teaching failed: {e}")
            return "Failed to teach face"
    
    def query_llm(self, prompt):
        """Query LLM with caching and improved timeout"""
        if not self.llm_available:
            return "LLM not available"
        
        # Check cache first
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        if cache_key in self.llm_cache:
            cached_time, cached_response = self.llm_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                logger.info("Using cached LLM response")
                return cached_response
        
        try:
            start_time = time.time()
            
            # Prepare request
            data = {
                "model": self.config['model_name'],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config['temperature'],
                    "num_predict": self.config['max_tokens']
                }
            }
            
            # Make request with longer timeout
            response = requests.post(
                f"{self.config['ollama_url']}/api/generate",
                json=data,
                timeout=30  # Increased from 10 to 30 seconds
            )
            
            if response.status_code == 200:
                result = response.json()
                llm_response = result.get('response', 'I could not generate a response.')
                
                # Cache the response
                self.llm_cache[cache_key] = (time.time(), llm_response)
                
                # Update metrics
                llm_time = time.time() - start_time
                self.latency_metrics['llm'].append(llm_time)
                
                return llm_response
            else:
                logger.error(f"LLM request failed: {response.status_code}")
                return "I'm having trouble connecting to my brain right now."
                
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return "I'm experiencing some technical difficulties."
    
    def listen(self):
        """Listen for voice commands with improved timeout handling"""
        if not self.microphone:
            return None
            
        try:
            with self.microphone as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen with shorter timeout for better responsiveness
                audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
            
            # Try to recognize speech
            try:
                command = self.recognizer.recognize_google(audio).lower()
                logger.info(f"Voice command: {command}")
                return command
            except sr.UnknownValueError:
                # Speech not understood, but that's normal
                return None
            except sr.RequestError as e:
                logger.error(f"Speech recognition error: {e}")
                return None
                
        except sr.WaitTimeoutError:
            # Timeout is normal - just means no speech detected
            return None
        except Exception as e:
            logger.error(f"Listen failed: {e}")
            return None
    
    def process_command(self, command, frame):
        """Process voice commands with natural language understanding"""
        if not command:
            return
        
        start_time = time.time()
        
        # Wake word detection
        if self.wake_word in command.lower() and not self.active:
            self.active = True
            self.speak("AI Glasses Pro activated. How can I help you?")
            return
        
        if not self.active:
            return
        
        # Natural language command processing
        command_lower = command.lower()
        
        # System control commands
        if any(word in command_lower for word in ["stop", "deactivate", "sleep"]):
            self.active = False
            self.speak("AI Glasses Pro deactivated.")
            return
        
        elif any(word in command_lower for word in ["bye", "exit", "quit", "shutdown"]):
            self.running = False
            self.speak("Goodbye!")
            return
        
        elif "help" in command_lower or "commands" in command_lower or "what can you do" in command_lower:
            help_text = """
            I can help you with:
            - Scene analysis and description
            - Reading text and OCR
            - Finding people and objects
            - Detecting lighting and colors
            - Recognizing currency
            - Face recognition
            - Answering questions
            Just ask naturally!
            """
            self.speak(help_text.strip())
            return
        
        # Natural language scene understanding
        elif any(word in command_lower for word in ["describe", "what do you see", "what's around", "scene", "look around"]):
            description = self.describe_scene(frame)
            self.speak(description)
        
        # Natural language OCR
        elif any(word in command_lower for word in ["read", "text", "ocr", "what does this say", "writing"]) or "scan" in command_lower:
            if "instant" in command_lower or "quick" in command_lower:
                text = self.perform_ocr(frame, "instant")
                response = f"Text: {text[:100]}" if text else "No text detected"
            else:
                text = self.perform_ocr(frame, "scan")
                response = f"Scanned text: {text[:150]}" if text else "No text found"
            self.speak(response)
        
        # Natural language people detection
        elif any(word in command_lower for word in ["people", "person", "anyone", "who", "people", "count people"]):
            people = self.find_people(frame)
            count = len(people)
            self.speak(f"I see {count} person{'s' if count != 1 else ''}")
        
        # Natural language object detection
        elif any(word in command_lower for word in ["objects", "what objects", "find objects", "detect"]):
            objects = self.find_objects(frame)
            if objects:
                names = [obj['class_name'] for obj in objects[:10]]
                self.speak(f"I see: {', '.join(names)}")
            else:
                self.speak("No objects detected")
        
        # Natural language detection
        elif any(word in command_lower for word in ["light", "lighting", "bright", "dark"]):
            light = self.detect_light_level(frame)
            self.speak(f"Lighting is {light}")
        
        # Natural language color detection
        elif any(word in command_lower for word in ["colors", "color", "what colors"]):
            colors = self.detect_colors(frame)
            if colors:
                self.speak(f"Dominant colors: {', '.join(colors[:3])}")
            else:
                self.speak("Could not detect colors")
        
        # Natural language cash detection
        elif any(word in command_lower for word in ["money", "cash", "currency", "dollars", "bill"]):
            cash = self.recognize_cash(frame)
            if cash:
                self.speak(f"I found {len(cash)} bills")
            else:
                self.speak("No cash detected")
        
        # Natural language face recognition
        elif any(word in command_lower for word in ["teach", "learn", "remember face", "who is this"]):
            # Extract name more naturally
            name = None
            if "teach" in command_lower:
                name = command_lower.replace("teach", "").replace("face", "").strip()
            elif "learn" in command_lower:
                name = command_lower.replace("learn", "").strip()
            elif "remember" in command_lower:
                name = command_lower.replace("remember", "").replace("face", "").strip()
            
            if name:
                result = self.teach_face(frame, name)
                self.speak(result)
            else:
                self.speak("Please provide a name after the command")
        
        elif "who do you know" in command_lower or "list faces" in command_lower or "known faces" in command_lower:
            if self.known_faces:
                names = list(self.known_faces.keys())
                self.speak(f"I know: {', '.join(names)}")
            else:
                self.speak("I don't know any faces yet")
        
        # Natural language exploration
        elif any(word in command_lower for word in ["explore", "tell me more", "analyze", "what's happening"]):
            description = self.describe_scene(frame)
            llm_prompt = f"Describe this scene in a helpful way for a visually impaired person: {description}"
            llm_response = self.query_llm(llm_prompt)
            self.speak(llm_response)
        
        # Default to LLM for any other natural language query
        else:
            # Let the LLM handle any other natural request
            llm_response = self.query_llm(command)
            self.speak(llm_response)
        
        # Update metrics
        processing_time = time.time() - start_time
        self.latency_metrics['total_response'].append(processing_time)
    
    def capture_frame(self):
        """Capture frame with ULTRA optimization for 16GB RAM chipset"""
        if not self.camera_available:
            # Create lightweight simulation frame for 16GB RAM
            simulation_frame = np.zeros((360, 480, 3), dtype=np.uint8)  # Reduced size
            simulation_frame[:] = (50, 100, 150)  # Light blue background
            
            # Add minimal simulation text
            cv2.rectangle(simulation_frame, (20, 20), (460, 100), (0, 100, 0), 2)
            cv2.putText(simulation_frame, "16GB RAM OPTIMIZED", 
                        (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(simulation_frame, "AI Glasses Pro", 
                        (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add simple animation
            t = time.time()
            x = int(240 + 60 * np.cos(t))
            y = int(180 + 30 * np.sin(t))
            cv2.circle(simulation_frame, (x, y), 15, (0, 255, 255), -1)
            
            return simulation_frame
        
        try:
            ret, frame = self.camera.read()
            if ret and frame is not None:
                # Quick quality check
                if frame.size > 0:
                    # Ultra optimization for 16GB RAM - ensure correct size
                    h, w = frame.shape[:2]
                    if w != 480 or h != 360:
                        # Fast resize using INTER_NEAREST for speed
                        frame = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_NEAREST)
                        print(f"🔄 Frame resized from {w}x{h} to 480x360")
                    
                    # Add minimal test indicator
                    cv2.rectangle(frame, (5, 5), (80, 30), (0, 255, 0), 1)
                    cv2.putText(frame, "OPTIMIZED", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    # Minimal processing for 16GB RAM
                    return frame
                else:
                    print("❌ Empty frame captured")
                    return None
            else:
                print(f"❌ Failed to capture frame - ret: {ret}")
                return None
        except Exception as e:
            print(f"❌ Frame capture error: {e}")
            logger.error(f"Frame capture error: {e}")
            return None
    
    def run(self):
        """Main execution loop"""
        print("🚀 Starting main execution loop...")
        
        if not self.initialized:
            logger.error("System not initialized")
            print("❌ System not initialized")
            return
        
        self.running = True
        logger.info("AI Glasses Pro system started")
        print("✅ AI Glasses Pro system started")
        
        # Create display window BEFORE camera initialization
        print("📹 Creating display window...")
        cv2.namedWindow('AI Glasses Pro - Meta Interface', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('AI Glasses Pro - Meta Interface', 480, 360)  # Match camera size
        cv2.moveWindow('AI Glasses Pro - Meta Interface', 200, 200)
        cv2.setWindowProperty('AI Glasses Pro - Meta Interface', cv2.WND_PROP_TOPMOST, 1)
        
        # Create lightweight test image for 16GB RAM
        test_image = np.zeros((360, 480, 3), dtype=np.uint8)
        test_image[:] = (50, 100, 150)  # Light blue
        cv2.rectangle(test_image, (30, 30), (450, 80), (0, 100, 0), 2)
        cv2.putText(test_image, "16GB RAM OPTIMIZED", 
                    (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(test_image, "Initializing...", 
                    (40, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show test image immediately
        cv2.imshow('AI Glasses Pro - Meta Interface', test_image)
        cv2.waitKey(1000)  # Wait 1 second
        print("✅ Display window created and visible")
        
        frame_count = 0
        last_voice_time = time.time()
        last_detection_time = 0
        current_detections = []
        status_text = "Ready"
        
        print("🔄 Entering main loop...")
        
        try:
            while self.running:
                start_time = time.time()
                
                # Capture frame
                frame = self.capture_frame()
                if frame is None:
                    print("⚠️ No frame captured, showing test image...")
                    # Show test image instead of blank
                    cv2.imshow('AI Glasses Pro - Meta Interface', test_image)
                    cv2.waitKey(1)
                    time.sleep(0.01)
                    continue
                
                frame_count += 1
                
                # Show first frame with special overlay
                if frame_count == 1:
                    print("📺 Displaying first camera frame...")
                    cv2.putText(frame, "CAMERA ACTIVE!", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    cv2.putText(frame, "You should see your face!", (120, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Perform object detection much less frequently for chipset
                if frame_count % self.detection_interval == 0:
                    current_detections = self.detect_objects(frame)
                    last_detection_time = time.time()
                
                # Listen for voice commands much less frequently for 16GB RAM
                if time.time() - last_voice_time > 3.0:  # Increased to 3.0 seconds
                    command = self.listen()
                    if command:
                        self.process_command(command, frame)
                        last_voice_time = time.time()
                
                # Update status text
                if self.active:
                    status_text = "Active"
                else:
                    status_text = "Standby"
                
                # Lightweight overlay for 16GB RAM
                display_frame = self.draw_overlay(frame, current_detections, status_text)
                
                # Show interface and handle key press
                cv2.imshow('AI Glasses Pro - Meta Interface', display_frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("🛑 Quit requested")
                    self.running = False
                elif key == ord(' '):  # Spacebar to toggle activation
                    self.active = not self.active
                    if self.active:
                        self.speak("Voice commands activated")
                    else:
                        self.speak("Voice commands deactivated")
                
                # Calculate FPS
                frame_time = time.time() - start_time
                self.performance_stats['fps'].append(1.0 / frame_time if frame_time > 0 else 0)
                
                # Optimized frame timing for 16GB RAM
                target_frame_time = 1.0 / self.target_fps
                if frame_time < target_frame_time:
                    sleep_time = target_frame_time - frame_time
                    if sleep_time > 0.01:  # Only sleep if significant
                        time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            print("⛔ Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            print(f"❌ Main loop error: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down AI Glasses Pro...")
        
        self.running = False
        self.active = False
        
        # Cleanup resources
        if self.camera and self.camera_available:
            self.camera.release()
        
        if self.audio_available:
            pygame.mixer.quit()
        
        # Cleanup thread pool if it exists
        if hasattr(self, 'processing_thread_pool'):
            self.processing_thread_pool.shutdown(wait=True)
        
        # Save final stats
        self.performance_stats['uptime'] = time.time() - self.performance_stats['uptime']
        
        logger.info("AI Glasses Pro shutdown complete")
    
    def get_performance_stats(self):
        """Get current performance statistics"""
        stats = self.performance_stats.copy()
        
        # Calculate averages
        for metric in self.latency_metrics:
            if self.latency_metrics[metric]:
                stats[f'avg_{metric}'] = np.mean(self.latency_metrics[metric])
        
        # Cache efficiency
        total_cache_ops = stats['cache_hits'] + stats['cache_misses']
        if total_cache_ops > 0:
            stats['cache_efficiency'] = stats['cache_hits'] / total_cache_ops
        else:
            stats['cache_efficiency'] = 0
        
        return stats

def main():
    """Main entry point"""
    try:
        # Create and run AI Glasses Pro
        ai_glasses = ProAIGlasses()
        
        # Run directly - already initialized in constructor
        ai_glasses.run()
        
        return 0
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
