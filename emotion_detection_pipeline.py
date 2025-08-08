import re
import warnings
from typing import Dict, Tuple, List
from transformers import pipeline
from sequence_model_handler import SequenceModelHandler
from logger import Logger

warnings.filterwarnings("ignore")

logger = Logger(
    name="AddictionEmotionDetection",
    log_file_needed=True,
    log_file_path='Logs/emotion_detection.log',
    level='DEV'
)

class EmotionDetector:
    def __init__(self, model_name="bhadresh-savani/bert-base-uncased-emotion"):
        logger.debug(f"Initializing EmotionDetector with model: {model_name}")
        try:
            self.model_handler = SequenceModelHandler(model_name)
            model, tokenizer = self.model_handler.load_sequence_model()
            self.emotion_classifier = pipeline(
                task="text-classification",
                model=model,
                tokenizer=tokenizer,
                return_all_scores=True,
                top_k=6,
            )
            
            self.educational_patterns = { }
            
            self.addiction_patterns = {
                'craving': [
                    r"(?i)(i need a fix|crave|withdrawal|dope|hit me)",
                    r"(?i)(feeling restless|jonesing|urge to use)"
                ],
                'relapse_risk': [
                    r"(?i)(i might use again|one won't hurt|just once)",
                    r"(?i)(thinking about using|tempted to use)"
                ],
                'stress_trigger': [
                    r"(?i)(can't cope|too stressed|overwhelmed|pressure)",
                    r"(?i)(life is hard|too much pain)"
                ]
            }

            logger.debug("EmotionDetector loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EmotionDetector: {e}")
            raise

    def detect(self, text: str, context: Dict = None) -> Dict:
        """
        Detect primary emotion and addiction context in text.
        """
        logger.debug(f"Detecting emotion for text: {text!r}")
        if not text or not text.strip():
            return {
                'primary_emotion': 'neutral',
                'confidence': 0.0,
                'addiction_context': 'none',
                'recommended_approach': 'standard'
            }
        
        emotion, score = self._detect_base_emotion(text)
        context_type = self._analyze_context(text)
        approach = self._get_recommended_approach(emotion, context_type)

        result = {
            'primary_emotion': emotion,
            'confidence': score,
            'addiction_context': context_type,
            'recommended_approach': approach
        }
        logger.debug(f"Detection result: {result}")
        return result

    def _detect_base_emotion(self, text: str) -> Tuple[str, float]:
        preds = self.emotion_classifier(text)
        batch = preds[0] if isinstance(preds, list) else preds
        best = batch[0]
        label = best['label'].lower()
        score = float(best['score'])
        if score < 0.6:
            return 'neutral', score
        return label, score

    def _analyze_context(self, text: str) -> str:
        for ctx, patterns in self.addiction_patterns.items():
            for pat in patterns:
                if re.search(pat, text):
                    logger.debug(f"Addiction context: {ctx}")
                    return ctx
        return 'none'

    def _get_recommended_approach(self, emotion: str, context: str) -> str:
        if context == 'relapse_risk':
            return 'relapse_prevention'
        if context == 'craving':
            return 'craving_management'
        if context == 'stress_trigger':
            return 'coping_skills'
        return 'standard'

emotion_detector = EmotionDetector()

def detect_emotion(text: str, context: Dict = None) -> Dict:
    return emotion_detector.detect(text, context)
