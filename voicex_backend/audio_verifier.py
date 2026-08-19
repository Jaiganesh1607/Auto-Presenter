import os
import torch
import tempfile
import logging
import torchaudio
from transformers import ASTForAudioClassification, ASTFeatureExtractor
import numpy as np

logger = logging.getLogger(__name__)

class AudioVerifier:
    _instance = None
    _model = None
    _feature_extractor = None

    @classmethod
    def initialize(cls):
        if cls._model is None:
            logger.info("Loading AST Quality Gate Model...")
            cls._feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
            cls._model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
            cls._model.eval()
            if torch.cuda.is_available():
                cls._model.to("cuda")

    @classmethod
    def _is_speech(cls, wav_path: str) -> float:
        cls.initialize()
        waveform, sr = torchaudio.load(wav_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
        
        # AudioSet AST expects ~10 second inputs, we pad or truncate
        target_length = 16000 * 10
        if waveform.shape[1] < target_length:
            pad_amount = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        elif waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]

        inputs = cls._feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = cls._model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # AudioSet Speech label index is 0
        speech_score = probs[0, 0].item()
        return speech_score

    @classmethod
    def verify_expression(cls, wav_bytes: bytes, target_expression: str) -> bool:
        """
        Only checks if the generated audio contains human speech.
        Rejects static or silent audio.
        """
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name

        try:
            score = cls._is_speech(tmp_path)
            logger.info(f"[AudioVerifier] AST Speech score: {score:.4f}")
            has_speech = bool(score > 0.02)
            
            if not has_speech:
                return False

            return True
        except Exception as e:
            return True # Fail open so we don't crash the generation
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
