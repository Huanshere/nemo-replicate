import os
import torch
import gc
import nemo.collections.asr as nemo_asr
from cog import BasePredictor, Input, Path
from pydub import AudioSegment

class Predictor(BasePredictor):
    def setup(self):
        """加载ASR模型，只运行一次"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
        self.model.eval()
        self.model = self.model.to(self.device)
        # 配置长音频优化
        self.model.change_attention_model("rel_pos_local_attn", [256, 256])
        self.model.change_subsampling_conv_chunking_factor(1)

    def preprocess_audio(self, audio_path, max_duration_minutes=30):
        audio = AudioSegment.from_file(audio_path)
        if max_duration_minutes != 0:
            max_duration_ms = max_duration_minutes * 60 * 1000
            # 截取前 max_duration_minutes 分钟
            if len(audio) > max_duration_ms:
                audio = audio[:max_duration_ms]
        if audio.channels != 1:
            audio = audio.set_channels(1)
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        processed_path = "temp_audio.wav"
        audio.export(processed_path, format="wav")
        return processed_path, audio.duration_seconds

    def predict(
        self,
        audio: Path = Input(description="输入音频文件 (mp3/wav 都支持)"),
        max_duration_minutes: int = Input(description="最大处理分钟数，0为不截断", default=0),
        return_word_timestamps: bool = Input(description="是否返回分词时间戳", default=True),
        return_char_timestamps: bool = Input(description="是否返回分字时间戳", default=False),
    ) -> dict:
        # 1. 预处理音频
        wav_path, duration = self.preprocess_audio(str(audio), max_duration_minutes=max_duration_minutes)

        # 2. 执行识别
        output = self.model.transcribe([wav_path], timestamps=True, batch_size=1)

        # 3. 组织输出
        result = {
            "duration_seconds": duration,
            "segments": [],
        }
        if hasattr(output[0], "timestamp") and "segment" in output[0].timestamp:
            for seg in output[0].timestamp["segment"]:
                result["segments"].append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["segment"]
                })

        if return_word_timestamps and hasattr(output[0], "timestamp") and "word" in output[0].timestamp:
            result["word_timestamps"] = output[0].timestamp["word"]
        if return_char_timestamps and hasattr(output[0], "timestamp") and "char" in output[0].timestamp:
            result["char_timestamps"] = output[0].timestamp["char"]

        # 4. 清理
        if os.path.exists(wav_path):
            os.remove(wav_path)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result
