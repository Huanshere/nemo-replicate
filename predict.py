import os, gc, re, time, torch
import numpy as np, pandas as pd
import nemo.collections.asr as nemo_asr
import torchaudio
from pydub import AudioSegment
from pyannote.audio import Pipeline
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        """仅在容器启动时运行一次"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ─ ASR ─
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v2"
        ).eval().to(self.device)
        # 长音频优化
        self.asr_model.change_attention_model("rel_pos_local_attn", [256, 256])
        self.asr_model.change_subsampling_conv_chunking_factor(1)

        # ─ Diarization (lazy) ─
        self.dia_model = None                      # 延迟加载

    # ────────────────────────── 工具函数 ──────────────────────────
    @staticmethod
    def preprocess_audio(audio_path, max_duration_minutes=0):
        audio = AudioSegment.from_file(audio_path)
        if max_duration_minutes:
            max_ms = max_duration_minutes * 60 * 1000
            if len(audio) > max_ms:
                audio = audio[:max_ms]
        # 转单声道 / 16 kHz
        if audio.channels != 1:
            audio = audio.set_channels(1)
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        out_path = "temp_audio.wav"
        audio.export(out_path, format="wav")
        return out_path, audio.duration_seconds

    def _load_diarization_pipeline(self):
        if self.dia_model is None:
            self.dia_model = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token="YOUR_HF_TOKEN_HERE",
            ).to(self.device)

    # ────────────────────────── 主要入口 ──────────────────────────
    def predict(
        self,
        audio: Path = Input(description="输入音频文件 (mp3/wav)"),
        max_duration_minutes: int = Input(description="最大处理分钟数，0 为不截断", default=0),
        return_word_timestamps: bool = Input(description="是否返回分词时间戳", default=True),
        speaker_diarization: bool = Input(description="是否执行说话人分离", default=False),
        min_speakers: int = Input(description="最少说话人，-1 表示不指定", default=-1),
        max_speakers: int = Input(description="最多说话人，-1 表示不指定", default=-1),
    ) -> dict:

        # ─ 1. 预处理 ─
        wav_path, duration = self.preprocess_audio(str(audio), max_duration_minutes)

        # ─ 2. ASR ─
        output = self.asr_model.transcribe([wav_path], timestamps=True, batch_size=1)
        segments_asr = [
            {"start": seg["start"], "end": seg["end"], "text": seg["segment"]}
            for seg in output[0].timestamp.get("segment", [])
            if seg["segment"]
        ]

        result = {"duration_seconds": duration, "segments": segments_asr}

        if return_word_timestamps and "word" in output[0].timestamp:
            result["word_timestamps"] = output[0].timestamp["word"]

        # ─ 3. Speaker Diarization ─
        if speaker_diarization and segments_asr:
            self._load_diarization_pipeline()

            waveform, sr = torchaudio.load(wav_path)
            diar_kwargs = {}
            if min_speakers > 0:
                diar_kwargs["min_speakers"] = min_speakers
            if max_speakers > 0:
                diar_kwargs["max_speakers"] = max_speakers

            dia_result = self.dia_model(
                {"waveform": waveform, "sample_rate": sr},
                **diar_kwargs,
            )

            dia_df = pd.DataFrame(
                [
                    {"start": turn.start, "end": turn.end, "speaker": spk}
                    for turn, _, spk in dia_result.itertracks(yield_label=True)
                ]
            )

            # ─ 3‑a 说话人与 ASR 片段对齐 ─
            for seg in segments_asr:
                dia_df["intersection"] = (
                    np.minimum(dia_df["end"], seg["end"])
                    - np.maximum(dia_df["start"], seg["start"])
                )
                overlap = dia_df[dia_df["intersection"] > 0]
                seg["speaker"] = (
                    overlap.groupby("speaker")["intersection"]
                    .sum()
                    .sort_values(ascending=False)
                    .idxmax()
                    if not overlap.empty
                    else "UNKNOWN"
                )

            # ─ 3‑b 合并同说话人、且句子连续的片段 ─
            merged, SENT_END = [], re.compile(r"[.!?]+$")
            cur = segments_asr[0].copy()
            for seg in segments_asr[1:]:
                cond_same_spk = seg["speaker"] == cur.get("speaker")
                cond_gap = seg["start"] - cur["end"] <= 1.0
                cond_len = (cur["end"] - cur["start"]) < 30.0
                cond_punc = not SENT_END.search(cur["text"][-1:])
                if cond_same_spk and cond_gap and cond_len and cond_punc:
                    cur["end"] = seg["end"]
                    cur["text"] += " " + seg["text"]
                else:
                    merged.append(cur)
                    cur = seg.copy()
            merged.append(cur)
            result["segments"] = merged

        # ─ 4. 清理 ─
        if os.path.exists(wav_path):
            os.remove(wav_path)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result
