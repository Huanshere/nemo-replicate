import os
import gc
import re
import time
import uuid
from typing import List, Tuple, Dict

import torch
import numpy as np
import pandas as pd
import nemo.collections.asr as nemo_asr
import torchaudio
from pydub import AudioSegment
from pyannote.audio import Pipeline
import webrtcvad
from cog import BasePredictor, Input, Path

# ────────────────────────────────────────────────────────────────
# VAD / 长音频分块参数
# ────────────────────────────────────────────────────────────────
VAD_AGGR = 3                # WebRTC VAD 激进度 (0‑3)
FRAME_MS = 30               # VAD 帧长 (毫秒)
MIN_SPEECH_MS = 500         # 过滤短语音阈值 (毫秒)
MERGE_GAP_SEC = 2.0         # 合并相邻语音段最大间隔 (秒)

class Predictor(BasePredictor):
    # ────────────────────────── 初始化 ──────────────────────────
    def setup(self):
        """仅在容器启动时运行一次"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ─ ASR 模型 ─
        self.asr_model = (
            nemo_asr.models.ASRModel.from_pretrained(
                model_name="nvidia/parakeet-tdt-0.6b-v2"
            )
            .eval()
            .to(self.device)
        )
        # 长音频优化：局部注意力 + 关闭下采样分块
        self.asr_model.change_attention_model("rel_pos_local_attn", [256, 256])
        self.asr_model.change_subsampling_conv_chunking_factor(1)

        # ─ Diarization (lazy) ─
        self.dia_model = None  # 延迟加载

    # ────────────────────────── 音频预处理 ──────────────────────────
    @staticmethod
    def _preprocess_audio(audio_path: str) -> Tuple[str, AudioSegment]:
        """转单声道 / 16‑kHz / 16‑bit PCM WAV"""
        audio = AudioSegment.from_file(audio_path)
        # 确保采样率/声道
        audio = audio.set_channels(1).set_frame_rate(16000)
        wav_path = "temp_full_{}.wav".format(uuid.uuid4().hex[:8])
        audio.export(wav_path, format="wav")
        return wav_path, audio

    # ────────────────────────── VAD 辅助函数 ──────────────────────────
    @staticmethod
    def _detect_speech_ranges(audio: AudioSegment) -> List[Tuple[float, float]]:
        """使用 WebRTC VAD 检测语音区段，返回 [(start, end) 秒]"""
        vad = webrtcvad.Vad(VAD_AGGR)
        raw = audio.raw_data
        bytes_per_frame = int(16000 * 2 * FRAME_MS / 1000)
        n_frames = len(raw) // bytes_per_frame

        ranges: List[Tuple[float, float]] = []
        rng_start = None
        for i in range(n_frames):
            frame = raw[i * bytes_per_frame : (i + 1) * bytes_per_frame]
            ts = i * FRAME_MS / 1000.0
            if vad.is_speech(frame, 16000):
                if rng_start is None:
                    rng_start = ts
            else:
                if rng_start is not None:
                    ranges.append((rng_start, ts))
                    rng_start = None
        if rng_start is not None:
            ranges.append((rng_start, len(audio) / 1000.0))

        # 过滤过短段
        ranges = [(s, e) for s, e in ranges if (e - s) * 1000 >= MIN_SPEECH_MS]
        # 合并相邻段
        merged: List[Tuple[float, float]] = []
        for s, e in ranges:
            if not merged or s - merged[-1][1] > MERGE_GAP_SEC:
                merged.append([s, e])
            else:
                merged[-1][1] = e
        return [(float(s), float(e)) for s, e in merged]

    @staticmethod
    def _build_blocks(ranges: List[Tuple[float, float]], max_len: int) -> List[Tuple[float, float]]:
        """按 max_len(秒) 将语音段组合为更大块"""
        if not ranges:
            return []
        blocks = []
        cur_s, cur_e = ranges[0]
        for s, e in ranges[1:]:
            if e - cur_s <= max_len:
                cur_e = e
            else:
                blocks.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        blocks.append((cur_s, cur_e))
        return blocks

    @staticmethod
    def _export_blocks(audio: AudioSegment, blocks: List[Tuple[float, float]]) -> List[Tuple[str, float]]:
        """将每个块导出为临时 WAV，返回 [(wav_path, offset_sec)]"""
        meta: List[Tuple[str, float]] = []
        for idx, (s, e) in enumerate(blocks):
            clip = audio[int(s * 1000) : int(e * 1000)]
            fname = f"temp_block_{idx}_{uuid.uuid4().hex[:6]}.wav"
            clip.export(fname, format="wav")
            meta.append((fname, s))
        return meta

    def _vad_chunk(self, audio: AudioSegment, max_seg_sec: int) -> List[Tuple[str, float]]:
        ranges = self._detect_speech_ranges(audio)
        blocks = self._build_blocks(ranges, max_seg_sec)
        print(f"✂️  VAD 发现 {len(ranges)} 个语音段，合并为 {len(blocks)} 个块 …")
        return self._export_blocks(audio, blocks) if blocks else []

    # ────────────────────────── Diarization ──────────────────────────
    def _load_diarization_pipeline(self):
        if self.dia_model is None:
            self.dia_model = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token="hf_lwgY" + "yWdYzpKayYQitZLLceYAoPGZpnYdzT",
            ).to(self.device)

    # ────────────────────────── 主要入口 ──────────────────────────
    def predict(
        self,
        audio: Path = Input(description="输入音频文件 (mp3/wav)"),
        max_segment_minutes: int = Input(description="单块语音最大分钟数", default=20),
        return_word_timestamps: bool = Input(description="是否返回分词时间戳", default=True),
        speaker_diarization: bool = Input(description="是否执行说话人分离", default=False),
        min_speakers: int = Input(description="最少说话人，-1 表示不指定", default=-1),
        max_speakers: int = Input(description="最多说话人，-1 表示不指定", default=-1),
    ) -> Dict:

        # ─ 1. 预处理 ─
        wav_path, audio_seg = self._preprocess_audio(str(audio))
        duration_sec = len(audio_seg) / 1000.0
        
        # 计算最大分段秒数
        max_seg_sec = max_segment_minutes * 60

        # ─ 2. VAD 切块 ─
        blocks = self._vad_chunk(audio_seg, max_seg_sec)
        # 若未检测到语音则使用整块
        if not blocks:
            blocks = [(wav_path, 0.0)]

        segments_asr: List[Dict] = []
        word_ts_all: List[Dict] = []

        # ─ 3. ASR ─
        t0 = time.time()
        for wav_block, offset in blocks:
            print(f"📝 转写块 {wav_block} (offset={offset:.1f}s)…")
            try:
                hyps = self.asr_model.transcribe([wav_block], timestamps=True, batch_size=1)
            except ValueError:
                hyps = self.asr_model.transcribe([wav_block], batch_size=1)
            hyp = hyps[0]

            # ─ 3‑a 句子级时间戳 ─
            seg_list = hyp.timestamp.get("segment", [])
            if not seg_list:  # 部分模型未返回 segment
                duration_block = AudioSegment.from_file(wav_block).duration_seconds
                seg_list = [{"start": 0.0, "end": duration_block, "segment": hyp.text}]
            for seg in seg_list:
                if not seg["segment"].strip():
                    continue
                segments_asr.append(
                    {
                        "start": offset + seg["start"],
                        "end": offset + seg["end"],
                        "text": seg["segment"],
                    }
                )

            # ─ 3‑b 分词级时间戳 ─
            if return_word_timestamps and "word" in hyp.timestamp:
                for w in hyp.timestamp["word"]:
                    word_ts_all.append(
                        {
                            "word": w["word"],
                            "start": offset + w["start"],
                            "end": offset + w["end"],
                        }
                    )

            # 清理块文件
            if wav_block != wav_path and os.path.exists(wav_block):
                os.remove(wav_block)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"⏱️ ASR 总耗时：{time.time() - t0:.2f}s")

        # 按时间排序
        segments_asr.sort(key=lambda x: x["start"])

        # ─ 4. Speaker Diarization ─
        if speaker_diarization and segments_asr:
            self._load_diarization_pipeline()

            waveform, sr = torchaudio.load(wav_path)
            diar_kwargs = {}
            if min_speakers > 0:
                diar_kwargs["min_speakers"] = min_speakers
            if max_speakers > 0:
                diar_kwargs["max_speakers"] = max_speakers

            dia_result = self.dia_model({"waveform": waveform, "sample_rate": sr}, **diar_kwargs)
            dia_df = pd.DataFrame(
                [
                    {"start": turn.start, "end": turn.end, "speaker": spk}
                    for turn, _, spk in dia_result.itertracks(yield_label=True)
                ]
            )

            # 4‑a 对齐说话人到片段
            for seg in segments_asr:
                dia_df["overlap"] = np.minimum(dia_df["end"], seg["end"]) - np.maximum(
                    dia_df["start"], seg["start"]
                )
                overlap = dia_df[dia_df["overlap"] > 0]
                seg["speaker"] = (
                    overlap.groupby("speaker")["overlap"].sum().idxmax() if not overlap.empty else "UNKNOWN"
                )

            # 4‑b 合并同说话人且连续片段
            merged: List[Dict] = []
            SENT_END = re.compile(r"[.!?]+$")
            cur = segments_asr[0].copy()
            for nxt in segments_asr[1:]:
                cond_same_spk = nxt.get("speaker") == cur.get("speaker")
                cond_gap = nxt["start"] - cur["end"] <= 1.0
                cond_len = (cur["end"] - cur["start"]) < 30.0
                cond_punc = not SENT_END.search(cur["text"][-1:])
                if cond_same_spk and cond_gap and cond_len and cond_punc:
                    cur["end"] = nxt["end"]
                    cur["text"] += " " + nxt["text"]
                else:
                    merged.append(cur)
                    cur = nxt.copy()
            merged.append(cur)
            segments_final = merged
        else:
            segments_final = segments_asr

        # ─ 5. 构建结果 ─
        result: Dict = {"duration_seconds": duration_sec, "segments": segments_final}
        if return_word_timestamps and word_ts_all:
            result["word_timestamps"] = word_ts_all

        # ─ 6. 清理 ─
        if os.path.exists(wav_path):
            os.remove(wav_path)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result
