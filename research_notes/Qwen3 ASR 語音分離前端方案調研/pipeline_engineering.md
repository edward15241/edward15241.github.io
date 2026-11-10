# Pipeline engineering: external speech separator + Qwen3-ASR-0.6B for single-channel overlapped speech

Legend for every figure: **[P]** = read by me in the primary source (paper HTML, model card, repo README, HF API) on 2026-11-10; **[S]** = seen only in a web-search snippet, not verified in the primary source. "Not Reported" = looked for in the cited source and not found. Sections "Inferences" are my own design reasoning, not published facts.

---

## 1. Reference pipeline designs actually published or open-sourced

### Takeaway
All four topologies exist in public form, but only (c) PixIT and (d) NOTSOFAR-1 CSS ship as runnable single-channel "separator → Whisper-class ASR" pipelines with WER numbers on real recordings; on real meetings their absolute error rates remain high (AMI-SDM cpWER ≈ 32–34 %, NOTSOFAR single-channel tcORC-WER 38.5 %), while on synthetic Libri2Mix a strong separator plus an off-the-shelf Whisper large-v3 reaches 5.0 % WER. Every published single-channel cascade that reports a no-overlap condition shows a small degradation there, which is the motivation for design (b).

### Cited Findings

**(a) Always-separate → N× ASR**
- [P] TF-CrossNet separator → ASR on Libri2Mix (noisy 2-talker mixtures separated into two streams): clean-trained WavLM-based backend 5.1 % dev / 5.1 % test WER; **off-the-shelf Whisper large-v3 on the separated streams: 5.6 % dev / 5.0 % test** (Whisper large-v3 on ground-truth single-speaker audio: 2.5 %). Same table lists single-pass SOT baselines: SOT-Conformer 23.3 % test, SOT-WavLM-Conformer 17.1 %, WavLM-LLM w/ fine-tune 9.0 %, PIT-SOT 7.0 %, TSE-Whisper 12.0 %. — [Yang et al., arXiv 2503.17886, Table 1](https://arxiv.org/html/2503.17886v1)
- [P] Same paper, 1-ch SMS-WSJ (8 kHz, reverberant): unprocessed 79.11 % WER; TF-GridNet 1-stage 9.49 %; TF-CrossNet 8.13 % with task-standard AM, 7.60 % with clean (direct-path-trained) AM; oracle direct-path 6.16 %. — [arXiv 2503.17886, Tables 2 & 4](https://arxiv.org/html/2503.17886v1)
- [P] ESPnet-SE++ provides `enh_asr.sh`, a "plug-and-play" combination of SE/separation front-ends with ASR/ST/SLU back-ends; the paper's combined recipes are CHiME-4, LT-S and SLURP-S (not a LibriMix separator+Whisper recipe); after pre-training "the entire SE+ASR system is fine-tuned together with both SE and ASR losses", and "some models resulted in worse WERs on the real data … especially when they are not jointly optimized with the downstream ASR task". — [ESPnet-SE++, arXiv 2207.09514](https://arxiv.org/html/2207.09514)
- [P] The E2E survey characterises separation-based cascades as: separate into single-speaker streams, then single-speaker ASR on each; overall accuracy "is dependent on the separation models". — [Survey of End-to-End Multi-Speaker ASR for Monaural Audio, arXiv 2505.10975v3 §1.1](https://arxiv.org/html/2505.10975v3)

**(b) Overlap detector → separate only overlapped regions → ASR**
- [P] USTC-NERCSLIP CHiME-8 NOTSOFAR-1 system: an overlap-segment-detection model (same architecture as the official CSS separator, 12.8 s window, frame-level binary output) gates processing; "for the detected overlapping segments" a 3-second CSS is applied; non-overlapped segments bypass separation (MVDR in the multi-channel track). Adding the overlap-classification network to the separator improved tcpWER "from 26.68 % to 25.14 %" with a fixed ASR back-end (their Table 6); the single-channel track uses the same structure with single-channel inputs and T-F-mask separation. — [arXiv 2409.02041v2](https://arxiv.org/html/2409.02041v2)
- [S] WhisperX maintainer comment: "probably worth using speech separation only for overlapping segments". — [whisperX issue #9](https://github.com/m-bain/whisperX/issues/9)
- [S] Czech streaming system with a compact overlap detector + target-speaker path reports WER on overlapped segments 68.0 % → 35.78 % on TV debates with 16 % overlap. — [arXiv 2506.20288](https://arxiv.org/html/2506.20288v1)
- [S] Earlier work argues for "performing overlap detection and speaker separation in tandem, i.e., only separating the signal in the overlapped regions" and notes training on mixed overlapped/non-overlapped data can degrade separation. — [par.nsf.gov/servlets/purl/10302081](https://par.nsf.gov/servlets/purl/10302081)

**(c) Joint diarization + separation (pyannote PixIT) → ASR per speaker stream**
- [P] `pyannote/speech-separation-ami-1.0`: "ingests mono audio sampled at 16kHz and outputs speaker diarization as an `Annotation` … and speech separation as a `SlidingWindowFeature`"; sources are written as 16 kHz wavs (`scipy.io.wavfile.write(f'{speaker}.wav', 16000, sources.data[:,s])`); trained with pyannote.audio 3.3.2 on AMI-SDM; requires accepting user conditions for both `pyannote/separation-ami-1.0` and `pyannote/speech-separation-ami-1.0`; runs on CPU by default, `.to(torch.device("cuda"))` for GPU. — [model card](https://huggingface.co/pyannote/speech-separation-ami-1.0)
- [P] Architecture (ToTaToNet): TasNet with 1-D conv encoder/decoder (kernel 32, stride 16, 64 filters) + concatenated **WavLM-large** features + DPRNN separator + small per-source diarization head; 5-second sliding windows, 500 ms step; K_max = 3 speakers per window; trained 100 epochs; training only on the target corpus train set (AMI ≈ 100 h, AliMeeting ≈ 120 h). — [PixIT, arXiv 2403.02288](https://arxiv.org/html/2403.02288)
- [P] Long-form stitching: per-chunk speaker activations are binarized, local ECAPA-TDNN speaker embeddings are clustered (agglomerative), and sources are stitched by cluster. **Leakage removal**: "setting the stitched separated sources at time t to zero when the diarization outputs predict that the corresponding speaker is not active and has not been active in a window [t−Δt, t+Δt]". — [arXiv 2403.02288 §2.3](https://arxiv.org/html/2403.02288)
- [P] AMI-SDM cpWER (ASR applied off-the-shelf to each separated source, WhisperX for timestamps; baseline = Whisper on mixture + pyannote 3.1 diarization): Whisper small.en 39.6 → 33.9 (−14.4 %); medium.en 38.8 → 32.8 (−15.4 %); large-v2 38.3 → 31.7 (−17.2 %); NeMo conformer-CTC-large 48.9 → 39.4 (−19.4 %). — [arXiv 2403.02288 Table 2](https://arxiv.org/html/2403.02288)
- [P] Speaker-agnostic uWER on AMI-SDM, original audio vs separated sources: small.en 37.6 → 36.3 (−3.5 %); medium.en 37.1 → 35.1 (−5.4 %); large-v2 35.4 → 33.8 (−4.5 %); NeMo 49.3 → 41.6. "the bulk of the WER improvement comes from deletions". — [arXiv 2403.02288 Table 3](https://arxiv.org/html/2403.02288)
- [P] AliMeeting ch1 (Mandarin) cpCER: Whisper small 68.6 → 54.0 (−21.3 %); medium 65.9 → 50.3 (−23.7 %); large-v2 65.1 → 48.3 (−25.8 %), "even though WavLM has been pre-trained on English data". — [arXiv 2403.02288 Table 4](https://arxiv.org/html/2403.02288)
- [P] Ablation (Whisper medium.en, AMI-SDM cpWER; baseline 38.8): no WavLM / no leakage removal **50.1** (sub 19.2, del 15.3, **ins 15.6**); no WavLM + leakage removal 36.2; WavLM / no leakage removal 34.1 (ins 3.8); WavLM + leakage removal 32.8 (ins 1.2). — [arXiv 2403.02288 Table 5](https://arxiv.org/html/2403.02288)
- [P] Code/training are part of pyannote.audio ≥ 3.3.0; evaluation scripts use openai-whisper, whisperx, jiwer. — [PixIT repo README](https://github.com/joonaskalda/PixIT)

**(d) CSS chunk-and-stitch → ASR**
- [P] CSS definition and LibriCSS (10 h, re-recorded LibriSpeech, overlap 0–40 %). Utterance-wise WER, single-channel mask separator (3×1024 BLSTM, trained on 206 h simulated) vs no separation: 0S 12.7 vs 11.8; 0L 12.1 vs 11.7; 10 % 17.6 vs 18.8; 20 % 23.2 vs 27.2; 30 % 30.5 vs 35.6; 40 % 35.6 vs 43.3. Authors: single-channel separation "degraded the WER when there was no speech overlap". — [Chen et al., arXiv 2001.11482, Table 1](https://arxiv.org/html/2001.11482)
- [P] Stitching: sliding window with past/current/future sub-windows; output order of neighbouring chunks aligned using shared frames; chunk configs tested (7-ch) 1.2-0.8-0.4 s (latency 1.2 s), 1.6-0.8-0.0 (0.8 s), 0.8-0.4-0.4 (0.8 s). — [arXiv 2001.11482 §3, Table 2](https://arxiv.org/html/2001.11482)
- [P] NOTSOFAR-1 open-source baseline: conformer CSS trained with PIT producing **N = 3 speech masks + 1 noise mask**, applied block-wise to overlapping fixed-length segments; adjacent segments aligned by trying all permutations and picking lowest MSE on shared frames; single-channel variant = mixture × mask; **Whisper large-v2** applied independently to each stream; diarization afterwards on words. ~1000 h simulated training set provided. — [Vinnikov et al., arXiv 2401.08887](https://arxiv.org/html/2401.08887)
- [P] NOTSOFAR baseline results, dev-set-1, tcpWER / tcORC-WER: **single-channel all sessions 46.8 / 38.5** (177 sessions); #TurnsNoOverlap 32.4 / 29.7; #DebateOverlaps 54.9 / 44.7; multi-channel all 32.4 / 26.7. — [NOTSOFAR1-Challenge README](https://github.com/microsoft/NOTSOFAR1-Challenge)
- [P] TF-GridNet as CSS separator on single-mic LibriCSS: 4 s window, 3 s shift (1 s overlap), two outputs, permutation by minimum MSE on the overlap, then energy-based VAD segmentation per output channel. ORC-WER (avg over overlap ratios): BLSTM separator 21.6 % (4-gram) / 19.4 % (Transformer LM); TF-GridNet 7.9 / 6.4 with no AM fine-tuning; 7.2 / 5.8 after fine-tuning AM on separated data; 7.0 / 5.6 with sMBR; oracle signals + segmentation 3.1 / 2.1. — [arXiv 2309.08454, Tables 2–3](https://arxiv.org/html/2309.08454)
- [P] Microsoft (Interspeech 2025) feeds single-channel CSS outputs (WavLM-based conformer CSS, 2.4 s window + 0.8 s hop, two channels) into an SOT model. LibriCSS SAgWER, streaming CT-tSOT without vs with CSS: 0L 7.76 → 8.18; 0S 7.33 → 8.76; 10 % 8.73 → 9.55; 20 % 11.78 → 11.62; 30 % 14.25 → 13.07; 40 % 16.47 → 14.97; avg 11.52 → 11.38. Offline S2S-segSOT: avg 10.24 → 9.93 with CSS; "some degradation" on single-talker sets, "8–9 % relative" gain at 30–40 % overlap. — [arXiv 2506.14204, Table 1](https://arxiv.org/html/2506.14204v1)

**Toolkits**
- [P] ClearerVoice-Studio: unified inference (`pip install clearvoice`) with `MossFormer2_SS_16K` for 16 kHz separation, plus train/finetune scripts for separation at 8 kHz and 16 kHz; numpy-in/numpy-out interface added 2025-06. It does not bundle an ASR step. — [ClearerVoice-Studio README](https://github.com/modelscope/ClearerVoice-Studio), [clearvoice README](https://github.com/modelscope/ClearerVoice-Studio/blob/main/clearvoice/README.md)
- [P] SpeechBrain SepFormer checkpoints (see §3/§5 for sample rates) — [speechbrain/sepformer-libri2mix](https://huggingface.co/speechbrain/sepformer-libri2mix)
- [P] A diarization (not separation) cascade with the same recognizer family is benchmarked in MS-Bench: pyannote `speaker-diarization-community-1` → Qwen3-ASR gets DER 17.35 %, cpWER 43.25 % overall; across all seven evaluated systems heavy overlap (r_ov ≥ 0.5 vs 0) raises lexical error by 27.2–38.6 pp. — [MS-Bench, arXiv 2610.25824](https://arxiv.org/abs/2610.25824)
- [S] Vendor (non-peer-reviewed) claim: separation + Whisper large-v3 on LibriCSS is "as accurate as … clean single-speaker stems" up to 20 % overlap. — [AudioShake blog](https://www.audioshake.ai/post/multi-speaker-2-0-technical-evaluation)

### Inferences
- For the stated requirement (all text, no speaker identity), topology (d) with N = 2–3 output streams, or (b) = OSD-gated (d), is the closest match; (c) adds speaker clustering the team does not need, but its diarization head is what enables leakage removal, which was worth 50.1 → 36.2 cpWER without WavLM and 34.1 → 32.8 with it.
- The pattern "single-channel separation helps at ≥ 10–20 % overlap and hurts slightly at 0 %" appears in three independent sources (LibriCSS 2020, Microsoft 2025, PixIT sub/ins breakdown), supporting an overlap gate or a pass-through path for non-overlapped audio.
- The 5.0 % Libri2Mix figure for Whisper large-v3 behind TF-CrossNet is synthetic, matched-domain, English read speech; the real-meeting numbers (31–47 %) are the more relevant planning anchor for spontaneous Taigi/Mandarin audio.

### Gaps
- No published or open-source pipeline combining any separator with Qwen3-ASR (either size) was found; MS-Bench's Qwen3-ASR cascade is diarization-only and its model size was not stated in the text I read.
- No NeMo recipe for single-channel separation → ASR was found; NeMo's multi-speaker path is Sortformer-based (diarization / speaker-supervised ASR), which falls under the end-to-end routes excluded from this note.
- ESPnet `egs2/librimix/enh_asr1` README could not be fetched (404); I could not verify a LibriMix separation+ASR joint recipe or its WER.
- No source reports single-speaker WER before/after an OSD-gated separation with a Whisper-class or Speech-LLM recognizer on the same test set.

---

## 2. Overlapped speech detection (OSD) and speaker counting components

### Takeaway
pyannote's segmentation models are the de-facto open OSD/speaker-count component (MIT licence, gated download), but published OSD accuracy is modest (AMI F1 75.3 % for pyannote 2.0; best recent WavLM model 82.76 %), and the powerset model represents at most two simultaneous speakers per frame, so it cannot flag three-way overlap as such.

### Cited Findings
- [P] `pyannote/segmentation-3.0`: ingests 10 s mono 16 kHz chunks; outputs 7 powerset classes: non-speech, speaker #1, #2, #3, speakers #1+#2, #1+#3, #2+#3 (`max_speakers_per_chunk, max_speakers_per_frame = 3, 2`); ships `OverlappedSpeechDetection` and `VoiceActivityDetection` pipelines with only `min_duration_on/off` hyper-parameters; trained on AISHELL, AliMeeting, AMI, AVA-AVD, DIHARD, Ego4D, MSDWild, REPERE, VoxConverse; "cannot be used to perform speaker diarization of full recordings on its own (it only processes 10s chunks)". — [model card](https://huggingface.co/pyannote/segmentation-3.0)
- [P] Why no 3-speaker-overlap class: frames with ≥ 3 simultaneous speakers are "marginal: 1.6 % in the compound dataset, 0.73 % in DIHARD III"; powerset removes the detection-threshold hyper-parameter and improves mostly on overlapped speech. — [Plaquet & Bredin, arXiv 2310.13025](https://arxiv.org/html/2310.13025)
- [P] pyannote 2.0 segmentation OSD results (precision / recall / F1; FA / miss): **AMI 80.7 / 70.5 / 75.3** (FA 16.9, miss 29.4); **DIHARD 3 57.2 / 62.8 / 59.9** (FA 46.9, miss 37.2); **VoxConverse 74.2 / 75.5 / 74.8**. pyannote 1.1 on AMI: 63.2 / 87.9 / 73.5. Model: SincNet + 4 BiLSTM, 5 s chunks, ~16 ms resolution; training "took around 3 days using 4 V100 GPUs". Authors: "overlapped speech detection remains an unsolved (and sometimes ill-defined) problem". — [Bredin & Laurent, arXiv 2104.04045, Table 2](https://arxiv.org/html/2104.04045)
- [P] Tuned OSD thresholds for that model (AMI: onset 0.448, offset 0.362, min_on 0.116, min_off 0.187; DIHARD3: 0.430 / 0.320 / 0.091 / 0.144). — [pyannote/segmentation card](https://huggingface.co/pyannote/segmentation)
- [P] Speaker-aware progressive OSD with WavLM (trained on LibriHeavyMix 240 h + AliMeeting 104.75 h + AMI 75 h): AMI test recall 81.48 / precision 84.08 / **F1 82.76**; XLSR-Conformer 79.21 F1; conformer 64.73. Its Table 2 lists pyannote as recall 80.7 / precision 70.5 / F1 75.3 — i.e. precision and recall **swapped relative to the pyannote paper's own Table 2** (same F1). Dataset overlap ratios quoted: AliMeeting 42.27 %, AMI 19 %. — [Sun et al., arXiv 2505.23207](https://arxiv.org/html/2505.23207v1); conflicts on P/R ordering with [arXiv 2104.04045](https://arxiv.org/html/2104.04045)
- [P] Licence/gating: `pyannote/segmentation-3.0`, `pyannote/segmentation`, `pyannote/speaker-diarization-3.1` are `license: mit`, `gated: auto` (must supply company/website; "we will occasionally email you about premium models"); `pyannote/speaker-diarization-community-1` is `cc-by-4.0`, gated. — [HF API segmentation-3.0](https://huggingface.co/api/models/pyannote/segmentation-3.0), [HF API community-1](https://huggingface.co/api/models/pyannote/speaker-diarization-community-1)
- [P] pyannote README DER benchmark (2025-09) legacy 3.1 / community-1 / precision-2: AliMeeting ch1 24.5 / 20.3 / 15.2; AMI-SDM 22.7 / 19.9 / 15.6; AISHELL-4 12.2 / 11.7 / 11.4 (DER, not OSD P/R). — [pyannote-audio README](https://github.com/pyannote/pyannote-audio)
- [P] Alternative used in a top challenge system: reuse the CSS separator architecture as a frame-level overlap classifier, or train overlap classification jointly with the separator (26.68 → 25.14 tcpWER). — [arXiv 2409.02041v2](https://arxiv.org/html/2409.02041v2)
- [P] NVIDIA Sortformer (frame-level activity for up to 4 speakers, from which overlap can be derived): `diar_sortformer_4spk-v1` is `cc-by-nc-4.0`; `diar_streaming_sortformer_4spk-v2` card metadata says `cc-by-4.0`. — [HF API v1](https://huggingface.co/api/models/nvidia/diar_sortformer_4spk-v1), [HF API v2](https://huggingface.co/api/models/nvidia/diar_streaming_sortformer_4spk-v2)
- [S] A concurrent-speaker-detection study on AMI/AliMeeting/AISHELL-4/NOTSOFAR-1 reports overlap F1 ≈ 0.7 for vanilla WavLM and ≈ 0.8 for a pruned WavLM (one-decimal rounding). — [arXiv 2607.12647](https://arxiv.org/pdf/2607.12647)

### Inferences
- With AMI recall ≈ 70 % (pyannote 2.0), an OSD-gated design leaves roughly 30 % of overlapped time unseparated on that domain; thresholds should be tuned for recall, since a false alarm costs only an unnecessary separation pass while a miss loses the secondary talker entirely.
- Because segmentation-3.0 caps at two speakers per frame, a 3-talker region will be labelled as 2-speaker overlap; the count cannot be used to switch between a 2-output and a 3-output separator.
- The same segmentation model yields VAD, so one forward pass can supply both the overlap gate and the per-stream activity gate needed in §5.

### Gaps
- OSD precision/recall of `segmentation-3.0` (powerset) specifically, and any OSD P/R on AliMeeting, were Not Reported in the sources I could read.
- Inference cost (RTF, VRAM) of pyannote segmentation/OSD: Not Reported in the model cards or papers read.
- No OSD accuracy on Taiwanese Hokkien or on Taigi+Mandarin cross-language overlap was found.

---

## 3. Training or fine-tuning a separator on own-language data

### Takeaway
Standard supervised recipes need only tens to low hundreds of hours of single-speaker source audio (WSJ0-2mix 30 h, Libri2Mix train-360 212 h of mixtures) plus dynamic mixing, and train on one to four GPUs over roughly a week or more; fine-tuning a pretrained separator is supported by tooling (ClearerVoice, pyannote/PixIT) and documented for domain shift, but I found no documented English→Hokkien/Mandarin fine-tuning study. Evidence that English-trained separators lose performance on Chinese speech exists (≈ 5–7.5 dB SI-SNRi on CommonVoice zh-TW/zh-CN/zh-HK), as does evidence that training from noisy/real material works via MixIT-family methods when combined with some supervised synthetic data.

### Cited Findings

**Data volumes used by recipes**
- [P] wsj0-{2,3}mix: 20,000 train utterances = 30 h; 5,000 dev = 8 h; 3,000 test = 5 h. Libri2Mix: train-360 50,800 mixtures = **212 h**; train-100 13,900 = 58 h; dev/test 3,000 each = 11 h. Libri3Mix train-360 33,900 = 146 h. Source LibriSpeech train-clean-360 = 364 h, 921 speakers; train-clean-100 = 101 h, 251 speakers. — [LibriMix, arXiv 2005.11262, Tables 1–2](https://arxiv.org/html/2005.11262)
- [P] LibriMix motivation: models trained on wsj0-2mix generalise poorly; "generalization error is smaller for models trained with LibriMix than with WHAM!"; real-world overlap ratios are "in the order of 20 % or less", motivating sparse-overlap test sets. — [arXiv 2005.11262](https://arxiv.org/html/2005.11262)
- [P] LibriCSS single-channel separator: trained on 206 h of artificially reverberated/mixed LibriSpeech. NOTSOFAR CSS: ~1000 h simulated set built from MOS-filtered Librivox speech with 15,000 real acoustic transfer functions. — [arXiv 2001.11482](https://arxiv.org/html/2001.11482), [arXiv 2401.08887](https://arxiv.org/html/2401.08887)
- [P] TIGER's EchoSet: 20,268 train / 4,604 val / 2,650 test utterances of 6 s (≈ 33.8 h train by count×length — arithmetic mine), simulated with realistic reverberation + WHAM! noise, random overlap ratio; models trained on EchoSet "generalized better on real-world data than those trained on … LRS2-2Mix and Libri2Mix". — [TIGER, arXiv 2410.01469](https://arxiv.org/html/2410.01469)

**Dynamic mixing (DM)**
- [P] SepFormer: DM = "on-the-fly creation of new mixtures from single speaker sources", extended with speed perturbation 95–105 %; WSJ0-2mix 22.3 dB SI-SNRi with DM; relative levels sampled 0–5 dB. — [arXiv 2010.13154](https://arxiv.org/html/2010.13154)
- [P] TF-GridNet reaches 23.5 dB SI-SDRi on WSJ0-2mix "without using any data augmentation and dynamic mixing"; trains on 4-second segments. — [arXiv 2211.12433](https://arxiv.org/html/2211.12433)
- [P] TF-Locoformer: up to 150 epochs (200 with DM), batch size 4, 4-second inputs; DM gave "noticeable improvements". — [arXiv 2408.03440](https://arxiv.org/html/2408.03440)

**Training compute as reported**
- [P] SepFormer (26 M params): trained with batch size 1 and AMP on a single NVIDIA V100-32GB; "reaches above 17 dB levels only after a full day of training, whereas the DPRNN model requires two days". — [arXiv 2010.13154 §4.4](https://arxiv.org/html/2010.13154)
- [P] SpeechBrain recipe: "Each epoch takes about 2 hours for WSJ0-2Mix and WSJ0-3Mix (DynamicMixing) on a NVIDIA V100 (32GB)". Total epochs / total days: Not Reported in the README section read. — [SpeechBrain WSJ0Mix separation README](https://github.com/speechbrain/speechbrain/blob/develop/recipes/WSJ0Mix/separation/README.md)
- [P] MossFormer2 (55.7 M params): "training takes place on a single NVIDIA V100 GPU with 32 GB", up to 200 epochs, batch size 1, LR constant for 85 epochs then halved. Wall-clock days: Not Reported. — [arXiv 2312.11825](https://arxiv.org/html/2312.11825)
- [P] TF-GridNet (14.5 M params, largest config): 231.1 GMAC/s; GPU memory 4,918.8 MB forward / 13,663.2 MB backward per 4-s segment; **151.4 min/epoch** for 20,000 4-s segments on an A100-40GB; smaller configs 8.2 M params → 119.5 or 71.6 min/epoch; 2.1 M params → 11.4 min/epoch. Total epochs/days: Not Reported in the rows read. (Forward/backward ms columns of this table did not parse reliably from HTML and are deliberately omitted.) — [arXiv 2211.12433, cost table](https://arxiv.org/html/2211.12433)
- [P] TF-Locoformer: "We used 4 GPUs for training (batch size was 1 on each GPU; GPU RAM depends on dataset)". GPU model and days: Not Reported. — [tf-locoformer README](https://github.com/merlresearch/tf-locoformer)
- [P] pyannote segmentation (OSD/VAD) model: ~3 days on 4 V100. — [arXiv 2104.04045](https://arxiv.org/html/2104.04045)
- [P] ASR-side reference: fine-tuning an acoustic model on TF-GridNet-separated data took ~17 h/epoch (f-CE) on a GTX 1080 Ti. — [arXiv 2309.08454](https://arxiv.org/html/2309.08454)
- [P] PixIT/ToTaToNet: 100 epochs, LR 1e-5 for WavLM and 3e-4 elsewhere; GPU type/time Not Reported. — [arXiv 2403.02288](https://arxiv.org/html/2403.02288)

**Fine-tuning a pretrained separator / cross-language behaviour**
- [P] ClearerVoice provides "model finetune and training scripts" for separation at 8 kHz and 16 kHz (`MossFormer2_SS_8K`, `MossFormer2_SS_16K`); data prep documented for WSJ0-2mix ("We assume you have purchased WSJ0") or LibriMix from LibriSpeech train-clean-360 + WHAM! noise; training is driven by scp lists of mixture/source paths, with a flag to continue from an existing checkpoint. — [ClearerVoice train/speech_separation README](https://github.com/modelscope/ClearerVoice-Studio/blob/main/train/speech_separation/README.md)
- [P] Released `MossFormer2_SS_16K` is a "unified model, evaluated without retraining on individual datasets": SI-SNRi (as tabulated) 15.5 on LRS2-2Mix (16 kHz), 22.0 on WSJ0-2Mix, 16.7 on Libri2Mix, 17.4 on WHAM! (scores computed on 8 kHz-downsampled audio). — [clearvoice README](https://github.com/modelscope/ClearerVoice-Studio/blob/main/clearvoice/README.md)
- [P] Fine-tuning for domain shift is documented: 1-ch TF-CrossNet pretrained on SMS-WSJ then fine-tuned with SI-SDR loss at lowered LR on a 4× larger, wider-condition set (134,244 utterances). — [arXiv 2503.17886 §3.2](https://arxiv.org/html/2503.17886v1)
- [P] English-trained separators evaluated on CommonVoice Chinese test splits (2-speaker SI-SNRi, joint-training baseline): Conv-TasNet zh-CN 5.69 / zh-HK 6.87 / **zh-TW 5.27** dB; DPRNN zh-CN 6.71 / zh-HK 7.47 dB; with fine-tuning Conv-TasNet 5.93 / 6.78 / 5.38; 3-speaker Conv-TasNet 3.82 / 5.47 / 3.47. On English accents the same models score ≈ 8–9 dB average. Meta-learning (MAML/FOMAML) variants: e.g. DPRNN FOMAML 6.56 / 6.89 / 6.21. — [Huang et al., arXiv 2203.05882, Tables 1–3](https://ar5iv.labs.arxiv.org/html/2203.05882)
- [S] GlobalPhone Mix-to-Separate (22 languages): "when trained on a multilingual dataset, the networks improve their performances for unseen languages". — [Borsdorf et al., Interspeech 2021](https://www.isca-archive.org/interspeech_2021/borsdorf21b_interspeech.html)
- [P] PixIT trained on Mandarin AliMeeting with an English-pretrained WavLM still gave −21 to −26 % cpCER, i.e. English SSL features did not prevent a Mandarin separator from working. — [arXiv 2403.02288](https://arxiv.org/html/2403.02288)

**Training from noisy / real / non-studio sources**
- [P] MixIT: "requires only single-channel acoustic mixtures"; adding "a small amount of unsupervised data (10 %) from a matched domain drastically improves performance": +2–3 dB with mismatched anechoic supervised + matched reverberant unsupervised data; no penalty for over-separation — semi-supervised training mitigates it. — [Wisdom et al., arXiv 2006.12701](https://arxiv.org/html/2006.12701)
- [P] MixIT adaptation to real AMI meetings: "a fine-tuned semi-supervised model yields the largest SI-SNR improvement, PESQ scores, and human listening ratings across synthetic and real datasets, outperforming unadapted generalist models trained on orders of magnitude more data"; best config = PIT on reverberant Libri2Mix + MixIT on AMI, warm-started from MixIT on 5,800 h AudioSet; AMI data is "tens of hours". — [Sivaraman et al., arXiv 2110.10739](https://arxiv.org/html/2110.10739)
- [P] PixIT needs only "single-channel recordings with speaker diarization labels", not clean sources; MoMs are built from two chunks **of the same recording** with disjoint speakers and total speakers ≤ K_max, because otherwise "the model might learn to exploit" recording-condition differences. — [arXiv 2403.02288](https://arxiv.org/html/2403.02288)
- [P] RemixIT: self-training for enhancement "without the need of a single isolated in-domain speech nor a noise waveform" (teacher–student with bootstrapped remixing). Self-Remixing: refines a PIT (out-of-domain) or MixIT pre-trained separator unsupervised and "outperforms baselines in semi-supervised domain adaptation", with experiments on both synthetic and recorded data. — [RemixIT, arXiv 2202.08862](https://arxiv.org/html/2202.08862); [Self-Remixing, arXiv 2211.10194](https://arxiv.org/html/2211.10194)
- [P] NOTSOFAR found source quality matters enough to MOS-filter Librivox before simulation ("quality variance compromises simulation"). — [arXiv 2401.08887](https://arxiv.org/html/2401.08887)
- [P] USTC simulated extra separation training data from near-field (close-talk) recordings of the target corpus. — [arXiv 2409.02041v2](https://arxiv.org/html/2409.02041v2)

### Inferences
- Combining the SpeechBrain figure (≈ 2 h/epoch on one V100) with the 200-epoch schedules used by SepFormer-family/MossFormer2 papers implies on the order of 400 GPU-hours for a from-scratch WSJ0-2mix-sized run; this product is my arithmetic, not a reported total.
- The team's single-speaker ASR corpora can serve as mixing sources in the standard supervised (PIT + dynamic mixing) recipe; the published caveats are (i) sources containing noise/reverb become part of the "target", so the separator learns to reproduce them (acceptable for ASR, cf. the "clean-backend vs distortion-tolerant" discussion in 2503.17886), (ii) utterances containing background speakers corrupt targets (NOTSOFAR's MOS filtering is the published precedent for screening), and (iii) mixing two utterances from different recording conditions gives the model a shortcut (PixIT's same-recording rule).
- Cross-language mixtures (Taigi+Mandarin) can be generated trivially by dynamic mixing across corpora; no source reports whether cross-language pairs are easier or harder than same-language pairs for a monaural separator (the [S] "blind language separation" result suggests language difference is an exploitable cue).
- The zh-TW result (5.27 dB for an English-trained Conv-TasNet vs ≈ 8–9 dB on English accents) is the only published number bearing on Taiwan-accent data and supports budgeting for in-language fine-tuning rather than relying on English checkpoints.

### Gaps
- No paper or repo documenting fine-tuning of SepFormer / TF-GridNet / MossFormer2 / TF-Locoformer from English to Mandarin or Taiwanese Hokkien was found; nothing at all on Hokkien separation.
- Training data composition of the released `MossFormer2_SS_16K` checkpoint: Not Reported in the READMEs read; whether it supports 3 speakers: Not Reported (documented benchmarks are all 2-speaker).
- Total wall-clock training time (days) for TF-GridNet, MossFormer2, TF-Locoformer: Not Reported in sources read; TF-Locoformer GPU model Not Reported.
- No study quantifying how source-corpus SNR (noisy ASR data vs studio data) affects downstream WER of the cascade.

---

## 4. Inference cost of the cascade

### Takeaway
Using only published figures, the recognizer is not the bottleneck: Qwen3-ASR-0.6B's offline RTF is 0.00923 at concurrency 1 and 0.01124 at concurrency 2, so two or three streams cost little; the separator dominates and varies by two orders of magnitude across architectures (GPU RTF 0.053 for MossFormer2 on a V100 at 8 kHz; TF-GridNet ≈ 94 ms GPU / ≈ 2,020 ms CPU per 1 s of 16 kHz audio). CPU-only deployment is realistic only with light separators (Conv-TasNet / SudoRM-RF class).

### Cited Findings

**Recognizer (Qwen3-ASR-0.6B)**
- [P] vLLM v0.14.0, CUDA Graph, bfloat16, ~2-minute inputs, "a single typical computing resource" (hardware not named). Offline batch: conc 1 RTF **0.00923** (throughput 108.34 s audio/s); conc 2 RTF **0.01124** (177.94); conc 4 0.01284 (311.53); conc 8 0.01600 (500.00); conc 16 0.02384 (671.14); conc 128 0.11264 (1136.36). Online async: conc 1 TTFT avg 92 ms / p95 105 ms, RTF 0.00940; conc 2 TTFT 103 / 168 ms, RTF 0.01108; conc 4 TTFT 132 / 203 ms; conc 128 RTF 0.064, throughput 2000. — [Qwen3-ASR Technical Report, arXiv 2601.21337, Table 2](https://arxiv.org/html/2601.21337)
- [P] Supports "single speech no longer than 20 minutes" and unified streaming/offline inference; streaming evaluation uses 2-second chunks, 5-token fallback, last four chunks unfixed. — [arXiv 2601.21337 §2.3, §4.5](https://arxiv.org/html/2601.21337)
- [P] Toolkit: batch inference with `max_inference_batch_size` (examples 32 transformers / 128 vLLM); "streaming inference is only available with the vLLM backend … does not support batch inference or returning timestamps"; FlashAttention 2 recommended "to reduce GPU memory usage"; `gpu_memory_utilization` examples 0.65–0.9. — [Qwen3-ASR README](https://github.com/QwenLM/Qwen3-ASR)
- [P] Community C port (antirez/qwen-asr), CUDA build on DGX Spark (GB10), Q8, 20-s segments, model preloaded: 0.6B transcribes 11.0 s in 0.264 s, 45.0 s in 0.626 s, 119.3 s in 2.498 s; fresh-process 11 s clip 2.44 s including load. The CPU build defaults to full offline decode; "for long files, memory/time grow with sequence length". — [antirez/qwen-asr README](https://github.com/antirez/qwen-asr)

**Separators**
- [P] MossFormer2 RTF on WSJ0-2mix test (8 kHz), single V100-32GB: MossFormer2 55.7 M params **0.053**; MossFormer2 (S) 37.8 M 0.036; MossFormer 42.1 M 0.038; MossFormer (S) 25.3 M 0.025. — [arXiv 2312.11825, Table 1](https://arxiv.org/html/2312.11825)
- [P] TIGER efficiency table (inference of one second of 16 kHz audio; GPU = RTX 4090, CPU = Intel Xeon Gold 6326). Params M / MACs G/s / CPU ms / GPU ms / GPU inference memory MB:
  - Conv-TasNet 5.62 / 7.19 / 64.21 / 13.17 / 28.78
  - DualPathRNN 2.72 / 45.01 / 723.13 / 30.38 / 298.09
  - SudoRM-RF1.0x 2.72 / 4.65 / 104.32 / 20.66 / 24.42
  - A-FRCNN-16 6.13 / 81.28 / 478.58 / 82.65 / 163.82
  - TDANet Large 2.33 / 9.19 / 434.44 / 74.27 / 136.96
  - BSRNN 25.97 / 98.70 / 897.27 / 78.27 / 130.24
  - **TF-GridNet 14.43 / 323.75 / 2019.60 / 94.30 / 491.73**
  - TIGER small 0.82 / 7.65 / 351.15 / 42.38 / 122.23
  - TIGER large 0.82 / 15.27 / 765.47 / 74.51 / 122.23
  — [TIGER, arXiv 2410.01469, Table 3](https://arxiv.org/html/2410.01469)
- [P] SepFormer: 26 M params; "not only faster but also less memory demanding than DPTNet, DPRNN, and Wavesplit" on an RTX8000-48GB for 1–5 s 8 kHz inputs (figure only; numeric ms/GB Not Reported in text); "same behavior using the CPU". — [arXiv 2010.13154 §4.4](https://arxiv.org/html/2010.13154)
- [P] TF-GridNet largest config: 231.1 GMAC/s; 4,918.8 MB GPU memory for a forward pass on a 4-s segment. — [arXiv 2211.12433](https://arxiv.org/html/2211.12433)
- [P] PixIT pipeline: WavLM-large + DPRNN on 5 s windows with 500 ms step (i.e. each audio instant is processed by ~10 overlapping windows), plus ECAPA embedding extraction and clustering; RTF/VRAM Not Reported. — [arXiv 2403.02288](https://arxiv.org/html/2403.02288)
- [P] CSS windowing in published systems: 2.4 s window / 0.8 s hop (Microsoft); 3 s (NOTSOFAR per USTC description); 4 s / 3 s shift (TF-GridNet LibriCSS). — [arXiv 2506.14204](https://arxiv.org/html/2506.14204v1), [arXiv 2409.02041v2](https://arxiv.org/html/2409.02041v2), [arXiv 2309.08454](https://arxiv.org/html/2309.08454)

**Latency / streaming**
- [P] CSS inherent latency in LibriCSS configs: 0.8–1.2 s (look-ahead sub-window). — [arXiv 2001.11482 Table 2](https://arxiv.org/html/2001.11482)
- [P] Microsoft's streaming CT-tSOT with CSS front-end is listed at 160 ms latency (ASR-side algorithmic latency) vs 5 s for the offline second pass. — [arXiv 2506.14204 Table 1](https://arxiv.org/html/2506.14204v1)

**Single SOT-style pass (published comparators)**
- [P] Cost: a single-pass SOT fine-tune of the same recognizer would run at the base recognizer cost, RTF 0.00923 at concurrency 1 (base model figure; no SOT-specific Qwen3-ASR timing is published). — [arXiv 2601.21337](https://arxiv.org/html/2601.21337)
- [P] Accuracy on Libri2Mix test: single-pass SOT-family 23.3 / 17.1 / 9.0 / 7.0 % vs separator cascade 4.9–5.2 % (5.0 % with Whisper large-v3). — [arXiv 2503.17886 Table 1](https://arxiv.org/html/2503.17886v1)

### Inferences
- Throughput arithmetic from the Qwen3 table (mine): at concurrency 2 the server transcribes 177.94 s of stream audio per second, i.e. ≈ 89 s of two-stream mixture per second; at concurrency 4 (311.53) ≈ 104 s of three-stream mixture per second. ASR adds ≈ 0.011–0.013 RTF per stream-set versus 0.009 for a single pass.
- Adding published RTFs across different hardware is not rigorous, but the orders of magnitude indicate total cascade cost ≈ separator cost: e.g. MossFormer2 0.053 (V100, 8 kHz) vs ASR ≈ 0.01. A 16 kHz MossFormer2 RTF is Not Reported; 16 kHz doubles the sample count.
- TF-GridNet-class quality costs ≈ 0.094 s GPU per second of audio on an RTX 4090 and is > 2× slower than real time on a Xeon CPU core set-up as measured by TIGER; Conv-TasNet (64 ms/s CPU) or SudoRM-RF (104 ms/s CPU) are the published CPU-feasible options.
- Overlap gating (§1b) reduces separator duty cycle to the overlapped fraction (AMI 19 %, AliMeeting 42 % per Sun et al.), at the price of OSD misses.
- Streaming: Qwen3-ASR streaming does not batch, so N streams need N concurrent streaming sessions; end-to-end latency ≥ CSS look-ahead (0.4–1.2 s published) + 2 s ASR chunk. No published system demonstrates this combination.
- VRAM planning from published parts only: TF-GridNet ≈ 0.5 GB inference (1 s input) to ≈ 4.9 GB (4 s forward, largest config); Qwen3-ASR-0.6B VRAM is Not Reported.

### Gaps
- Qwen3-ASR-0.6B VRAM footprint and the benchmark GPU model: Not Reported in the tech report or README.
- CPU RTF for Qwen3-ASR-0.6B from an official source: Not Reported; I did not extract CPU timings from community ports (only CUDA timings were read).
- RTF/VRAM for `MossFormer2_SS_16K`, SepFormer at 16 kHz, TF-Locoformer, and the pyannote PixIT pipeline: Not Reported in sources read.
- No published end-to-end latency measurement for a separator + Speech-LLM streaming cascade.

---

## 5. Failure modes and published mitigations

### Takeaway
The documented failure modes are: degradation on non-overlapped speech, leakage of the other talker into a nominally silent stream (which turns into insertions/hallucinations in generative ASR), sample-rate mismatch of public checkpoints (most are 8 kHz), fixed output count, and chunk-permutation errors in long-form audio. Published mitigations are activity gating/leakage removal, overlap gating, windowed CSS with overlap-based permutation alignment, N = 3 output masks, and matched back-end training.

### Cited Findings
- **Separator on single-speaker audio.** [P] LibriCSS: single-channel separation worsened WER at 0 % overlap (0S 11.8 → 12.7; 0L 11.7 → 12.1). — [arXiv 2001.11482](https://arxiv.org/html/2001.11482). [P] CSS front-end before SOT: 0L 7.76 → 8.18, 0S 7.33 → 8.76. — [arXiv 2506.14204](https://arxiv.org/html/2506.14204v1). [P] TF-GridNet CSS training used "fully overlapping data and single-speaker utterances" so the model learns the one-speaker case. — [arXiv 2309.08454](https://arxiv.org/html/2309.08454)
- **Residual/leaked second stream → insertions.** [P] PixIT without WavLM and without leakage removal: insertion rate 15.6 % (cpWER 50.1) because leakage "pass[es] through the VAD component of WhisperX"; diarization-driven zeroing cuts insertions to 1.7 %; with WavLM 3.8 % → 1.2 %. The Δt context margin is tuned on dev data "to give downstream ASR systems additional context". — [arXiv 2403.02288 Table 5](https://arxiv.org/html/2403.02288). [P] TF-GridNet CSS pipeline applies "a simple energy-based voice activity detection" per output channel before ASR. — [arXiv 2309.08454](https://arxiv.org/html/2309.08454)
- **ASR behaviour on empty/non-speech input.** [P] Qwen3-ASR's SFT stage "also utilizes non-speech data" and defines "Output style 2: For no speech detected" = `language None<asr_text>` with empty text; an RL stage with 30 % "functional data" targets "transcribing stability in complex environments". — [arXiv 2601.21337 §2.2](https://arxiv.org/html/2601.21337). [P] For comparison, Whisper hallucinates on non-speech: "roughly 1 % of transcriptions" contained hallucinated phrases, concentrated in audio with long non-vocal durations — [Careless Whisper, arXiv 2402.08021](https://arxiv.org/html/2402.08021); 3 of 20 decoder heads account for > 75 % of non-speech hallucinations and fine-tuning them gives > 80 % reduction with < 0.1 % WER loss — [Calm-Whisper, arXiv 2505.12969](https://arxiv.org/html/2505.12969)
- **8 kHz vs 16 kHz checkpoints.** [P] SpeechBrain `sepformer-wsj02mix` (22.4 dB), `sepformer-wsj03mix` (19.8 dB), `sepformer-libri2mix` (20.6 dB), `sepformer-libri3mix` (19.0 dB table / 19.8 dB text), `resepformer-wsj02mix` (18.6 dB) all state "expects input recordings sampled at 8kHz"; only `sepformer-whamr16k` (13.5 dB SI-SNRi, noisy-reverberant) is 16 kHz. — [sepformer-libri2mix](https://huggingface.co/speechbrain/sepformer-libri2mix), [sepformer-wsj03mix](https://huggingface.co/speechbrain/sepformer-wsj03mix), [sepformer-libri3mix](https://huggingface.co/speechbrain/sepformer-libri3mix), [sepformer-whamr16k](https://huggingface.co/speechbrain/sepformer-whamr16k). [P] 16 kHz options: `MossFormer2_SS_16K`, pyannote PixIT (16 kHz in/out), TIGER (16 kHz), Asteroid `ConvTasNet_Libri2Mix_sepclean_16k`. — [clearvoice README](https://github.com/modelscope/ClearerVoice-Studio/blob/main/clearvoice/README.md), [pyannote card](https://huggingface.co/pyannote/speech-separation-ami-1.0), [arXiv 2410.01469](https://arxiv.org/html/2410.01469), [JorisCos card](https://huggingface.co/JorisCos/ConvTasNet_Libri2Mix_sepclean_16k). Qwen3-ASR input is resampled to 16 kHz in its demo pipeline. — [Qwen3-ASR README](https://github.com/QwenLM/Qwen3-ASR)
- **Loudness mismatch.** [P] WSJ0-2mix relative levels are only 0–5 dB — [arXiv 2010.13154](https://arxiv.org/html/2010.13154); in real AMI overlap the two sources had average input SI-SNR 0.5 dB and −9.2 dB, and improvement differs strongly between dominant and weak speaker — [arXiv 2110.10739](https://arxiv.org/html/2110.10739); PixIT notes ASR on mixtures "may miss the quieter speakers utterances during overlap" (deletions) — [arXiv 2403.02288](https://arxiv.org/html/2403.02288).
- **More speakers than outputs.** [P] CSS relies on the assumption that a short window (2.4–4 s) contains at most N simultaneous speakers; NOTSOFAR uses N = 3 "which can support up to 3 speakers overlapping at the same time"; TF-GridNet CSS uses 2 outputs with a 4 s window; PixIT sets K_max = 3 because < 1 % of 5 s windows have more than three active speakers. — [arXiv 2401.08887](https://arxiv.org/html/2401.08887), [arXiv 2309.08454](https://arxiv.org/html/2309.08454), [arXiv 2403.02288](https://arxiv.org/html/2403.02288). [P] MixIT-style models can over-separate (more outputs than sources) — [arXiv 2006.12701](https://arxiv.org/html/2006.12701).
- **Long-form chunk permutation.** [P] Published solutions: (i) align adjacent windows by minimum MSE over shared frames across all permutations (LibriCSS, NOTSOFAR, TF-GridNet CSS); (ii) speaker-embedding clustering of per-chunk sources (PixIT). — [arXiv 2401.08887](https://arxiv.org/html/2401.08887), [arXiv 2403.02288](https://arxiv.org/html/2403.02288)
- **Duplicated words across streams / cross-talk.** [P] Leakage analysis on LibriCSS with TF-GridNet: 1-best WER on the separated primary channel 5.9 % (4-gram) / 4.4 % (Transformer LM) vs 3.1 / 2.1 % on oracle clean audio; the paper measures "coincidence rates" of the cross-talker's words appearing in the primary channel. — [arXiv 2309.08454, Tables 4–5](https://arxiv.org/html/2309.08454)
- **Merging transcripts.** [P] PixIT's speaker-agnostic transcript: "concatenating the ASR predictions from all the long-form sources and then sorting the words by start time" (requires word timestamps; WhisperX forced alignment was used). — [arXiv 2403.02288 §3.2](https://arxiv.org/html/2403.02288). [P] Qwen3 ecosystem has Qwen3-ForcedAligner-0.6B (RTF ≈ 0.001) for timestamps. — [arXiv 2601.21337](https://arxiv.org/html/2601.21337)
- **Simulated-vs-real gap.** [P] "Supervised … systems … reliance on synthetic data, leading to poor real-world generalization" (PixIT); models trained on Libri2Mix/LRS2-2Mix do worse on real recordings than EchoSet-trained ones (TIGER); ESPnet-SE++ observed enhanced-speech WER worse on real data unless jointly optimised. — [arXiv 2403.02288](https://arxiv.org/html/2403.02288), [arXiv 2410.01469](https://arxiv.org/html/2410.01469), [arXiv 2207.09514](https://arxiv.org/html/2207.09514)

### Inferences
- A per-stream activity gate (energy VAD or diarization/segmentation activity) before ASR is the single best-evidenced mitigation; with Qwen3-ASR the `language None` empty output provides a second line of defence, but its reliability on *low-level residual speech* (as opposed to true non-speech) is not documented, and residual speech is exactly what a leaky separator produces.
- When overlap gating is used, text from the gated (separated) region and the surrounding single-pass regions must be merged; timestamps from the forced aligner make the PixIT "sort by start time" merge applicable. De-duplication of words that appear in both streams has no published algorithm in the sources read; it would have to be designed (e.g. time-aligned duplicate suppression) and validated in-house.
- Resampling 16 kHz audio to 8 kHz for an 8 kHz separator and back discards the 4–8 kHz band the recognizer was trained to use; no source quantifies the WER cost for a Speech-LLM, so 16 kHz-native separators are the lower-risk choice.
- Training mixtures should cover a wider level range than WSJ0-2mix's 0–5 dB given the −9 dB weak-speaker condition observed on AMI.

### Gaps
- No measurement of Qwen3-ASR hallucination/insertion rate on separator residuals or near-silent streams.
- No published de-duplication/merging algorithm for per-stream transcripts in a speaker-agnostic setting beyond time-sorted concatenation.
- No published WER impact of 8 kHz separation followed by 16 kHz ASR.

---

## 6. Fine-tuning the ASR on separator outputs (matched training)

### Takeaway
Matched training of the back-end on separated audio is a documented and consistently beneficial step for classical and Whisper back-ends (roughly 7–18 % relative WER reduction in the two controlled studies read), and top CHiME-8 systems fine-tuned Whisper large-v2/v3 on separated/enhanced data mixed with clean LibriSpeech; no such recipe exists yet for a Speech-LLM, and one study argues a clean-trained back-end is preferable once the separator is strong.

### Cited Findings
- [P] LibriCSS, TF-GridNet CSS → hybrid AM: fine-tuning the AM "using separated audio tracks which are obtained by applying the previously trained separators to the simulated training data" improves ORC-WER 7.9 → 7.2 (4-gram) and 6.4 → 5.8 (Transformer LM); with a weak BLSTM separator 21.6 → 19.9 / 19.4 → 17.9. — [arXiv 2309.08454 Table 2](https://arxiv.org/html/2309.08454)
- [P] SMS-WSJ 1-ch: AM trained on TF-CrossNet-separated training data gets **6.26 %** vs 7.60 % for the clean (direct-path) AM and 8.13 % task-standard; authors' caveat: "these backends depend on the data from pre-trained TF-CrossNets. When a better frontend is available, retraining these backends is required … making these models less practical". — [arXiv 2503.17886 Table 4](https://arxiv.org/html/2503.17886v1)
- [P] Counter-position: "when a powerful speaker separation frontend is available, training the ASR backend on clean speech is an effective option"; on Libri2Mix all back-ends (clean, noise-dependent, noise-independent, Whisper large-v3) land within 4.9–5.2 % test WER. — [arXiv 2503.17886 §4.1](https://arxiv.org/html/2503.17886v1)
- [P] Whisper matched fine-tuning in practice: USTC's NOTSOFAR-1 "enhanced" Whisper large-v2/v3 models were fine-tuned on front-end-processed target data — per split 14 h + 16 h + 10 h each of multi-channel GSS output, GSS with timestamps, neural-separation ("MC NN") output and "MC ch0 NN" output — **plus 960 h LibriSpeech**; posterior fusion of three such models was used. — [arXiv 2409.02041v2 Table 1](https://arxiv.org/html/2409.02041v2)
- [P] Joint optimisation precedent: ESPnet-SE++ fine-tunes SE + ASR with both losses; mixture-encoder work lists several studies showing "joint fine-tuning of the speech separation- and ASR-module improves the performance of such modular systems". — [arXiv 2207.09514](https://arxiv.org/html/2207.09514), [arXiv 2306.12173](https://arxiv.org/html/2306.12173)
- [P] PixIT deliberately used ASR "off-the-shelf with no fine-tuning required". — [arXiv 2403.02288](https://arxiv.org/html/2403.02288)
- [P] Qwen3-ASR ships an official SFT script (`finetuning/qwen3_asr_sft.py`) taking JSONL audio–text pairs, single- or multi-GPU via `torchrun`, example `--batch_size 32`; the README section read does not mention LoRA. A community LoRA repository exists (MeiLiMaoA/Qwen3-ASR-LoRA) — existence verified, contents and results not reviewed. — [Qwen3-ASR finetuning README](https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning), [Qwen3-ASR-LoRA](https://github.com/MeiLiMaoA/Qwen3-ASR-LoRA)

### Inferences
- A matched-training set can be produced with no new annotation: run the trained separator over the same dynamically mixed single-speaker data used to train it and pair each output stream with the source utterance's transcript (the procedure in 2309.08454).
- Risk to single-speaker performance is manageable by the published precedent of mixing large amounts of unprocessed clean data into the fine-tuning set (USTC: 960 h clean vs ≈ 160 h processed). A LoRA adapter that is only enabled for separated streams would isolate the base model entirely; this is a design option, not something a source demonstrates for Qwen3-ASR.
- The 2503.17886 caveat (back-end must be re-trained when the separator changes) argues for an adapter-style, cheaply re-trainable matched component rather than full fine-tuning.

### Gaps
- No published recipe or result for fine-tuning any Speech-LLM (Qwen3-ASR, Qwen2-Audio, SALMONN, etc.) on separator outputs.
- No published measurement of single-speaker WER change after matched fine-tuning of Whisper on separated audio (USTC does not report it).
- LoRA suitability for Qwen3-ASR on separated audio: no evidence either way.

---

## 7. Evaluation when speaker identity is not needed

### Takeaway
MeetEval is the standard open tool; for a speaker-agnostic cascade the appropriate primary metric is ORC-WER (or tcORC-WER when timestamps exist), feeding each separator output stream as a hypothesis "speaker"/stream. cpWER additionally penalises stream-permutation errors and, in the current MeetEval definition, redundant output streams — useful as a diagnostic for stitching but stricter than the requirement.

### Cited Findings
- [P] MeetEval CLI: `meeteval-wer cpwer|orcwer|greedy_orcwer|mimower|tcmimower|tcpwer|tcorcwer|greedy_tcorcwer|greedy_dicpwer -r ref.stm -h hyp.stm` (time-constrained variants take `--collar 5`); in STM/CTM the speaker field is the "ID of the speaker or system output stream/channel (not microphone channel)"; multiple per-channel CTM hypotheses can be passed (`-h hyp1.ctm -h hyp2.ctm`); `start_time`/`end_time` only "required for time-constrained metrics". — [MeetEval README](https://github.com/fgnt/meeteval)
- [P] NOTSOFAR-1 ranks by speaker-attributed tcpWER and uses "the speaker-agnostic tcORC-WER … as a supplementary metric"; baseline single-channel 46.8 tcpWER vs 38.5 tcORC-WER. — [NOTSOFAR1-Challenge README](https://github.com/microsoft/NOTSOFAR1-Challenge), [arXiv 2401.08887](https://arxiv.org/html/2401.08887)
- [P] Metrics used by published cascades: PixIT — cpWER/cpCER (main) and Kaldi/asclite utterance-wise uWER for the speaker-agnostic view; TF-GridNet CSS — ORC-WER on LibriCSS; Microsoft CSS+SOT — speaker-agnostic SAgWER via NIST asclite; Yang et al. — permutation-minimised concatenated WER on Libri2Mix and speaker-attributed WER on LibriCSS; LibriCSS original — utterance-wise and continuous-input WER per overlap ratio (0S/0L/10/20/30/40). — [arXiv 2403.02288](https://arxiv.org/html/2403.02288), [arXiv 2309.08454](https://arxiv.org/html/2309.08454), [arXiv 2506.14204](https://arxiv.org/html/2506.14204v1), [arXiv 2503.17886](https://arxiv.org/html/2503.17886v1), [arXiv 2001.11482](https://arxiv.org/html/2001.11482)
- [P] cpWER definition pitfall: the earlier definition "does not penalize redundant hypothesis speaker channels"; the later MeetEval definition penalises over-estimation "by adding dummy channels to the reference"; PixIT reports both and notes numbers across papers "were calculated in different ways". 2309.08454 likewise warns its comparison table mixes asclite WER, cpWER with oracle assignment and ORC-WER. — [arXiv 2403.02288 App. A](https://arxiv.org/html/2403.02288), [arXiv 2309.08454 Table 3](https://arxiv.org/html/2309.08454)
- [P] MS-Bench decomposes WER into lexical (E_lex) and segmentation (E_seg) parts and stratifies by overlap ratio (r_ov = 0 vs ≥ 0.5 within the same recording). — [arXiv 2610.25824](https://arxiv.org/abs/2610.25824)
- [P] Reporting sub/del/ins separately is informative for cascades: separation mainly reduces deletions, leakage shows up as insertions. — [arXiv 2403.02288 Tables 2, 3, 5](https://arxiv.org/html/2403.02288)

### Inferences
- Recommended scoring set (design suggestion): ORC-WER/ORC-CER as the headline (streams as pseudo-speakers, no penalty for which stream carries which utterance), with sub/del/ins breakdown; a per-overlap-ratio breakdown in the LibriCSS style including a 0 % overlap condition to expose single-speaker regressions; and plain WER on a single-speaker test set as a guard-rail.
- For mixed Taigi/Mandarin/English text, character- vs word-level tokenisation must be fixed before scoring (PixIT used cpCER for Mandarin); MeetEval operates on whatever tokens are supplied.
- An in-house test set can be simulated from held-out single-speaker utterances (Libri2Mix-style full overlap plus LibriCSS-style partial overlap), since no overlapped Taigi corpus was found.

### Gaps
- I did not re-read the MeetEval paper (arXiv 2211.16112) text for the formal ORC-WER/MIMO-WER definitions in this session; the description above relies on the README and on how NOTSOFAR/TF-GridNet papers use the metric.
- No public overlapped-speech benchmark for Taiwanese Hokkien was found.

---

## 8. Licences affecting commercial deployment

### Takeaway
Code/toolkit licences are permissive (Apache-2.0 / MIT), but three issues need legal review: gated pyannote downloads (MIT, registration required), checkpoints trained on LDC WSJ0 (model card says Apache-2.0, but LDC data terms restrict commercial use for non-members), and non-commercial or share-alike model licences (Sortformer v1 CC-BY-NC-4.0; Asteroid LibriMix ConvTasNet CC-BY-SA-4.0).

### Cited Findings
- [P] Qwen3-ASR: Apache-2.0 (per task context; repository LICENSE present). — [Qwen3-ASR repo](https://github.com/QwenLM/Qwen3-ASR)
- [P] pyannote `segmentation-3.0`, `segmentation`, `speaker-diarization-3.1`, `separation-ami-1.0`, `speech-separation-ami-1.0`: `license: mit`, `gated: auto`, form asks company/website; cards advertise the commercial pyannoteAI service for production. `speaker-diarization-community-1`: `cc-by-4.0`, gated. — [HF API speech-separation-ami-1.0](https://huggingface.co/api/models/pyannote/speech-separation-ami-1.0), [HF API community-1](https://huggingface.co/api/models/pyannote/speaker-diarization-community-1)
- [P] PixIT checkpoint was trained on AMI (corpus licence not checked here) and embeds WavLM-large features. — [model card](https://huggingface.co/pyannote/speech-separation-ami-1.0), [arXiv 2403.02288](https://arxiv.org/html/2403.02288)
- [P] ClearerVoice `alibabasglab/MossFormer2_SS_16K`: `license: apache-2.0`, not gated. — [HF API](https://huggingface.co/api/models/alibabasglab/MossFormer2_SS_16K)
- [P] SpeechBrain SepFormer cards (`sepformer-wsj02mix`, `-wsj03mix`, `-libri2mix`, `-libri3mix`, `-whamr16k`, `resepformer-wsj02mix`): `license: apache-2.0`. The WSJ0/WHAMR-based ones derive from LDC WSJ0 (LDC93S6A); ClearerVoice's own docs say "We assume you have purchased WSJ0". — [sepformer-wsj02mix](https://huggingface.co/speechbrain/sepformer-wsj02mix), [ClearerVoice train README](https://github.com/modelscope/ClearerVoice-Studio/blob/main/train/speech_separation/README.md)
- [S] LDC general terms: non-member licensees may use LDC data "for noncommercial linguistic research and education only"; a for-profit must join as a for-profit member to gain commercial rights; corpus-specific agreements can supersede. Whether a model trained on WSJ0 is covered is not addressed in the snippets. — [LDC licensing page](https://www.ldc.upenn.edu/node/33)
- [P] LibriSpeech-derived checkpoints avoid LDC: Libri2Mix/Libri3Mix are built from LibriSpeech + WHAM! noise. — [arXiv 2005.11262](https://arxiv.org/html/2005.11262). But `JorisCos/ConvTasNet_Libri2Mix_sepclean_16k` is `cc-by-sa-4.0` (share-alike). — [model card](https://huggingface.co/JorisCos/ConvTasNet_Libri2Mix_sepclean_16k)
- [P] TF-Locoformer code: Apache-2.0 (MERL). TIGER HF checkpoint: `apache-2.0`. — [tf-locoformer README](https://github.com/merlresearch/tf-locoformer), [HF API TIGER-speech](https://huggingface.co/api/models/JusperLee/TIGER-speech)
- [P] NVIDIA: `diar_sortformer_4spk-v1` = `cc-by-nc-4.0` (non-commercial); `diar_streaming_sortformer_4spk-v2` card metadata = `cc-by-4.0`. — [HF API v1](https://huggingface.co/api/models/nvidia/diar_sortformer_4spk-v1), [HF API v2](https://huggingface.co/api/models/nvidia/diar_streaming_sortformer_4spk-v2)
- [P] NOTSOFAR datasets: CC BY 4.0. — [NOTSOFAR1-Challenge README](https://github.com/microsoft/NOTSOFAR1-Challenge)

### Inferences
- Training the separator in-house on the team's own single-speaker corpora (plus LibriSpeech-derived data if English coverage is wanted) removes the WSJ0/LDC and share-alike questions entirely and is the cleanest commercial path; ClearerVoice (Apache-2.0 code and checkpoint) is the lowest-friction starting point for 16 kHz.
- WHAM! noise (used by LibriMix and ClearerVoice data prep) has its own licence, which I did not check; it should be verified before using it for commercial training data.

### Gaps
- The WSJ0 (LDC93S6A) corpus-specific agreement text was not retrieved; the LDC statements above are snippet-level only.
- I could not confirm any current NVIDIA model relevant to this pipeline that is under the "NVIDIA Open Model License"; the two Sortformer cards checked today show CC licences in their metadata. If a specific NVIDIA model is under consideration, its card must be checked individually.
- AMI corpus and WHAM! noise licence terms were not verified in this session.
- Licence of the upstream training data for `MossFormer2_SS_16K`: Not Reported in the READMEs read.
