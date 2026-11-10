# Implementation and Deployment Cost of Overlapped-Speech ASR Approaches for Qwen3-ASR-0.6B

> **Source tags.** Every finding carries one of two tags:
> - **[D]** = I read the primary page or PDF directly (GitHub, arXiv, Hugging Face, ISCA archive).
> - **[S]** = the figure comes from a search-engine extract of the cited page; the page itself was not opened. Treat as "reported, not yet checked against the primary source".
>
> A second pass (after arxiv.org, huggingface.co and isca-archive.org became reachable) verified the Qwen3-ASR tech report, the NVIDIA and MOSS model cards, and the SOP, Sidecar, t-SOT, self-speaker-adaptation, Wu et al. 2021 and Shi et al. 2026 papers. What is still [S] is listed at the end of each Gaps section. No number below is estimated; anything not found is marked **Not Reported**.

## 1. Qwen3-ASR-0.6B baseline facts (params, speed, VRAM, streaming, vLLM, CPU/quantization, max length, license)

### Takeaway
The baseline is an Apache-2.0 model built from a 180M-parameter AuT encoder (12.5 Hz output), a projector and Qwen3-0.6B, with official vLLM serving and an official fine-tuning script but no official quantized or CPU build. On GPU it reports TTFT 92 ms at concurrency 1 and RTF 0.064 (2,000 audio-seconds per second) at concurrency 128; CPU-only use exists only through community ports, with results ranging from several times faster than real time to slower than real time in streaming mode.

### Cited Findings

**Architecture and parameters**
- [D] "Qwen3-ASR-0.6B is built with Qwen3-0.6B, a projector and an AuT encoder with 180M parameters and a hidden size of 896." The 1.7B model uses Qwen3-1.7B, a projector and a 300M-parameter AuT encoder with hidden size 1024. — [arXiv 2601.21337v2](https://arxiv.org/html/2601.21337v2)
- [D] AuT is an attention-encoder-decoder ASR model that downsamples 128-dimensional Fbank features 8x, giving a 12.5 Hz token rate; it uses a dynamic flash-attention window from 1 s to 8 s, which is what allows both streaming and offline inference. — [arXiv 2601.21337v2](https://arxiv.org/html/2601.21337v2)
- [D] AuT pre-training used approximately 40 million hours of pseudo-labeled ASR data, mostly Chinese and English. — [arXiv 2601.21337v2](https://arxiv.org/html/2601.21337v2)
- [D] Layer-level structure from an independent C re-implementation: 0.6B has 18 encoder layers (dim 896), 28 decoder layers (dim 1024), 16/8 Q/KV heads, vocab 151,936, BF16 weights; a 3-layer Conv2D stem (480 channels, 8x time downsampling); a Linear-GELU-Linear projector. — [antirez/qwen-asr](https://github.com/antirez/qwen-asr)
- Projector parameter count: **Not Reported** (the tech report names the projector without a size).

**Speed (GPU), Table 2 of the tech report**
- [D] Setup: vLLM v0.14.0, CUDA Graph enabled, bfloat16, input audio of about 2 minutes, "a single typical computing resource" (GPU model not named). Two settings: offline batch generation and online asynchronous serving through vLLM Serve. — [arXiv 2601.21337v2](https://arxiv.org/html/2601.21337v2)
- [D] Qwen3-ASR-0.6B rows (throughput = audio-seconds per second):

| Conc. | Offline RTF | Offline throughput | Online TTFT avg (ms) | Online TTFT p95 (ms) | Online RTF | Online throughput |
|---|---|---|---|---|---|---|
| 1 | 0.00923 | 108.34 | 92 | 105 | 0.00940 | 106.38 |
| 2 | 0.01124 | 177.94 | 103 | 168 | 0.01108 | 180.51 |
| 4 | 0.01284 | 311.53 | 132 | 203 | 0.01224 | 326.80 |
| 8 | 0.01600 | 500.00 | 228 | 417 | 0.01472 | 543.48 |
| 16 | 0.02384 | 671.14 | 459 | 882 | 0.01936 | 826.45 |
| 32 | 0.03808 | 840.34 | 820 | 1575 | 0.02912 | 1098.90 |
| 64 | 0.06336 | 1010.10 | 1631 | 3196 | 0.04352 | 1470.59 |
| 128 | 0.11264 | 1136.36 | 3210 | 6195 | 0.06400 | 2000.00 |
| 256 | 0.21504 | 1190.48 | - | - | - | - |
| 512 | 0.44544 | 1149.43 | - | - | - | - |

  — [arXiv 2601.21337v2](https://arxiv.org/html/2601.21337v2)
- [D] Correction to a common misreading: the 92 ms TTFT is the concurrency-1 row. At concurrency 128, where RTF is 0.064 and throughput 2,000, average TTFT is 3,210 ms. — [arXiv 2601.21337v2](https://arxiv.org/html/2601.21337v2)
- [D] For comparison, Qwen3-ASR-1.7B at concurrency 1: TTFT 102 ms, online RTF 0.01483; at concurrency 128: TTFT 3,392 ms, online RTF 0.10496, throughput 1219.51. — [arXiv 2601.21337v2](https://arxiv.org/html/2601.21337v2)

**VRAM**
- [D] The official README gives no VRAM figure. It shows `gpu_memory_utilization` examples (0.7-0.9), recommends FlashAttention 2 to reduce GPU memory, and says a smaller `max_inference_batch_size` avoids OOM. The tech report gives no memory figure either. — [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR); [arXiv 2601.21337v2](https://arxiv.org/html/2601.21337v2)
- [D] vLLM 0.14.0 startup log for Qwen3-ASR-0.6B posted by a user: "Model loading took 1.53 GiB memory"; CUDA graph capture "took 0.50 GiB"; with `--gpu-memory-utilization 0.2` and `--max-model-len 10000` the KV cache got 1.74 GiB (16,272 tokens). The user asked for recommended VRAM; no maintainer answered and the issue was closed as "not planned". — [Qwen3-ASR issue #20](https://github.com/QwenLM/Qwen3-ASR/issues/20)
- [D] A community member (not a maintainer) in the same issue claims "consistently < 2GB VRAM usage under load" with their own vLLM fork. Unverified claim. — [Qwen3-ASR issue #20](https://github.com/QwenLM/Qwen3-ASR/issues/20)

**Streaming mechanics**
- [D] One model serves both offline and streaming. Streaming is available only with the vLLM backend, and "streaming inference does not support batch inference or returning timestamps". — [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- [D] The report's streaming evaluation uses "a 2-second chunk size, a 5-token fallback, and keeping the last four chunks unfixed". — [arXiv 2601.21337v2](https://arxiv.org/html/2601.21337v2)
- [D] An independent re-implementation describes the same scheme: 2-second chunks; each step re-feeds the accumulated audio with the previous output minus a 5-token rollback suffix as decoder prefix; encoder windows (8 s default) are cached once complete; each chunk's decode is capped at 32 new tokens by default. — [antirez/qwen-asr](https://github.com/antirez/qwen-asr)
- [D] Streaming costs accuracy: 1.7B averages 3.33 WER streaming vs 2.69 offline on the README's table. — [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)

**vLLM, max length, license, languages**
- [D] vLLM "officially provides day-0 model support"; `qwen-asr-serve` wraps `vllm serve`; OpenAI-compatible chat and transcription endpoints; official Docker image `qwenllm/qwen3-asr`. — [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- [D] Both models "naturally support single speech no longer than 20 minutes". — [arXiv 2601.21337v2](https://arxiv.org/html/2601.21337v2)
- [D] License Apache-2.0; 30 languages plus 22 Chinese dialects. — [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- [S] vLLM did not support LoRA adapters for `Qwen3ASRForConditionalGeneration` as of feature request #37223; a later PR (#50793) fixes audio-tower Q/K/V LoRA adapters being silently skipped. Merge status not checked. — [vLLM issue #37223](https://github.com/vllm-project/vllm/issues/37223); [vLLM PR #50793](https://github.com/vllm-project/vllm/pull/50793)

**Quantization / GGUF / ONNX / CPU (all community, none official)**
- [D] The official README mentions no quantization, GGUF, ONNX or CPU path. — [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- [D] antirez/qwen-asr (pure C, MIT, BLAS only on CPU), Apple M3 Max, 0.6B: 11.0 s clip in 1.4 s (7.99x real time); 45 s clip with 30 s segments in 3.4 s (13.38x); 88.9 s in 13.1 s (6.78x). Streaming mode on a 45 s clip: 9.6 s (4.69x) with encoder cache, 22.0 s (2.05x) without. On a 135 s file `--stream` took 141.3 s (0.96x, i.e. not real time) vs 14.0 s segmented. — [antirez/qwen-asr](https://github.com/antirez/qwen-asr)
- [D] Same port, memory on CPU: 0.6B static total about 2.770 GiB; measured peak RSS at 119 s audio about 3.25 GiB (single pass) and 2.79 GiB (20 s segments). — [antirez/qwen-asr](https://github.com/antirez/qwen-asr)
- [D] HaujetZhao/Qwen3-ASR-GGUF splits the model into ONNX encoder + GGUF decoder (llama.cpp, Q4_K). Its published benchmark is for the **1.7B** model on 50.2 s Chinese audio: RTF 0.390 on CPU (CPU model not named) and 0.052 on an RTX 5050 laptop; on CPU, LLM prefill was 10.7 s and generation 7.0 s (27.1 tok/s). — [HaujetZhao/Qwen3-ASR-GGUF](https://github.com/HaujetZhao/Qwen3-ASR-GGUF)
- [S] An arXiv on-device streaming benchmark reports Qwen3-ASR at RTFx 0.49 on CPU in chunked mode (slower than real time), while ONNX builds of other models exceeded 6x real time. Hardware and model size not captured. — [arXiv 2604.14493](https://arxiv.org/abs/2604.14493)
- [S] An Android benchmark lists Qwen3-ASR 0.6B at RTF 0.53 with ONNX Runtime INT8 and 0.17 with a pure C/NEON build; Whisper Small on sherpa-onnx is 0.41 in the same table. Device not captured. — [VoicePing benchmark](https://voiceping-landingpage-prod.pages.dev/en/blog/research-offline-speech-transcription-benchmark/)
- [S] Other community builds: CrispASR GGUF (Q4_K, "faster than realtime on a 4-core CPU", file size listed as 676 MB or 601.8 MB depending on mirror); predict-woo/qwen3-asr.cpp (GGML; F16 about 1.8 GB, Q8_0 about 1.3 GB); memoravox GGUF Q4_K_M 654 MB with 2.26% WER on LibriSpeech test-clean; a full-ONNX CPU package; an MLX port reporting RTF about 0.08 on M4 Pro. — [HF discussion #12](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/discussions/12); [predict-woo/qwen3-asr.cpp](https://github.com/predict-woo/qwen3-asr.cpp); [memoravox GGUF](https://huggingface.co/memoravox/Qwen3-ASR-0.6B-gguf); [Daumee ONNX CPU](https://huggingface-proxy.pages.dev/Daumee/Qwen3-ASR-0.6B-ONNX-CPU); [MLX article](https://besthub.dev/articles/qwen3-asr-runs-natively-on-apple-silicon-via-mlx-for-full-speed-speech-recognition-b8eed0035bbf)
- [S] A CSDN write-up reports about 5-6 s to transcribe 30 s on a laptop CPU with a Q5_K_M GGUF of about 760 MB; self-reported, hardware unspecified. — [CSDN](https://aicoding.csdn.net/6a33c72910ee7a33f27f80b7.html)

### Inferences
- Any approach that keeps the stock architecture (plain SOT fine-tune) inherits all of the above deployment paths unchanged, including the community CPU ports, because only weights change. Approaches that add modules (per-speaker CTC heads, separator, speaker kernels) would need each port to be modified.
- The official table measures about 2-minute single-talker audio. Offline RTF at concurrency 1 is 0.009, so there is large headroom on GPU even if SOT doubles output length; the tight case is CPU streaming, where one port already fell to 0.96x on a 135 s file.
- The streaming scheme re-decodes from a rolled-back prefix on every 2 s chunk, so longer outputs per chunk raise per-chunk cost directly.
- vLLM's LoRA gap means a LoRA-trained multi-talker adapter should be merged into the base weights before serving.

### Gaps
- Projector parameter count and an exact encoder/projector/LLM split summing to the total: Not Reported.
- Official VRAM guidance: none exists (issue #20 went unanswered; the report has no memory column).
- GPU model behind Table 2: the report says only "a single typical computing resource".
- No sherpa-onnx or OpenVINO benchmark for Qwen3-ASR was found.
- Still [S]: the vLLM LoRA issue and PR, the arXiv 2604.14493 CPU figure, the VoicePing Android figures, and all community GGUF/ONNX/MLX size and speed claims.

## 2. Fine-tuning practicalities for Qwen3-ASR and comparable Speech-LLMs (applies to approaches 1-3)

### Takeaway
An official fine-tuning script exists and takes plain audio + text JSONL, so SOT targets can be fed without code changes, but it gives no VRAM or time guidance and does not say whether it is full fine-tuning or LoRA. No published GPU-memory or GPU-hour figure for fine-tuning Qwen3-ASR was found; the nearest reported training cost for a comparable add-on is 7 hours on 8 V100 GPUs (Sidecar, frozen backbone).

### Cited Findings
- [D] Official script: `finetuning/qwen3_asr_sft.py`. Data is JSONL with `audio` (WAV path) and `text` (transcript, optionally prefixed `language English<asr_text>`; `language None<asr_text>` if unknown). — [Qwen3-ASR finetuning](https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning)
- [D] Example command: `--batch_size 32 --grad_acc 4 --lr 2e-5 --epochs 1 --save_steps 200`; multi-GPU via `torchrun --nproc_per_node=2`; checkpoints load with `Qwen3ASRModel.from_pretrained`. — [Qwen3-ASR finetuning](https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning)
- [D] The page states no GPU type, VRAM requirement or training time, and does not say whether training is full or LoRA. Only hardware hint: FlashAttention 2 is recommended for lower memory, and `MAX_JOBS=4` when building it on machines with under 96 GB RAM. — [Qwen3-ASR finetuning](https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning)
- [S] Open issue #176 (May 26, 2026): the fine-tuning callback silently skips copying the preprocessor/chat template when `--model_path` is a Hub ID. — [Qwen3-ASR issues](https://github.com/QwenLM/Qwen3-ASR/issues)
- [D] Comparable recipe (Hao Shi et al., WavLM-Large + LLaMA): LoRA r=16, alpha=32, encoder and decoder frozen, projector trainable, 30 epochs, per-device batch 8, bf16, lr 3e-5. GPU type and hours are not stated in the repo or the paper. — [hshi-speech/Multi-talker-ASR-with-LLMs](https://github.com/hshi-speech/Multi-talker-ASR-with-LLMs); [arXiv 2509.04488](https://arxiv.org/html/2509.04488)
- [D] Comparable recipe (MOSS-Transcribe-Diarize 0.9B, same Qwen3-0.6B-style decoder): `finetune.py` with per-device batch 1, 3 epochs, lr 1e-5, bf16, gradient checkpointing; default max sequence length 131,072 tokens. No VRAM or time stated. — [MOSS FINETUNING.md](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/blob/main/FINETUNING.md)
- [D] Reported training costs for related systems: Sidecar converges in "about 7 hours ... with 8 NVIDIA V100 GPUs" (at most 100k updates, backbone frozen); t-SOT used 16 GPUs for 225K iterations (GPU type not stated); the NVIDIA self-speaker-adaptation models were trained 200k steps on 8 A100 GPUs. — [arXiv 2302.09908](https://arxiv.org/pdf/2302.09908); [Kanda et al., Interspeech 2022](https://www.isca-archive.org/interspeech_2022/kanda22_interspeech.pdf); [arXiv 2506.22646](https://arxiv.org/html/2506.22646)
- [D] Standard training-set sizes used by these papers: Libri2Mix about 270 hours, Libri3Mix about 186 hours. — [arXiv 2509.04488](https://arxiv.org/html/2509.04488)

### Inferences
- Because the official script consumes free-form target text, a first SOT experiment is a data-preparation task (build mixtures, write serialized targets with a separator token), not an engineering task.
- The example batch size of 32 was written for the 1.7B model on unspecified hardware; it says nothing reliable about what fits on a given GPU for 0.6B.

### Gaps
- Training compute (GPU type x hours) for Qwen3-ASR fine-tuning: Not Reported.
- Community fine-tuning reports with memory or time numbers: none found.
- Whether adding a new special token (speaker-change) works with the official script without modification: not documented.
- Still [S]: issue #176.

## 3. Approach 1 — plain SOT fine-tuning, and how SOT changes LLM decoding cost

### Takeaway
SOT adds no modules and keeps the single-model pipeline; the t-SOT authors classify its decoder inference cost and architecture as "same with single-talker model", but mark it as not streamable. The only measured LLM-decoder figure is RTF 0.115 (Libri2Mix) and 0.098 (Libri3Mix) for a LLaMA-1B SOT system on an H100 at batch size 1. No paper reports single-talker vs multi-talker RTF for the same LLM decoder.

### Cited Findings
- [D] Comparison table in the t-SOT paper, SOT column: streaming inference "Not-available"; inference cost on decoder "Same with single-talker model"; max concurrent utterances "Unlimited"; speaker counting "Available"; model architecture "Same with single-talker model"; "Restricted to AED-based ASR". — [Kanda et al., Interspeech 2022](https://www.isca-archive.org/interspeech_2022/kanda22_interspeech.pdf)
- [D] RTF table (Shi et al., Interspeech 2026): LLaMA-1B SOT system 0.1150 on Libri2Mix and 0.0981 on Libri3Mix; encoder-only CTC model 0.0043 and 0.0106. "RTF was computed as the ratio of total inference time to total input audio duration, measured on a single NVIDIA H100 80GB GPU with batch size 1." — [Shi et al., Interspeech 2026](https://www.isca-archive.org/interspeech_2026/shi26_interspeech.pdf)
- [D] SOT baseline WER with WavLM-Large + LLaMA (noisy LibriMix eval, 2-mix / 3-mix): 1B 11.3 / 39.1; 3B 9.8 / 31.7; 8B 14.3 / 47.5. Clean eval for 1B: 4.6 / 21.6. A larger LLM did not help monotonically. — [arXiv 2509.04488](https://arxiv.org/html/2509.04488)
- [D] Stage 1 of the SOP recipe is plain SOT fine-tuning: it trains the speech encoder on mixtures and adapts the LLM decoder "to serialize the mixed token sequences". — [arXiv 2509.04488](https://arxiv.org/html/2509.04488)
- [D] In the Hao Shi et al. code, the plain SOT recipe (`attention` mode, CE only) uses the unmodified encoder-projector-LLM path plus an added `<sc>` speaker-change token. — [hshi-speech/Multi-talker-ASR-with-LLMs](https://github.com/hshi-speech/Multi-talker-ASR-with-LLMs)
- [S] LLM-based SOT with a WavLM-Large encoder reached 9.0% WER on two-speaker LibriMix vs 9.2% for an AED baseline. — [arXiv 2408.17431](https://arxiv.org/pdf/2408.17431)

### Inferences
- "Same decoder cost as single-talker" in the t-SOT table is per token; total decoding time still grows with the number of tokens emitted, which for an overlapped clip is the sum over speakers. No source quantifies this for an LLM decoder.
- The Libri3Mix RTF being lower than the Libri2Mix RTF shows RTF does not scale simply with speaker count across test sets, so "tokens double, therefore RTF doubles" should not be asserted without measuring.
- Plain SOT degrades sharply at three speakers in every LLM size (31.7-47.5% WER noisy), so a plain-SOT fine-tune is credible mainly for the two-speaker case.
- Utterance-level SOT emits speaker 1's whole text before speaker 2's, which conflicts with Qwen3-ASR's prefix-rollback streaming that assumes monotonic output. The t-SOT paper's "streaming: not available" for SOT supports this for the general case; the Qwen3-specific conflict is my inference.

### Gaps
- Single- vs multi-talker RTF on the same LLM-decoder model: Not Reported.
- Training compute for LLM-based SOT (GPU type x hours): Not Reported in the SOP paper or repo.
- Still [S]: arXiv 2408.17431.

## 4. Approach 2 — t-SOT (streaming)

### Takeaway
t-SOT orders tokens by time with a channel-change token, which makes serialized output streamable while keeping decoder cost and architecture "same with single-talker model". On monaural LibriCSS it beat single-talker streaming models by a wide margin and was close to or better than offline Conformer-CSS cascades. All results are on transformer-transducers, the basic form handles up to two concurrent utterances, and it needs word-level time alignments to build targets.

### Cited Findings
- [D] t-SOT column of the comparison table: streaming "Available"; inference cost on decoder "Same with single-talker model"; max concurrent utterances "Pre-defined"; speaker counting "Not-available"; architecture "Same with single-talker model"; "No restriction" on ASR type. — [Kanda et al., Interspeech 2022](https://www.isca-archive.org/interspeech_2022/kanda22_interspeech.pdf)
- [D] Experiments used up to two concurrent utterances "because overlaps of more than two speakers are rare"; a generalized form for up to M concurrent utterances is described. — [Kanda et al., Interspeech 2022](https://www.isca-archive.org/interspeech_2022/kanda22_interspeech.pdf)
- [D] Model sizes: t-SOT TT-18 82M parameters, TT-36 139M. Minimum algorithmic latency 40 ms; 160 ms is the target configuration. — [Kanda et al., Interspeech 2022](https://www.isca-archive.org/interspeech_2022/kanda22_interspeech.pdf)
- [D] LibriSpeechMix test WER (1spk / 2spk): TT-18 at 40 ms 5.1 / 8.4; at 160 ms 4.9 / 6.9; at 640 ms 4.2 / 6.2; at 2560 ms 3.9 / 5.2. TT-36 at 160 ms 4.3 / 6.2; at 2560 ms 3.3 / 4.4. Offline SOT Conformer-AED (129M): 3.6 / 4.9. — [Kanda et al., Interspeech 2022](https://www.isca-archive.org/interspeech_2022/kanda22_interspeech.pdf)
- [D] A single-talker TT-18 scores 63.7% on the 2-speaker test set; t-SOT TT-18 trained with 33% two-speaker data scores 7.4% while single-speaker WER moves from 4.5 to 4.7. — [Kanda et al., Interspeech 2022](https://www.isca-archive.org/interspeech_2022/kanda22_interspeech.pdf)
- [D] Training: 225K iterations with 16 GPUs (type not stated), mini-batches of 12,000 frames each; 50K more iterations for LibriCSS. Training targets used time alignments from the Montreal Forced Aligner. — [Kanda et al., Interspeech 2022](https://www.isca-archive.org/interspeech_2022/kanda22_interspeech.pdf)
- LibriCSS numbers are in section 8, which holds the cascade comparison.

### Inferences
- Since token order is monotonic in time, t-SOT is the serialization that fits Qwen3-ASR's chunked prefix-rollback streaming. No published work applies t-SOT to a Speech-LLM decoder in the sources read.
- Building t-SOT targets needs word-level timestamps per speaker. Qwen3-ForcedAligner-0.6B (11 languages, up to 300 seconds, [arXiv 2601.21337v2](https://arxiv.org/html/2601.21337v2) [D]) is an available tool for that.
- The trade against plain SOT is explicit in the paper's table: t-SOT gains streaming but loses unlimited speaker count and speaker counting.

### Gaps
- t-SOT on LLM-decoder ASR: no reported results found.
- RTF and VRAM for t-SOT: Not Reported in the paper.

## 5. Approach 3 — SOT + per-speaker CTC / feature separator (Hao Shi et al.; Sidecar)

### Takeaway
These methods insert a small separator (and CTC heads) into or after the encoder. Sidecar is 8.7M parameters (8.4% of the system) and trained in 7 hours on 8 V100s with the backbone frozen. In the SOP method the separator and CTC layers are needed at inference, training takes three stages, and the gain over plain SOT is small for two speakers (11.3 to 10.5 WER) but large for three (39.1 to 28.5).

### Cited Findings
- [D] Sidecar (Meng et al., ICASSP 2023): mounted between two encoder layers of a wav2vec 2.0-based ASR model; only the Sidecar is trained (8.7M, 8.4% of all parameters); 10.36% WER on 2-speaker LibriMix and 7.56% on LibriSpeechMix. — [arXiv 2302.09908](https://arxiv.org/abs/2302.09908)
- [D] Sidecar training cost: at most 100k updates; "about 7 hours for models' convergence with 8 NVIDIA V100 GPUs". — [arXiv 2302.09908](https://arxiv.org/pdf/2302.09908)
- [D] SOP architecture: "a speech encoder, several downsampling layers, a projector, a separator, multiple CTC layers, and an LLM-based decoder". The separator is an LSTM followed by LayerNorm and per-speaker Linear + ReLU. — [arXiv 2509.04488](https://arxiv.org/html/2509.04488)
- [D] The separator and CTC layers are used at inference: the decoder input is the concatenation of the SOP embeddings, the mixture speech embedding and the text embedding. Removing the mixture embedding and keeping only the SOP prompt collapses accuracy (1B: 34.4% on noisy Libri2Mix eval). — [arXiv 2509.04488](https://arxiv.org/html/2509.04488)
- [D] The separator takes the second CNN downsampling layer's output, not the final one, because "the final layer applies excessive downsampling". — [arXiv 2509.04488](https://arxiv.org/html/2509.04488)
- [D] Three training stages: (1) SOT fine-tuning; (2) train encoder, CNN downsampling, separator and serialized CTC; (3) freeze those and train only a new set of LoRA modules. The authors "attempted to train all parameters directly in a single stage, but the system performance was suboptimal". — [arXiv 2509.04488](https://arxiv.org/html/2509.04488)
- [D] WER, noisy LibriMix eval (2-mix / 3-mix): LLaMA-1B SOT 11.3 / 39.1, SOP 10.5 / 28.5; LLaMA-3B SOT 9.8 / 31.7, SOP 9.2 / 28.1. Clean eval, 1B: SOT 4.6 / 21.6, SOP 4.0 / 22.0. — [arXiv 2509.04488](https://arxiv.org/html/2509.04488)
- [D] The paper states no GPU type, training time, RTF or parameter count for the separator and CTC layers. — [arXiv 2509.04488](https://arxiv.org/html/2509.04488)
- [D] Repo contents: WavLM-Large encoder; LLaMA-3.2 1B/3B and LLaMA-3.1 8B decoders; `separator.py`, `ctc.py`, hybrid loss alpha*CE + (1-alpha)*CTC with alpha = 0.7; cross-attention adapter variants. No WER, RTF, hardware or license is listed. — [hshi-speech/Multi-talker-ASR-with-LLMs](https://github.com/hshi-speech/Multi-talker-ASR-with-LLMs)
- [S] A Sidecar follow-up adds diarization with only 768 additional parameters. — [arXiv 2305.16263](https://arxiv.org/pdf/2305.16263)

### Inferences
- For two speakers the SOP gain over plain SOT is under 1 point absolute, bought with two extra modules, two extra training stages and a modified inference graph. On the team's "simple, single-model" priority, this approach only pays off if three-speaker overlap matters.
- The SOP separator needed a less-downsampled feature than the encoder's final output. AuT outputs 12.5 Hz, lower than the rates used there, so a separator on AuT's final output may face the same "excessive downsampling" problem. This is an inference; nothing in the literature tests it.
- The absence of a license file means the Hao Shi et al. code cannot be assumed reusable in a product.

### Gaps
- Parameter count of the SOP separator and CTC heads; RTF, VRAM and GPU hours for approach 3 on an LLM decoder: Not Reported.
- Streaming capability: not addressed in either paper.
- Still [S]: arXiv 2305.16263.

## 6. Approach 4 — multi-output / PIT ASR

### Takeaway
The one sourced cost statement for multi-branch models is that their decoder inference cost is K times a single-talker model, with most parameters loaded and used twice for two branches. No cost data exists for a PIT design on an LLM decoder.

### Cited Findings
- [D] Multi-output streaming models (SURT, MS-RNN-T) in the t-SOT comparison table: inference cost on decoder "K-times of single-talker model"; architecture "Multiple output branches"; max concurrent utterances "Pre-defined"; accuracy on non-overlapping audio "Bad". — [Kanda et al., Interspeech 2022](https://www.isca-archive.org/interspeech_2022/kanda22_interspeech.pdf)
- [D] Parameter counts and LibriSpeechMix 2-speaker WER for multi-branch models: PIT LSTM-AED 161M, 11.9 (offline); LSTM-SURT 81M, 10.3; PIT-MS-RNN-T 81M, 10.2 and 8.8; Transformer-SURT 85M, 9.1. These are models "with two output branches, for which most parameters are loaded and used twice in the inference". All are worse than t-SOT TT-18 (82M, 6.9 at 160 ms). — [Kanda et al., Interspeech 2022](https://www.isca-archive.org/interspeech_2022/kanda22_interspeech.pdf)

### Inferences
- For an LLM decoder, a multi-output design means several decoder passes or several output heads, which the Qwen3-ASR serving stack (vLLM, one text stream per request) does not provide. This is a structural inference.
- On the evidence above, PIT multi-output is dominated by serialized output on both cost and accuracy for transducer and AED models.

### Gaps
- RTF, VRAM, training compute for PIT-based ASR, and any PIT result on a Speech-LLM: Not Reported.

## 7. Approach 5 — target-speaker / speaker-conditioned ASR (NVIDIA multitalker Parakeet, DiCoW)

### Takeaway
NVIDIA's model runs one 600M ASR instance per speaker, each fed the same audio plus that speaker's activity from a mandatory streaming Sortformer diarizer, so compute scales with speaker count. It is under the NVIDIA Open Model License, built on an English base model, and reports cpWER 15.81-37.44 on real conversations at 1.12 s latency. It is a transducer, so it serves as a comparison baseline, not something to graft onto Qwen3-ASR.

### Cited Findings
- [D] Model card badge: 600M parameters; "a streaming multitalker ASR model based on the Nemotron-Speech-Streaming model" (`nvidia/nemotron-speech-streaming-en-0.6b`). — [NVIDIA model card](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1)
- [D] "The model architecture requires deploying one model instance per speaker, meaning the number of model instances matches the number of speakers in the conversation. While this necessitates additional computational resources, it achieves state-of-the-art performance in handling fully overlapped speech." — [NVIDIA model card](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1)
- [D] Each instance receives the same mixed audio, injects speaker-specific kernels at the pre-encode layer, outputs only its target speaker's text, and "operates independently and can run in parallel with other instances". — [NVIDIA model card](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1)
- [D] The paper implements the instances as a batch: "the batch size corresponds to the number of speakers in the input audio". — [arXiv 2506.22646](https://arxiv.org/html/2506.22646)
- [D] A separate diarizer is required: the example loads `nvidia/diar_streaming_sortformer_4spk-v2.1` (v2 also listed). No speaker enrollment is needed. — [NVIDIA model card](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1)
- [D] Latency options via `att_context_size`, in 80 ms frames: [70,0] = 0.08 s; [70,1] = 0.16 s; [70,6] = 0.56 s; [70,13] = 1.12 s. — [NVIDIA model card](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1)
- [D] cpWER with Streaming Sortformer v2 at 1.12 s latency: AMI IHM 21.26, AMI SDM 37.44, CH109 15.81, Mixer 6 23.81. AMI sets have 3-4 speakers; CH109 and Mixer 6 have 2. — [NVIDIA model card](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1)
- [D] Single-speaker mode keeps base accuracy: 7.44 average WER on the Open ASR Leaderboard sets vs 7.16 for the base model. — [NVIDIA model card](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1)
- [D] License: `nvidia-open-model-license`. Input: mono 16 kHz. Real training conversations: Granary (single speaker), Fisher English, LibriSpeech, AMI, NOTSOFAR, ICSI; mixtures simulated from LibriSpeech. — [NVIDIA model card](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1)
- [D] The card states no explicit maximum speaker count, no language field (the language badge is commented out), no training hours, and no RTF or VRAM. — [NVIDIA model card](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1)
- [D] Research system in the paper: 238M parameters = FastConformer Transducer ASR (114M) + Sortformer diarizer (123M) + speaker injection module (1.1M). Trained 200k steps on 8 A100 GPUs. — [arXiv 2506.22646](https://arxiv.org/html/2506.22646)
- [D] On CH109 at 1120 ms latency, the paper's cascaded baseline (streaming single-speaker ASR + streaming diarizer, words matched to speakers by timestamps) scores 27.38% cpWER vs 26.21% for the proposed model; an offline 2.2B LLM-based system scores 26.31%. — [arXiv 2506.22646](https://arxiv.org/html/2506.22646)
- [D] A Rust runtime lists the model as English only, fed in chunks of 17,920 samples (about 1.12 s), requiring a separate diarization ONNX file. — [altunenes/parakeet-rs](https://github.com/altunenes/parakeet-rs)
- [S] DiCoW conditions Whisper on diarization output through frame-level diarization-dependent transforms (four class-specific affine transforms in the encoder layers); a follow-up initializes from a DiCoW checkpoint based on Whisper-large-v3-turbo. — [arXiv 2501.00114](https://arxiv.org/pdf/2501.00114); [arXiv 2510.03723](https://arxiv.org/pdf/2510.03723)
- [S] A third-party GGUF port measured 19.35% cpWER on AMI-IHM test (16 meetings); this is the port's measurement, not NVIDIA's. — [handy-computer GGUF port](https://huggingface.co/handy-computer/multitalker-parakeet-streaming-0.6b-v1-gguf)

### Inferences
- The speaker-injection module is tiny (1.1M in the paper), so this family's cost is the mandatory diarizer plus N parallel ASR passes, not added parameters.
- The effective speaker ceiling is set by the diarizer, which is a 4-speaker model; the card does not state a limit itself.
- "English only" is supported by the base model's name and a third-party runtime, not by an explicit statement on the card.
- DiCoW has the same shape: one decoding pass per diarized speaker plus an external diarizer. Neither keeps a single-model pipeline.

### Gaps
- NVIDIA model: RTF, VRAM, training hours, explicit speaker limit and explicit language statement: Not Reported on the card.
- DiCoW added parameters, per-speaker inference cost, license, training GPUs: not checked against the primary paper.
- Still [S]: both DiCoW items and the third-party GGUF cpWER.

## 8. Approach 6 — separation front-end / CSS cascade

### Takeaway
On monaural LibriCSS, the same-test-set comparison shows a streaming single model (t-SOT, 9.0-10.1 average WER at 160 ms latency) matching or beating offline Conformer-CSS + ASR cascades (9.6-11.8), and the cascade's best figure required fine-tuning the separator against the ASR. Separation models add 10-56M parameters and a second inference stage, and the Conformer-CSS authors themselves describe them as expensive and slow on edge devices.

### Cited Findings

**Cascade vs single model on the same test set (monaural LibriCSS, continuous input, WER % by overlap ratio)**

| System | Latency | 0L | 0S | 10 | 20 | 30 | 40 | Avg |
|---|---|---|---|---|---|---|---|---|
| BLSTM-CSS + Hybrid ASR | 1.2 s + utterance length | 16.3 | 17.6 | 20.9 | 26.1 | 32.6 | 36.1 | 24.9 |
| Conformer-CSS + Transformer-AED ASR w/ LM | 1.2 s + utterance length | 6.1 | 6.9 | 9.1 | 12.5 | 16.7 | 19.3 | 11.8 |
| Conformer-CSS (ASR-oriented fine-tuning) + Transformer-AED ASR w/ LM | 1.2 s + utterance length | 6.4 | 7.5 | 8.4 | 9.4 | 12.4 | 13.2 | 9.6 |
| Single-talker TT-18 (no separation) | 160 ms | 7.0 | 7.3 | 14.0 | 20.9 | 27.9 | 34.3 | 18.6 |
| t-SOT TT-18 | 160 ms | 7.5 | 7.5 | 8.5 | 10.5 | 12.6 | 14.0 | 10.1 |
| t-SOT TT-36 | 160 ms | 6.7 | 6.1 | 7.5 | 9.3 | 11.6 | 12.9 | 9.0 |

- [D] Table above. The 1.2 s is latency incurred by CSS; the rest by VAD and ASR (average utterance 7.5 s). — [Kanda et al., Interspeech 2022](https://www.isca-archive.org/interspeech_2022/kanda22_interspeech.pdf)

**Cascade problems and the need for ASR-aware training**
- [D] Conventional mask-based separation "introduces target speech distortion, resulting in a sub-optimum word error rate"; a potential reason for the single- vs multi-channel gap is "the mismatch between the front-end objective function and speech recognition". — [Wu et al., Interspeech 2021](https://www.isca-archive.org/interspeech_2021/wu21f_interspeech.pdf)
- [D] Single-channel Conformer separation trails multi-channel by 5.9% and 8.2% absolute WER at 30% and 40% overlap. — [Wu et al., Interspeech 2021](https://www.isca-archive.org/interspeech_2021/wu21f_interspeech.pdf)
- [D] Fine-tuning the separator with an ASR-oriented loss, continuous evaluation: WER at 40% overlap falls from 19.3 to 13.2 and at 30% from 16.7 to 12.4, but non-overlapped conditions get slightly worse (0S 6.9 to 7.5; 0L 6.1 to 6.4). — [Wu et al., Interspeech 2021](https://www.isca-archive.org/interspeech_2021/wu21f_interspeech.pdf)
- [D] Without separation the matched ASR scores 38.3% at 40% overlap (utterance-wise); with a 26.03M Conformer separator, 15.1 (signal loss) or 13.9 (feature loss). In non-overlapped conditions separation gives little or no gain (0S 6.0 without vs 5.7-6.1 with). — [Wu et al., Interspeech 2021](https://www.isca-archive.org/interspeech_2021/wu21f_interspeech.pdf)
- [D] The authors keep the ASR fixed during ASR-oriented training and note that "a jointly trained speech separation model may suffer from performance" issues when paired with a different ASR; they test a second, unmatched ASR for that reason. — [Wu et al., Interspeech 2021](https://www.isca-archive.org/interspeech_2021/wu21f_interspeech.pdf)
- [S] A 2026 paper states that off-the-shelf ASR models not exposed to a separation front-end's output distribution often fail to benefit from it. — [arXiv 2605.19695](https://arxiv.org/pdf/2605.19695)
- [S] In a music-mixed speech study, fine-tuning only the separation front-end was less effective than fine-tuning the ASR back-end. — [arXiv 2008.12048](https://arxiv.org/pdf/2008.12048)
- [S] The CHiME-8 default ASR favours GSS because it was trained on GSS-enhanced signals. — [MERL TR2026-008](https://www.merl.com/publications/docs/TR2026-008.pdf)

**Size and compute**
- [D] LibriCSS separation model sizes: Conformer-base 26.03M, Conformer-small 9.97M, Transformer 12.97M, CRN 20.37M, Dense-CRN 17.99M. Conformer separation "usually brings difficulties in model deployment because of expensive computation and slow inference, especially on edge devices". — [Wu et al., Interspeech 2021](https://www.isca-archive.org/interspeech_2021/wu21f_interspeech.pdf)
- [S] WSJ0-2mix parameter counts and SI-SDRi from one survey: MossFormer2 55.7M / 24.1 dB; TF-GridNet 14.5M / 23.5 dB; SepFormer 26.0M / 22.3 dB; DPRNN 2.9M / 18.8 dB; Conv-TasNet 5.1M / 15.3 dB. — [arXiv 2508.10830](https://arxiv.org/pdf/2508.10830)
- [S] One paper reports MACs of 10.09G (Conv-TasNet), 86.31G (DPRNN), 74.94G (SepFormer) with RTFs of 0.16, 1.18 and 12.40. Hardware not captured. — [OpenReview: Separate and Reconstruct](https://openreview.net/pdf?id=99y2EfLe3B)
- [S] MossFormer2 RTF 0.053 on a V100 GPU. — [emergentmind MossFormer2 summary](https://www.emergentmind.com/topics/mossformer2-separated-speech)
- [S] TF-GridNet (14M) and SPMamba (6M) require 445 and 238 GMAC/s and are described as not real-time friendly. — [arXiv 2510.12275](https://arxiv.org/pdf/2510.12275)
- [S] TF-Locoformer: medium 15.0M params, 22.1 dB SI-SNRi on WSJ0-2mix; small 5.0M params, 85.9 G MACs/s. — [arXiv 2408.03440](https://arxiv.org/pdf/2408.03440); [arXiv 2607.21128](https://arxiv.org/pdf/2607.21128)

### Inferences
- The cascade adds a second model and decodes each separated stream separately, so with a 0.6B LLM decoder it means two full autoregressive passes for two speakers versus one longer pass for SOT.
- The LibriCSS table is the strongest published argument against the cascade for this team: a single 82M-139M streaming model equals offline cascades that carry a separator, an AED ASR and an LM.
- Separation helps only in overlapped regions and can slightly hurt non-overlapped speech after ASR-oriented tuning, so a cascade risks a regression on the single-talker traffic that Qwen3-ASR already handles well.
- The best cascade number depended on tuning the separator to one specific ASR, so a cascade does not avoid model-specific training work.
- The separators with the best WSJ0-2mix scores are the ones with the worst reported compute, pushing a CPU-deployable cascade toward Conv-TasNet-class models that trail by 7-9 dB.

### Gaps
- Same-hardware CPU RTF for the listed separators: none found; cross-paper RTFs use different hardware and cannot be compared.
- A cascade vs SOT comparison on an LLM-decoder ASR, or on NOTSOFAR single-channel: not found.
- Still [S]: all WSJ0-2mix size/RTF figures for Conv-TasNet, DPRNN, SepFormer, TF-GridNet, MossFormer2 and TF-Locoformer; arXiv 2605.19695; arXiv 2008.12048; the MERL review.
- Dropped as unverified: an earlier draft cited "17.1 WER at 40% overlap" for Conformer CSS and "7.6 average at 2560 ms" for t-SOT on LibriCSS; neither appears in the primary tables read.

## 9. Approach 7 — external baseline: MOSS-Transcribe-Diarize 0.9B

### Takeaway
MOSS-Transcribe-Diarize is the closest architectural sibling to Qwen3-ASR-0.6B (Whisper-medium-style encoder + Qwen3-0.6B-style decoder, Apache-2.0, up to 90 minutes in one pass) and ships a fine-tuning script and data format publicly. Neither its model card nor its repo mentions overlapping or simultaneous speech, and neither states VRAM.

### Cited Findings
- [D] Architecture: "Qwen3-0.6B style causal decoder"; Whisper-Medium encoder configuration; WhisperFeatureExtractor (16 kHz, 80 mel bins, 30 s chunks); 4x temporal merge + MLP adaptor. Stated size 0.9B. — [OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
- [D] Model card: Apache License 2.0; released 2026-07-09; 50+ languages; "single-pass inference on audio recordings up to 90 minutes long"; hotword prompting. — [MOSS model card](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)
- [D] It "jointly performs speech transcription and speaker diarization" instead of "stitching together separate ASR and diarization systems", assigning anonymous labels such as `[S01]`. Output format: `[start_time][Sxx]text[end_time]`. — [MOSS model card](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize); [OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
- [D] A keyword check of the model card for "overlap", "simultaneous" and "cross-talk" returns nothing; the card describes "long or messy multi-speaker recordings" only. The fine-tuning guide does not say how overlapping segments are represented or ordered. — [MOSS model card](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize); [FINETUNING.md](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/blob/main/FINETUNING.md)
- [D] Training code is public: `finetune.py` and `FINETUNING.md`. Data is JSONL, one conversation per record, with three messages: text prompt, audio path, reference transcript in the output format (example `[0.00][S01]Welcome[0.72]`). — [FINETUNING.md](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/blob/main/FINETUNING.md)
- [D] Speed on a single H100 with SGLang Omni: short clips, concurrency 1, mean RTF 0.0612 (29.76 audio-s/s); concurrency 16, RTF 0.0922 (81.98 audio-s/s). Long AISHELL-4 sessions: concurrency 1, RTF 0.0197 (50.64 audio-s/s); concurrency 16, RTF 0.1237 (98.83 audio-s/s). — [OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
- [D] VRAM is not stated on the card or in the repo. The README warns eager attention memory "grows quadratically with audio length and OOMs on long recordings". Default output cap is 5120 new tokens, to be raised to 65536 for long audio. — [OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
- [D] 0.9B results (CER / cpCER): AISHELL-4 14.84 / 15.83; AliMeeting 24.86 / 22.17; Podcast 5.97 / 7.37; Movies 6.36 / 12.76. A stronger "Pro" model is available only via an online playground. — [OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
- [D] Serving: Transformers 5.x with `trust_remote_code=True`; vLLM needs a pinned nightly; SGLang Omni is the recommended backend and targets CUDA 13. — [OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
- [S] A conversion card gives 908,513,280 total parameters: about 312M audio side and about 596M text decoder. — [soniqo ONNX FP16 card](https://huggingface.co/soniqo/MOSS-Transcribe-Diarize-0.9B-ONNX-FP16)
- [S] A conversion page warns the source model can produce malformed or overlapping timestamps and can miss speakers. Community CoreML, ONNX and GGUF conversions exist. — [handy-computer GGUF card](https://huggingface.co/handy-computer/moss-transcribe-diarize-gguf)

### Inferences
- MOSS is evidence that a sub-1B, single-model, single-output-stream Speech-LLM can emit speaker-tagged multi-speaker transcripts with ordinary fine-tuning tooling, which is the same deployment shape as SOT on Qwen3-ASR.
- AliMeeting is the overlap-heavy set in its table and has the worst CER (24.86), which is consistent with overlap being a weak point, but the authors do not attribute the error to overlap.
- Its segment-with-timestamps format can in principle express overlapping segments, but whether the released model does so is undocumented and needs a direct test on overlapped clips.
- MOSS's RTF 0.061 at concurrency 1 (H100, SGLang, short clips, diarized output) and Qwen3-ASR-0.6B's 0.009 (unnamed GPU, vLLM, 2-minute clips, plain ASR) differ in backend, hardware and task, so they are not a like-for-like comparison.

### Gaps
- Whether it truly transcribes simultaneous speech: not documented anywhere official.
- VRAM, training data composition, training compute: Not Reported.
- LoRA vs full fine-tuning in `finetune.py`: not stated.
- Still [S]: the 908.5M parameter split and the conversion-page warning.

## 10. Cross-approach summary of reported structural facts

### Takeaway
Only plain SOT and t-SOT keep the stock single-model pipeline with zero added inference modules; every other approach adds an in-model module (separator/CTC) or a whole second model (diarizer or separator) and, for approaches 4-6, extra decoding passes per speaker. VRAM is not reported for any approach, and RTF is reported for only three.

### Cited Findings

| Approach | Added parameters (reported) | Extra model at inference | Single-model pipeline kept | Streaming | Reported speed | Reported training compute |
|---|---|---|---|---|---|---|
| 1. SOT fine-tune | None in recipe (one `<sc>` token) | No | Yes | "Not-available" | LLaMA-1B SOT: RTF 0.1150 (Libri2Mix), 0.0981 (Libri3Mix), H100, batch 1 | Not Reported |
| 2. t-SOT | None; "same with single-talker model" | No | Yes | Yes, 40-2560 ms (transducers) | Not Reported | 16 GPUs, 225K iterations (GPU type not stated) |
| 3. SOT + separator / CTC | Sidecar 8.7M (8.4%); SOP separator Not Reported | No separate model; in-model modules used at inference | Modified single model | Not Reported | Not Reported | Sidecar: 7 h on 8 V100; SOP: Not Reported |
| 4. PIT multi-output | Two branches; "most parameters loaded and used twice" | No | Multiple output branches | Yes (SURT, MS-RNN-T) | Decoder cost "K-times" single-talker | Not Reported |
| 5. NVIDIA multitalker Parakeet | Speaker injection 1.1M (research system) | Yes: streaming Sortformer (123M in research system) | No; one 600M ASR instance per speaker | Yes, 0.08 / 0.16 / 0.56 / 1.12 s | Not Reported | Research system: 200k steps on 8 A100 |
| 6. Separation cascade | Separator 10-26M (LibriCSS Conformer); 2.9-55.7M (WSJ0-2mix models, [S]) | Yes: separator | No; one ASR pass per stream | CSS adds 1.2 s latency | [S] RTF 0.16-12.40 across models, hardware unknown | Not Reported |
| 7. MOSS-TD 0.9B | n/a (separate 0.9B model) | No | Yes | Not Reported | RTF 0.0612 (c=1), 0.0922 (c=16), H100 | Not Reported |

Sources for the table rows: [Kanda et al., Interspeech 2022](https://www.isca-archive.org/interspeech_2022/kanda22_interspeech.pdf); [Shi et al., Interspeech 2026](https://www.isca-archive.org/interspeech_2026/shi26_interspeech.pdf); [arXiv 2509.04488](https://arxiv.org/html/2509.04488); [arXiv 2302.09908](https://arxiv.org/pdf/2302.09908); [arXiv 2506.22646](https://arxiv.org/html/2506.22646); [NVIDIA model card](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1); [Wu et al., Interspeech 2021](https://www.isca-archive.org/interspeech_2021/wu21f_interspeech.pdf); [arXiv 2508.10830](https://arxiv.org/pdf/2508.10830); [OpenReview: Separate and Reconstruct](https://openreview.net/pdf?id=99y2EfLe3B); [OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize).

### Inferences
- On the team's stated priorities (single model, no front-end, existing CPU ports, existing fine-tuning script), plain SOT has the lowest integration cost, and t-SOT is the variant to consider if streaming must be kept. The LibriCSS table adds an accuracy argument: a serialized single model matched offline separation cascades.
- Approach 3 is the only one whose extra cost buys something specific: three-speaker accuracy (39.1 to 28.5 WER with LLaMA-1B). For two speakers its gain is under 1 point.
- No approach has a published VRAM figure, so any VRAM comparison in a final report has to come from the team's own measurement.

### Gaps
- VRAM for every approach; RTF for approaches 2, 3 and 5; training GPU-hours for LLM-based SOT, SOP and MOSS: Not Reported in the primary sources read.
- No source applies t-SOT, Sidecar or PIT to a Speech-LLM decoder, so all cost transfer to Qwen3-ASR is inference.
- Hosts other than GitHub, arXiv, Hugging Face and the ISCA archive were not reachable, so blog and aggregator figures remain [S] as marked in each section.
