# Implementation and Deployment Cost of Overlapped-Speech ASR Approaches for Qwen3-ASR-0.6B

> **Read this first: source access.** This session's network only allowed `github.com` and web search. `arxiv.org`, `huggingface.co`, HF mirrors and most blog hosts did not resolve. So every finding carries one of two tags:
> - **[D]** = I opened the page directly (all GitHub pages).
> - **[S]** = the figure comes from a search-engine extract of the cited page; I could not open the page itself. Treat as "reported, needs a direct check against the primary source before it goes in a final report".
>
> No number below is estimated. Anything not found is marked **Not Reported** or listed under Gaps.

## 1. Qwen3-ASR-0.6B baseline facts (params, speed, VRAM, streaming, vLLM, CPU/quantization, max length, license)

### Takeaway
The baseline is an Apache-2.0, 18-layer AuT encoder (reported 180M params, 12.5 Hz output) + projector + Qwen3-0.6B decoder (28 layers), with official vLLM serving and an official full-fine-tuning script but no official quantized/CPU build. Reported speed is TTFT 92 ms and RTF 0.064 at concurrency 128 on GPU; CPU-only use exists only through community ports, with reported results ranging from several times faster than real time (Apple M3 Max, C port) to slower than real time (one arXiv benchmark, chunked mode).

### Cited Findings

**Architecture and parameters**
- [S] The tech report says Qwen3-ASR-0.6B is built from Qwen3-0.6B, a projector, and an AuT encoder with 180M parameters and hidden size 896; the 1.7B model uses a 300M-parameter AuT encoder with hidden size 1024. — [arXiv 2601.21337](https://arxiv.org/abs/2601.21337); same figures in [hyper.ai summary](https://hyper.ai/en/papers/Qwen3-ASR)
- [S] AuT is an attention-encoder-decoder model that downsamples 100 Hz Fbank features 8x to a 12.5 Hz token rate. — [hyper.ai summary](https://hyper.ai/en/papers/Qwen3-ASR)
- [D] Layer-level structure read from the weights by an independent C re-implementation: 0.6B = 18 encoder layers, encoder dim 896, 28 decoder layers, decoder dim 1024, 16/8 Q/KV heads, vocab 151,936, BF16 weights. The stem is 3 Conv2D layers (480 channels, stride 2, 8x time downsampling); the projector is Linear-GELU-Linear; roughly 13 encoder tokens per second of audio. — [antirez/qwen-asr](https://github.com/antirez/qwen-asr)
- [D] The same page lists the 0.6B static footprint as about 1.747 GiB of mmapped safetensors (BF16). — [antirez/qwen-asr](https://github.com/antirez/qwen-asr)
- Projector parameter count: **Not Reported** in any source found.

**Speed (GPU)**
- [S] The abstract states the 0.6B model reaches average TTFT as low as 92 ms and transcribes 2,000 seconds of speech per second at concurrency 128. — [arXiv 2601.21337](https://arxiv.org/abs/2601.21337)
- [D] The official README repeats that the 0.6B model "reaches 2000 times throughput at a concurrency of 128". — [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- [S] A summary of the report gives RTF 0.064 at concurrency 128 for 0.6B. — [arxivlens summary](https://arxivlens.com/paperview/details/qwen3-asr-technical-report-232-4fcaadd0)
- [S] The report's 0.6B vLLM-serve table lists, at concurrency 1, average TTFT 92 ms and p95 105 ms; measurements used vLLM v0.14.0 with CUDA Graph, in offline-batch and online-serving settings. — [arXiv 2601.21337 HTML v2](https://arxiv.org/html/2601.21337v2)
- [S] Conflict to resolve: a secondary article shows a per-concurrency table where TTFT at concurrency 128 is 3210 ms, so "92 ms" and "concurrency 128" are most likely different rows of the table, not the same operating point. That article's parameter counts are wrong, so it is low-reliability. — [besthub.dev](https://besthub.dev/articles/qwen3-asr-open-source-speech-recognition-supporting-52-languages-and-dialects-outperforming-whisper-ef37f210febf)

**VRAM**
- [D] The official README gives no VRAM figure. It only shows `gpu_memory_utilization` examples (0.7-0.9), recommends FlashAttention 2 to reduce GPU memory, and says a smaller `max_inference_batch_size` avoids OOM. — [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- [D] vLLM 0.14.0 startup log for Qwen3-ASR-0.6B posted by a user: "Model loading took 1.53 GiB memory"; CUDA graph capture "took 0.50 GiB"; with `--gpu-memory-utilization 0.2` and `--max-model-len 10000` the KV cache got 1.74 GiB (16,272 tokens). The user asked for recommended VRAM; no maintainer answered and the issue was closed as "not planned". — [Qwen3-ASR issue #20](https://github.com/QwenLM/Qwen3-ASR/issues/20)
- [D] A community member (not a maintainer) in the same issue claims "consistently < 2GB VRAM usage under load" with their own vLLM fork. Unverified claim. — [Qwen3-ASR issue #20](https://github.com/QwenLM/Qwen3-ASR/issues/20)

**Streaming mechanics**
- [D] One model serves both offline and streaming. Streaming is available only with the vLLM backend, and "streaming inference does not support batch inference or returning timestamps". — [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- [S] The report's streaming evaluation uses a 2-second chunk, a 5-token fallback (rollback), and keeps the last four chunks unfixed. — [arXiv 2601.21337](https://arxiv.org/pdf/2601.21337)
- [D] An independent re-implementation describes the same scheme: 2-second chunks; each step re-feeds the accumulated audio with the previous output minus a 5-token rollback suffix as decoder prefix; encoder attention uses local windows (8 s default) so finished windows can be cached. — [antirez/qwen-asr](https://github.com/antirez/qwen-asr)
- [D] Streaming costs accuracy: 1.7B averages 3.33 WER streaming vs 2.69 offline on the README's table. — [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)

**vLLM, max length, license, languages**
- [D] vLLM "officially provides day-0 model support"; `qwen-asr-serve` wraps `vllm serve`; OpenAI-compatible chat and transcription endpoints; official Docker image `qwenllm/qwen3-asr`. — [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- [S] The report says both models support a single utterance no longer than 20 minutes. — [arXiv 2601.21337](https://arxiv.org/pdf/2601.21337)
- [D] License Apache-2.0; 30 languages plus 22 Chinese dialects. — [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- [D] vLLM did not support LoRA adapters for `Qwen3ASRForConditionalGeneration` as of feature request #37223; a later PR (#50793) fixes audio-tower Q/K/V LoRA adapters being silently skipped. Merge status not checked. — [vLLM issue #37223](https://github.com/vllm-project/vllm/issues/37223); [vLLM PR #50793](https://github.com/vllm-project/vllm/pull/50793)

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
- The streaming scheme re-decodes from a rolled-back prefix on every 2 s chunk. The 135 s `--stream` result (0.96x) on a fast laptop CPU shows CPU streaming is the fragile case, and it is the case a longer SOT output would stress further.
- vLLM's LoRA gap means a LoRA-trained multi-talker adapter should be merged into the base weights before serving.

### Gaps
- Projector parameter count and an exact encoder/projector/LLM split summing to the total: not found.
- Official VRAM guidance: none exists (issue #20 went unanswered).
- GPU model used in the report's efficiency table: one third-party page says a single A100 with bfloat16, unverified.
- The full per-concurrency TTFT/RTF table could not be read; arxiv.org must be allowed to confirm it.
- No sherpa-onnx or OpenVINO benchmark for Qwen3-ASR was found.
- AuT pre-training data size conflicts between sources (about 40M hours in one summary of the ASR report, 20M hours in the Qwen3-Omni report); unresolved.

## 2. Fine-tuning practicalities for Qwen3-ASR and comparable Speech-LLMs (applies to approaches 1-3)

### Takeaway
An official fine-tuning script exists and takes plain audio + text JSONL, so SOT targets can be fed without code changes, but it gives no VRAM or time guidance and appears to do full fine-tuning rather than LoRA. I found no published GPU-memory or GPU-hour figure for fine-tuning Qwen3-ASR from either the maintainers or the community.

### Cited Findings
- [D] Official script: `finetuning/qwen3_asr_sft.py`. Data is JSONL with `audio` (WAV path) and `text` (transcript, optionally prefixed `language English<asr_text>`; `language None<asr_text>` if unknown). — [Qwen3-ASR finetuning](https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning)
- [D] Example command: `--batch_size 32 --grad_acc 4 --lr 2e-5 --epochs 1 --save_steps 200`; multi-GPU via `torchrun --nproc_per_node=2`; checkpoints load with `Qwen3ASRModel.from_pretrained`. — [Qwen3-ASR finetuning](https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning)
- [D] The page states no GPU type, VRAM requirement or training time, and does not say whether training is full or LoRA. Only hardware hint: FlashAttention 2 is recommended for lower memory, and `MAX_JOBS=4` when building it on machines with under 96 GB RAM. — [Qwen3-ASR finetuning](https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning)
- [S] Open issue #176 (May 26, 2026): the fine-tuning callback silently skips copying the preprocessor/chat template when `--model_path` is a Hub ID. — [Qwen3-ASR issues](https://github.com/QwenLM/Qwen3-ASR/issues)
- [D] Comparable recipe (Hao Shi et al., WavLM-Large + LLaMA): LoRA r=16, alpha=32, encoder and decoder frozen, projector trainable, 30 epochs, per-device batch 8, bf16, lr 3e-5. GPU type and hours are not stated. — [hshi-speech/Multi-talker-ASR-with-LLMs](https://github.com/hshi-speech/Multi-talker-ASR-with-LLMs)
- [D] Comparable recipe (MOSS-Transcribe-Diarize 0.9B, same Qwen3-0.6B-style decoder): `finetune.py` with per-device batch 1, 3 epochs, lr 1e-5, bf16, gradient checkpointing; default max sequence length 131,072 tokens. No VRAM or time stated. — [MOSS FINETUNING.md](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/blob/main/FINETUNING.md)

### Inferences
- Because the official script consumes free-form target text, a first SOT experiment is a data-preparation task (build mixtures, write serialized targets with a separator token), not an engineering task.
- The example batch size of 32 was written for the 1.7B model on unspecified hardware; it says nothing reliable about what fits on a given GPU for 0.6B.

### Gaps
- Training compute (GPU type x hours) for Qwen3-ASR fine-tuning: Not Reported anywhere I could reach.
- Community fine-tuning reports with memory/time numbers: none found. HF discussions could not be opened.
- Whether adding a new special token (speaker-change) works with the official script without modification: not documented.

## 3. Approach 1 — plain SOT fine-tuning, and how SOT changes LLM decoding cost

### Takeaway
SOT adds no modules and keeps the single-model pipeline, so its only deployment cost is longer autoregressive output. Published evidence that this cost is material comes from one LLM-based system: a LLaMA-1B decoder ran at RTF about 0.10-0.12 on LibriMix versus 0.004-0.011 for an encoder-only serialized-CTC model. I found no paper reporting single-talker vs multi-talker RTF for the same LLM decoder.

### Cited Findings
- [S] An Interspeech 2026 paper (Shi et al.) reports RTF 0.1150 on Libri2Mix and 0.0981 on Libri3Mix for a LLaMA-1B decoder, versus 0.0043 and 0.0106 for its encoder-only serialized-CTC model. — [Interspeech 2026 paper page](https://interspeech-2026-wiki.vercel.app/papers/shi26_interspeech/)
- [S] LLM-based SOT with a WavLM-Large encoder reached 9.0% WER on two-speaker LibriMix vs 9.2% for an AED baseline. — [arXiv 2408.17431](https://arxiv.org/pdf/2408.17431)
- [D] In the Hao Shi et al. code, the plain SOT recipe (`attention` mode, CE only) uses the unmodified encoder-projector-LLM path plus an added `<sc>` speaker-change token. — [hshi-speech/Multi-talker-ASR-with-LLMs](https://github.com/hshi-speech/Multi-talker-ASR-with-LLMs)
- [D] Qwen3-ASR's streaming mode caps each 2 s chunk at 32 new tokens by default in one re-implementation. — [antirez/qwen-asr](https://github.com/antirez/qwen-asr)
- [S] A persistent-acoustic-memory LLM multi-talker paper includes an RTF column in its efficiency table; values were not retrievable. — [arXiv 2603.27205](https://arxiv.org/html/2603.27205)

### Inferences
- The Libri3Mix RTF being lower than the Libri2Mix RTF in the cited table shows RTF does not simply scale with speaker count across test sets (clip lengths differ), so "output tokens double, therefore RTF doubles" should not be asserted without measuring on the team's own data.
- SOT in offline mode is compatible with every existing Qwen3-ASR serving path. SOT in the model's 2 s-chunk streaming mode is not obviously valid: utterance-level SOT emits speaker 1's full text before speaker 2's, which conflicts with a prefix-rollback scheme that assumes monotonic output. This is a structural inference, not a published result.
- A per-chunk token cap (32 in the cited port) is a concrete place where overlapped speech could truncate output in streaming.

### Gaps
- Single- vs multi-talker RTF on the same LLM-decoder model: Not Reported in sources found.
- Training compute for LLM-based SOT (GPU type x hours): Not Reported in the repo; papers not reachable.
- Added parameters: structurally zero apart from one token embedding, but no source states a figure.

## 4. Approach 2 — t-SOT (streaming)

### Takeaway
t-SOT is the published way to make serialized output streamable: tokens are ordered by time with a channel-change token, and the authors state the architecture and compute cost stay the same as a single-talker model. All reported results are on transducer models, not LLM decoders, and it needs token-level timestamps for training targets.

### Cited Findings
- [S] t-SOT serializes multi-speaker transcripts into one token stream sorted chronologically, inserting a special `<cc>` token when adjacent tokens belong to different virtual channels. — [arXiv 2202.00842](https://arxiv.org/pdf/2202.00842)
- [S] The authors state t-SOT achieves better accuracy than prior multi-talker models "while keeping the model architecture and computational cost the same as conventional single-talker models", with less inference cost than multi-branch streaming models. — [arXiv 2202.00842](https://arxiv.org/pdf/2202.00842)
- [S] A t-SOT transformer-transducer (TT-18) with 40 ms algorithmic latency outperformed all prior streaming multi-talker models on LibriSpeechMix. — [arXiv 2202.00842](https://arxiv.org/pdf/2202.00842)
- [S] LibriCSS average WER as summarized by a secondary site: 10.1 (TT-18, 160 ms), 9.0 (TT-36, 160 ms), 7.6 (2560 ms latency). Needs checking against the paper. — [emergentmind LibriCSS summary](https://www.emergentmind.com/topics/libricss)

### Inferences
- Since the token order is monotonic in time, t-SOT is the serialization that fits Qwen3-ASR's chunked prefix-rollback streaming; utterance-level SOT does not. No published work applies t-SOT to a Speech-LLM decoder in the sources I found.
- Building t-SOT targets needs word-level timestamps per speaker in the training mixtures. Qwen3-ForcedAligner-0.6B (11 languages, up to 5 minutes) is an available tool for that ([QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) [D]).
- t-SOT with two virtual channels is limited to two simultaneous speakers per stream; more need more channel tokens.

### Gaps
- t-SOT on LLM-decoder ASR: no reported results found.
- Training compute and RTF figures from the t-SOT papers: not retrievable.
- Direct t-SOT vs CSS cascade comparison: not found.

## 5. Approach 3 — SOT + per-speaker CTC / feature separator (Hao Shi et al.; Sidecar)

### Takeaway
These methods add a small separator (and CTC heads) between or after encoder layers. The Sidecar is reported at 8.7M parameters, 8.4% of a wav2vec 2.0-base system, trained with the backbone frozen. They break the "no new modules" property, and the public code targets WavLM + LLaMA, not Qwen3-ASR.

### Cited Findings
- [S] Sidecar (Meng et al., ICASSP 2023): only the Sidecar is trained, 8.7M parameters, 8.4% of all parameters, with the single-talker model frozen; 10.36% WER on 2-speaker LibriMix and 7.56% on LibriSpeechMix "with limited training". — [arXiv 2302.09908](https://arxiv.org/pdf/2302.09908)
- [S] The Sidecar is inserted between two encoder layers of a wav2vec 2.0-base ASR model. — [arXiv 2302.09908](https://arxiv.org/pdf/2302.09908)
- [S] A follow-up adds diarization to the Sidecar with only 768 additional parameters. — [arXiv 2305.16263](https://arxiv.org/pdf/2305.16263)
- [D] Hao Shi et al. repo contents: WavLM-Large encoder; LLaMA-3.2 1B/3B and LLaMA-3.1 8B decoders; an LSTM separator that splits encoder features into N per-speaker streams; per-speaker CTC heads; hybrid loss alpha*CE + (1-alpha)*CTC with alpha = 0.7; cross-attention adapter variants (`tiny`, `gatetiny`, `adapgatetiny`, `ctcaware`). — [hshi-speech/Multi-talker-ASR-with-LLMs](https://github.com/hshi-speech/Multi-talker-ASR-with-LLMs)
- [D] Training is staged: `ctc` mode pre-trains the separator, then `hybrid` mode trains all components; a CTC-only decoding stage "uses the per-speaker CTC heads directly". — [hshi-speech/Multi-talker-ASR-with-LLMs](https://github.com/hshi-speech/Multi-talker-ASR-with-LLMs)
- [D] The repo lists no WER numbers, no RTF, no training hardware, and no license. The papers are ASRU 2025 (Serialized Output Prompting) and SLT 2024 (overlapped encoding separation). — [hshi-speech/Multi-talker-ASR-with-LLMs](https://github.com/hshi-speech/Multi-talker-ASR-with-LLMs)
- [S] The SOP paper trains in three stages, starting with SOT, and extracts talker-specific serialized prompts to guide the LLM, evaluated on LibriMix 2- and 3-talker. — [arXiv 2509.04488](https://arxiv.org/abs/2509.04488)

### Inferences
- In the SOP design the separator and CTC heads produce prompts consumed by the LLM, so they are needed at inference; in designs where per-speaker CTC is only an auxiliary loss they can be dropped. The repo README does not state which case applies to each recipe; the "needed at inference" reading for SOP follows from the method description, not an explicit statement.
- The absence of a license file means the code cannot be assumed reusable in a product; it is a reference implementation to re-write.
- The cited encoder runs at 50 Hz (WavLM) while AuT outputs 12.5 Hz; whether a feature separator works as well on 12.5 Hz features is untested in the literature I found.

### Gaps
- Parameter count of the LSTM separator and CTC heads in Hao Shi et al.: Not Reported in the repo.
- RTF, VRAM, GPU hours for approach 3 on an LLM decoder: Not Reported.
- Streaming capability: not addressed in the sources.

## 6. Approach 4 — multi-output / PIT ASR

### Takeaway
I found no cost figures specific to PIT multi-output ASR in this session. The only relevant published statement is comparative: the t-SOT authors say multi-branch streaming models have higher inference cost and a more complex architecture than a single serialized output.

### Cited Findings
- [S] Prior streaming multi-talker models used multiple output branches; t-SOT has "less inference cost and a simpler model architecture" by comparison. — [arXiv 2202.00842](https://arxiv.org/pdf/2202.00842)

### Inferences
- For an LLM decoder, a multi-output design means either several decoder passes or several output heads, which the Qwen3-ASR serving stack (vLLM, one text stream per request) does not provide. This is a structural inference.

### Gaps
- Added parameters, RTF, VRAM, training compute for PIT-based ASR: Not Reported in sources reached. This approach is under-researched in these notes.

## 7. Approach 5 — target-speaker / speaker-conditioned ASR (NVIDIA multitalker Parakeet, DiCoW)

### Takeaway
NVIDIA's model avoids serialization by running one ASR instance per speaker, conditioned by a streaming Sortformer diarizer, so compute scales with speaker count and a second model is mandatory. It is English-only, under the NVIDIA Open Model License, and it is a transducer, not a Speech-LLM, so it is a comparison baseline rather than something to graft onto Qwen3-ASR.

### Cited Findings
- [S] The model card says the model requires deploying one model instance per speaker, with the number of instances matching the number of speakers; each instance focuses on one speaker, which avoids the permutation problem under heavy overlap. — [NVIDIA model card](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1)
- [S] It uses speaker kernel injection at some layers of the FastConformer encoder and needs a separate Sortformer diarizer; the example loads `nvidia/diar_streaming_sortformer_4spk-v2.1`. — [NVIDIA model card](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1)
- [S] Latency is set by `att_context_size` in 80 ms frames; the example uses `[70,13]`, and the card metadata lists a 1.04 s input buffer for one evaluation config. A community port describes the default as 1.12 s and says four latency settings are selectable. — [NVIDIA model card](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1); [handy-computer GGUF port](https://huggingface.co/handy-computer/multitalker-parakeet-streaming-0.6b-v1-gguf)
- [D] A Rust runtime lists the model as English only, streaming, per-speaker output, fed in chunks of 17,920 samples (about 1.12 s at 16 kHz), and requiring a separate diarization ONNX file. — [altunenes/parakeet-rs](https://github.com/altunenes/parakeet-rs)
- [S] License field: `other`, name `nvidia-open-model-license`. — [NVIDIA model card README](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1/blob/main/README.md)
- [S] Training datasets listed on the card: AMI, NOTSOFAR1, Fisher, MMLPC, LibriSpeech train-clean-100. — [NVIDIA model card](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1)
- [S] The underlying paper (Wang, Park et al., Interspeech 2025) reports a smaller research system of 238M parameters: 114M FastConformer transducer + 123M Sortformer + 1.1M speaker injection module. — [arXiv 2506.22646](https://arxiv.org/abs/2506.22646)
- [S] A third-party GGUF port measured 19.35% cpWER on AMI-IHM test (16 meetings) in default kernel mode; this is the port's measurement, not NVIDIA's. — [handy-computer GGUF port](https://huggingface.co/handy-computer/multitalker-parakeet-streaming-0.6b-v1-gguf)
- [D] For the related offline Multitalker TDT model, the Rust runtime reports cpWER on two overlapping speakers falling from 36.7% (plain TDT + diarization) to 4.4% on LibriSpeechMix. — [altunenes/parakeet-rs](https://github.com/altunenes/parakeet-rs)
- [S] DiCoW conditions Whisper on diarization output through frame-level diarization-dependent transforms (FDDT): four class-specific affine transforms (silence, target, non-target, overlap) inserted in the encoder layers. A follow-up initializes from a DiCoW checkpoint based on Whisper-large-v3-turbo. — [arXiv 2501.00114](https://arxiv.org/pdf/2501.00114); [arXiv 2510.03723](https://arxiv.org/pdf/2510.03723)

### Inferences
- The speaker-injection module itself is tiny (1.1M in the paper), so the cost of this family is not added parameters; it is the mandatory diarizer plus N parallel ASR passes. The diarizer in the paper is larger than the ASR model (123M vs 114M).
- DiCoW has the same shape: one decoding pass per diarized speaker plus an external diarizer. Neither keeps a single-model pipeline.
- Community int8 ONNX, GGUF and MLX conversions exist for the NVIDIA model, which suggests CPU deployment is practical for it, but I have no measured CPU RTF.

### Gaps
- Maximum speaker count, the four latency values, NVIDIA's own cpWER table, training hours, and reported RTF/VRAM: on the model card, which could not be opened.
- DiCoW added parameters, per-speaker inference cost, license, training GPUs: Not Reported in sources reached.
- Whether the 0.6B base is described as Parakeet or Nemotron-Speech-Streaming differs between cached versions of the card.

## 8. Approach 6 — separation front-end / CSS cascade

### Takeaway
Separation front-ends range from 2.6M to 55.7M parameters, but compute varies far more than size: reported RTF runs from 0.16 (Conv-TasNet) to 12.40 (SepFormer) in one comparison, and the strongest TF-domain models are described as not real-time friendly. Published evidence consistently shows that an off-the-shelf ASR does not reliably benefit from a separator unless one side is fine-tuned on the other's output. I found no clean same-test-set cascade-vs-SOT table.

### Cited Findings

**Size and compute**
- [S] WSJ0-2mix parameter counts and SI-SDRi from one survey: MossFormer2 55.7M / 24.1 dB; TF-GridNet 14.5M / 23.5 dB; SepFormer 26.0M / 22.3 dB; DPRNN 2.9M / 18.8 dB; Conv-TasNet 5.1M / 15.3 dB. Other papers list 14.4M for TF-GridNet and 2.6M for DPRNN. — [arXiv 2508.10830](https://arxiv.org/pdf/2508.10830)
- [S] One paper reports MACs of 10.09G (Conv-TasNet), 86.31G (DPRNN), 74.94G (SepFormer) with RTFs of 0.16, 1.18 and 12.40 respectively. Hardware for those RTFs was not captured. — [OpenReview: Separate and Reconstruct](https://openreview.net/pdf?id=99y2EfLe3B)
- [S] A MossFormer2 table reports RTF 0.053 on a V100 GPU. — [emergentmind MossFormer2 summary](https://www.emergentmind.com/topics/mossformer2-separated-speech)
- [S] TF-GridNet (14M params) and SPMamba (6M) require 445 and 238 GMAC/s and are described as not real-time friendly. — [arXiv 2510.12275](https://arxiv.org/pdf/2510.12275)
- [S] TF-Locoformer: medium model 15.0M params, 22.1 dB SI-SNRi on WSJ0-2mix; small model 5.0M params and 85.9 G MACs/s. — [arXiv 2408.03440](https://arxiv.org/pdf/2408.03440); [arXiv 2607.21128](https://arxiv.org/pdf/2607.21128)

**Cascade problems**
- [S] On single-channel LibriCSS, a separation model trained with a mask objective and then fine-tuned with ASR-oriented criteria beat the previous best numbers, i.e. ASR-aware training of the separator was needed for the best result. — [Wu et al., Interspeech 2021](https://www.isca-archive.org/interspeech_2021/wu21f_interspeech.pdf)
- [S] The same paper reports a large single- vs multi-channel gap on LibriCSS, about 8.2% absolute WER at 40% overlap. — [Wu et al., Interspeech 2021](https://www.isca-archive.org/interspeech_2021/wu21f_interspeech.pdf)
- [S] A 2026 paper states that off-the-shelf ASR models trained without exposure to a separation front-end's output distribution often fail to benefit from it; its ASR is fine-tuned on separated and close-talk data mixed together. — [arXiv 2605.19695](https://arxiv.org/pdf/2605.19695)
- [S] In a music-mixed speech study, fine-tuning only the separation front-end was less effective than fine-tuning the ASR back-end, and joint fine-tuning did not improve over ASR-only fine-tuning while significantly degrading separation quality. — [arXiv 2008.12048](https://arxiv.org/pdf/2008.12048)
- [S] The CHiME-8 default ASR favours GSS because it was trained on GSS-enhanced signals; the CHiME-7/8 review also notes that re-tuning for NOTSOFAR-1 degraded the baseline on other scenarios. — [MERL TR2026-008](https://www.merl.com/publications/docs/TR2026-008.pdf)

**Cascade vs end-to-end numbers (assembled from separate papers, not like-for-like)**
- [S] Conformer CSS baseline, single-channel LibriCSS utterance-wise with an E2E Transformer ASR: 17.1 WER at 40% overlap, 10.6 average (as quoted in the WavLM paper). — [arXiv 2110.13900](https://arxiv.org/pdf/2110.13900)
- [S] t-SOT streaming models on LibriCSS average 7.6-10.1 WER depending on size and latency (secondary summary; evaluation mode not confirmed to match the CSS figure). — [emergentmind LibriCSS summary](https://www.emergentmind.com/topics/libricss)
- [S] A 2025 paper argues for decoupling separation from recognition and cites an ESPnet SOT-Conformer baseline at 23.3% WER and a WavLM-feature variant at 17.1%; the test-set label was ambiguous in the extract (LibriCSS vs Libri2Mix). — [arXiv 2503.17886](https://arxiv.org/pdf/2503.17886)

### Inferences
- The cascade always adds a second model at inference and roughly doubles ASR work, because each separated stream is decoded separately. With a 0.6B LLM decoder, that is two full autoregressive passes for two speakers versus one longer pass for SOT.
- The models with the best separation scores (TF-GridNet, MossFormer2, SepFormer) are the ones with the worst reported compute, so a CPU-deployable cascade would be pushed toward Conv-TasNet-class models, which trail by 7-9 dB SI-SDRi on WSJ0-2mix.
- Given the mismatch evidence, a cascade realistically still requires fine-tuning Qwen3-ASR on separator output, so it does not save the fine-tuning effort that SOT needs.
- WSJ0-2mix scores are for fully overlapped clean two-speaker mixtures; they do not show behaviour on partially overlapped conversational audio.

### Gaps
- Same-hardware CPU RTF for the listed separators: none found; RTFs across papers use different hardware and input lengths and cannot be compared.
- A published table comparing cascade and SOT on one test set with one ASR back-end: not found.
- NOTSOFAR-specific single-channel cascade results: not retrieved.
- ClearerVoice / SpeechBrain model-card runtime figures: HF not reachable.

## 9. Approach 7 — external baseline: MOSS-Transcribe-Diarize 0.9B

### Takeaway
MOSS-Transcribe-Diarize is the closest architectural sibling to Qwen3-ASR-0.6B (Whisper-medium-style encoder + Qwen3-0.6B-style decoder, Apache-2.0) and ships a fine-tuning script and data format publicly. Its documentation does not claim or describe handling of simultaneous speech, and its own results show a large gap between CER and speaker-attributed cpCER on harder material.

### Cited Findings
- [D] Architecture: "Qwen3-0.6B style causal decoder"; Whisper-Medium encoder configuration; WhisperFeatureExtractor (16 kHz, 80 mel bins, 30 s chunks); 4x temporal merge + MLP adaptor; audio features replace `<|audio_pad|>` embeddings. Stated size 0.9B. License Apache-2.0. 50+ languages. — [OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
- [S] A conversion card gives 908,513,280 total parameters: about 312M audio side and about 596M text decoder; 24-layer encoder, 28-layer decoder. — [soniqo ONNX FP16 card](https://huggingface.co/soniqo/MOSS-Transcribe-Diarize-0.9B-ONNX-FP16)
- [D] Output format is a single serialized stream: `[start_time][Sxx]text[end_time]` with speaker tags such as `[S01]`. — [OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
- [D] Overlapping speech is not addressed in the README. It says the model handles "long, messy, multi-speaker recordings" without describing overlap. The fine-tuning guide likewise does not say how overlapping segments are represented or ordered. — [OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize); [FINETUNING.md](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/blob/main/FINETUNING.md)
- [S] A conversion page warns the source model can produce malformed or overlapping timestamps and can miss speakers. — [handy-computer GGUF card](https://huggingface.co/handy-computer/moss-transcribe-diarize-gguf)
- [D] Training code is public: `finetune.py` and `FINETUNING.md`. Data is JSONL, one conversation per record, with three messages: text prompt, audio path, reference transcript in the output format (example `[0.00][S01]Welcome[0.72]`). — [FINETUNING.md](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/blob/main/FINETUNING.md)
- [D] Speed on a single H100 with SGLang Omni: short clips, concurrency 1, mean RTF 0.0612 (29.76 audio-s/s); concurrency 16, RTF 0.0922 (81.98 audio-s/s). Long AISHELL-4 sessions: concurrency 1, RTF 0.0197 (50.64 audio-s/s); concurrency 16, RTF 0.1237 (98.83 audio-s/s). — [OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
- [D] VRAM is not stated. The README warns eager attention memory "grows quadratically with audio length and OOMs on long recordings". Default output cap is 5120 new tokens, to be raised to 65536 for long audio. — [OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
- [D] 0.9B results (CER / cpCER): AISHELL-4 14.84 / 15.83; AliMeeting 24.86 / 22.17; Podcast 5.97 / 7.37; Movies 6.36 / 12.76. A stronger "Pro" model is available only via an online playground. — [OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
- [D] Serving: Transformers 5.x with `trust_remote_code=True`; vLLM needs a pinned nightly; SGLang Omni is the recommended backend and targets CUDA 13. — [OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
- [S] Community CoreML (INT8, FP16), ONNX FP16 and GGUF conversions exist. — [aufklarer CoreML INT8](https://huggingface.co/aufklarer/MOSS-Transcribe-Diarize-0.9B-CoreML-INT8); [handy-computer GGUF card](https://huggingface.co/handy-computer/moss-transcribe-diarize-gguf)

### Inferences
- MOSS is evidence that a sub-1B, single-model, single-output-stream Speech-LLM can emit speaker-tagged multi-speaker transcripts with ordinary fine-tuning tooling, which is the same deployment shape as SOT on Qwen3-ASR.
- AliMeeting is the overlap-heavy set in its table and has the worst CER (24.86), which is consistent with overlap being a weak point, but the README does not attribute the error to overlap, so this stays an inference.
- Its segment-with-timestamps format can in principle express overlapping segments (two segments with intersecting time ranges), but whether the released model does so is undocumented.
- MOSS throughput (about 82-99 audio-s/s at concurrency 16 on an H100) and Qwen3-ASR's (2,000 audio-s/s at concurrency 128, GPU unconfirmed) were measured with different backends, concurrency and tasks, and are not comparable.

### Gaps
- Whether it truly transcribes simultaneous speech: not documented; would need a direct test on overlapped clips.
- VRAM, training data composition, training compute: Not Reported.
- LoRA vs full fine-tuning in `finetune.py`: not stated.
- Maximum audio length: not stated (only the 131,072-token training sequence cap).

## 10. Cross-approach summary of reported structural facts

### Takeaway
Only plain SOT and t-SOT keep the stock single-model pipeline with zero added inference modules; every other approach adds either an in-model module (separator/CTC heads) or a whole second model (diarizer or separator) and, for approaches 5 and 6, one ASR pass per speaker. Reported cost numbers are sparse: most cells below are "Not Reported", which is itself a finding.

### Cited Findings

| Approach | Added parameters (reported) | Extra model at inference | Single-model pipeline kept | Streaming | Reported speed | Reported training compute |
|---|---|---|---|---|---|---|
| 1. SOT fine-tune | Not Reported (no new modules in recipe) | No | Yes | Not Reported for LLM decoders | LLaMA-1B SOT decoder RTF 0.1150 (Libri2Mix), 0.0981 (Libri3Mix) | Not Reported |
| 2. t-SOT | Authors: same architecture and cost as single-talker | No | Yes | Yes (40 ms-2560 ms latency reported, transducer models) | Not Reported | Not Reported |
| 3. SOT + separator / CTC | Sidecar 8.7M (8.4% of a wav2vec 2.0-base system); Shi et al. separator Not Reported | No separate model; in-model modules | Modified single model | Not Reported | Not Reported | Not Reported |
| 4. PIT multi-output | Not Reported | Not Reported | Not Reported | Prior multi-branch streaming models exist | Not Reported | Not Reported |
| 5. NVIDIA multitalker Parakeet | Speaker injection 1.1M (research system) | Yes: streaming Sortformer (123M in research system) | No; one ASR instance per speaker | Yes (about 1.12 s default chunk) | Not Reported | Not Reported |
| 6. Separation cascade | Separator 2.6M-55.7M | Yes: separator | No; one ASR pass per stream | Model-dependent | RTF 0.16 (Conv-TasNet) to 12.40 (SepFormer); MossFormer2 0.053 on V100 | Not Reported |
| 7. MOSS-TD 0.9B | n/a (separate 908.5M model) | No | Yes | Not Reported | RTF 0.0612 (c=1), 0.0922 (c=16) on H100 | Not Reported |

Sources for the table rows: [Interspeech 2026 paper page](https://interspeech-2026-wiki.vercel.app/papers/shi26_interspeech/); [arXiv 2202.00842](https://arxiv.org/pdf/2202.00842); [arXiv 2302.09908](https://arxiv.org/pdf/2302.09908); [arXiv 2506.22646](https://arxiv.org/abs/2506.22646); [altunenes/parakeet-rs](https://github.com/altunenes/parakeet-rs); [arXiv 2508.10830](https://arxiv.org/pdf/2508.10830); [OpenReview: Separate and Reconstruct](https://openreview.net/pdf?id=99y2EfLe3B); [emergentmind MossFormer2 summary](https://www.emergentmind.com/topics/mossformer2-separated-speech); [OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize).

### Inferences
- On the team's stated priorities (single model, no front-end, existing CPU ports, existing fine-tuning script), plain SOT has the lowest integration cost on the evidence here, and t-SOT is the variant to consider if streaming must be kept. This ranking rests on structural facts, since measured cost data for the alternatives is largely unpublished.
- No approach has a published VRAM figure. VRAM comparisons in a final report would have to come from the team's own measurement.
- Training compute (GPU type x hours) was not reported in any source I could reach, for any approach.

### Gaps
- Nearly all RTF/VRAM/training-compute cells are empty because the primary papers (arXiv) and model cards (Hugging Face) were unreachable from this session. Allowing `arxiv.org` and `huggingface.co` would let a follow-up pass fill: the Qwen3-ASR efficiency table, the NVIDIA model card (latency options, speaker limit, cpWER, training hours), the SOP / Sidecar / t-SOT / DiCoW experimental-setup sections, and HF community discussions on fine-tuning.
- PIT multi-output ASR (approach 4) has effectively no sourced cost data in these notes.
