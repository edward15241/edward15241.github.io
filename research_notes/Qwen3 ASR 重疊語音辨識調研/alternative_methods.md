# Alternative approaches to single-channel overlapped speech recognition (beyond SOT / t-SOT / SOP / per-speaker CTC / Sidecar / PIT / Parakeet-multitalker / DiCoW / CSS cascade / MOSS-TD)

> Research date: 2026-11-10. Scope: methods NOT in the "already covered" list, 2021–2026, assessed for attachment to Qwen3-ASR-0.6B (audio encoder → projector → Qwen3 LLM decoder).
>
> **Verification caveat (important for the report writer):** in this session `arxiv.org` and `ar5iv.labs.arxiv.org` were not reachable by direct fetch (network allow-list). All findings below come from search-engine result summaries of the primary pages (arXiv abstract/HTML, ISCA archive, GitHub, Hugging Face). Abstract-level claims are reliable; **individual table numbers could usually not be checked against the PDF**. Every number is tagged either `[abstract]` (stated in the paper abstract as surfaced by search) or `[snippet/secondary — verify]` (pulled from a table fragment or third-party summary). Anything not found is marked "Not Reported". Nothing is estimated.

---

## Q1. Prompt-/instruction-based decoding with an LLM decoder (attribute-conditioned transcription without enrollment; encoder once, decoder N times)

### Takeaway
This is the best-evidenced alternative that fits an encoder→projector→LLM model almost unchanged: MT-LLM (CUHK, 2024) shows a LoRA-tuned LLM can transcribe "the first/second speaker", "the male/female speaker", "the speaker speaking language X", or "the speaker who says keyword Y" from text instructions alone, and can also do all-speaker transcription in the same model. No paper was found that measures the specific "run the decoder once per ordinal prompt and concatenate" recipe against single-pass SOT on cost or accuracy.

### Cited Findings
**MT-LLM — "Large Language Model Can Transcribe Speech in Multi-Talker Scenarios with Versatile Instructions" (Meng, Hu, et al., CUHK; arXiv 2409.08596, Sept 2024)**
- Core idea: one instruction-following speech-LLM handles (a) transcribe all talkers, (b) transcribe a target talker given reference audio, (c) transcribe a talker selected by an attribute; attributes are sex, occurrence order, language, and keyword spoken. v1 describes six instruction types. — [arXiv 2409.08596 abstract/HTML](https://arxiv.org/html/2409.08596v2)
- Architecture: Llama 2 as the LLM; Whisper encoder for semantic context plus WavLM multi-layer features for speaker/acoustic cues (dual encoder); LLM tuned with LoRA. — [arXiv 2409.08596](https://arxiv.org/pdf/2409.08596)
- Loss: Not Reported in retrieved text (standard next-token cross-entropy is implied by "LoRA fine-tuned LLM" but was not confirmed).
- Evaluation sets: 2- and 3-speaker LibriSpeechMix and an En-De-Mix (cross-lingual mixture) test set; minimum-error permutation is used for multi-talker WER. — [arXiv 2409.08596v2 HTML](https://arxiv.org/html/2409.08596v2)
- Result `[snippet/secondary — verify]`: 2-speaker LibriSpeechMix WER 5.2% vs 4.9% for the task-specific SOT-Conformer. — [arXiv 2409.08596v2 HTML (as summarised by search)](https://arxiv.org/html/2409.08596v2)
- Per-attribute WERs (order / sex / language / keyword): Not Reported in retrieved text.
- Limitation noted by a later paper: LLM-based SOT "performs well in the two-talker scenario but fails to fully leverage LLMs in more complex conditions, such as three talkers". — [ResearchGate listing of 2409.08596 citing context](https://www.researchgate.net/publication/390538905_Large_Language_Model_Can_Transcribe_Speech_in_Multi-Talker_Scenarios_with_Versatile_Instructions)
- Limitation noted by SpeakerLM authors: MT-LLM "is limited to detecting speaker change points and does not explicitly tackle speaker attribution" (irrelevant for this team, which needs no attribution). — [SpeakerLM, arXiv 2508.06372](https://arxiv.org/html/2508.06372v3)
- Code: a repository exists at cuhealthybrains/MT-LLM, described by one search summary as "samples" and by another as "the implementation"; whether training code and weights are included was not verified. — [GitHub cuhealthybrains/MT-LLM](https://github.com/cuhealthybrains/MT-LLM)

**"Empowering Whisper as a Joint Multi-Talker and Target-Talker Speech Recognition System" (Meng et al., arXiv 2407.09817, Interspeech 2024)** — mentioned only for the parts not covered elsewhere (Sidecar itself is covered).
- Whisper frozen; Sidecar separator in the encoder; a Target Talker Identifier that needs a three-second enrollment clip; soft prompt tuning on the decoder. — [arXiv 2407.09817](https://arxiv.org/html/2407.09817v2)
- Reports acceptable zero-shot multi-talker ASR on the Mandarin AishellMix set (trained on English). — [arXiv 2407.09817](https://arxiv.org/pdf/2407.09817)
- Result `[snippet/secondary — verify]`: an aggregator lists WER 6.99 on LibriSpeech2Mix and 11.4 on LibriSpeech3Mix. — [sotaverified listing](https://sotaverified.org/papers/empowering-whisper-as-a-joint-multi-talker)
- Code: github.com/LingweiMeng/Whisper-Sidecar. — [arXiv 2407.09817](https://arxiv.org/pdf/2407.09817)

**"Extending Whisper with prompt tuning to target-speaker ASR" (Ma, Peng, Shao, Li, Liu; arXiv 2312.08079, ICASSP 2024)**
- Target-speaker embeddings plus trainable soft prompts fed to a frozen Whisper; about 1% task-specific parameters; "comparable to state-of-the-art full fine-tuning"; keeps Whisper's inverse text normalisation and timestamps. Requires enrollment (speaker embedding). WER numbers: Not Reported in retrieved text. — [arXiv 2312.08079](https://arxiv.org/abs/2312.08079v2)

**Enrollment-conditioned LLM TS-ASR (comparison points; they need a reference clip, which this team does not want)**
- Xiaomi-CocktailASR-1 (arXiv 2609.11274, Sept 2026): LLM-based end-to-end TS-ASR using reference audio as a "voiceprint prompt", no explicit separation; claims competitive single-speaker performance, outputs empty text when the target is absent, and has an optional Chain-of-Thought mode. Results `[snippet — verify]`: TS-WER LibriMix 2mix 4.84%→4.11%; LibriSpeechMix 2mix 5.40%→2.90%; AMI-SDM 21.81% (vs SQ-Whisper 22.0%); AliMeeting-Far 20.63% (vs prior 27.5%). — [arXiv 2609.11274](https://arxiv.org/html/2609.11274v1). A news site says it was open-sourced (secondary, unverified). — [ProPakistani](https://propakistani.pk/2026/09/14/xiaomis-new-open-source-ai-can-isolate-one-speaker-from-overlapping-voices/)
- "Thinking in cocktail party: Chain-of-Thought and reinforcement learning for target speaker ASR" (Zhang, Su, et al.; arXiv 2509.15612): the model first reasons about number of speakers, each speaker's gender, and similarity to the reference, then transcribes; trained on regular data, then CoT data, then GRPO-style RL with a format+WER reward. Single-speaker WER 3.65% `[snippet — verify]`. Relative gains are reported inconsistently by secondary sources (21% on 2-mix / 16.2% on 3-mix in one; ~26% in another) — conflicting, verify in paper. — [arXiv 2509.15612](https://arxiv.org/abs/2509.15612). A secondary review notes that emitting a CoT before the transcript increases latency. — [emergentmind topic page](https://www.emergentmind.com/topics/target-speaker-automatic-speech-recognition-asr)
- SQ-Whisper (arXiv 2412.05589): trainable speaker queries from enrollment; a secondary review reports WER rising from roughly 20% to over 70% when the enrolled speaker is not in the mixture `[secondary — verify]`. — [arXiv 2412.05589](https://arxiv.org/abs/2412.05589); [emergentmind topic page](https://www.emergentmind.com/topics/target-speaker-asr-with-whisper)

**One model, two modes via conditioning dropout**
- "Asymmetric Classifier-Free Guidance for Target-Speaker ASR" (Guan & Whitehill, arXiv 2609.30476, Sept 2026): one shared Whisper is trained so the speaker-conditioned branch outputs the target transcript and the unconditioned branch outputs serialized multi-speaker transcripts; guidance scale w=1 gives TS-ASR, w=0 gives multi-speaker ASR. Up to 21.8% relative WER reduction over a condition-only baseline under domain shift and 5.6% over standard conditional decoding `[abstract]`. Datasets: Not Reported in retrieved text. — [arXiv 2609.30476](https://arxiv.org/abs/2609.30476)

**Enrollment-free cues other than text attributes**
- "Speaker Targeting via Self-Speaker Adaptation for Multi-talker ASR" (NVIDIA, Interspeech 2025, arXiv 2506.22646): the only extra input is the target speaker's binary speech-activity mask (from diarization); cpWER 4.0% (1-mix) and 5.6% (2-mix) at 560 ms latency `[snippet — verify]`. Belongs to the covered NVIDIA/DiCoW family; listed for comparison only. — [ISCA archive](https://www.isca-archive.org/interspeech_2025/wang25y_interspeech.pdf)
- Searched for TS-ASR conditioned on an onset time, "the louder speaker", or "the speaker who starts later" as a text prompt: no paper found other than MT-LLM's "occurrence order" attribute. — [search record: arXiv 2409.08596](https://arxiv.org/pdf/2409.08596)

### Inferences (researcher's assessment, not from sources)
- Attachability to Qwen3-ASR: high. The recipe is data + prompt only: same encoder, same projector, LoRA on the LLM (and probably on the encoder top layers/projector), with training targets "speaker who starts first → text A", "speaker who starts second → text B", and "all speakers → A <sep> B". No new modules are mandatory; MT-LLM's WavLM branch is an add-on (see Q3).
- Cost: audio is encoded once; the audio prefix can be kept in the KV cache and only the short instruction + answer are re-decoded per speaker. Total generated tokens are about the same as SOT (N transcripts either way), so RTF should be close to single-pass SOT, plus one extra short prompt per speaker; batch the N prompts to hide the latency. This is reasoning, not a measured result.
- Risk 1: per-speaker passes are independent, so the two passes can transcribe the same speaker twice or drop a speaker (no cross-pass exclusivity). SOT's single pass conditions speaker 2 on speaker 1's text; multi-pass does not unless the first transcript is put back in the prompt (see Q2).
- Risk 2: "first/second" is ill-defined when both speakers start within a few hundred ms; sex/loudness attributes fail for same-sex or equal-level mixtures. A joint model that offers both "all speakers" and "ordinal" prompts lets the team compare the two decodings on the same weights.
- Risk 3: the number of passes must be known; a cheap "how many speakers?" prompt or head (Q6) is needed to avoid hallucinating a second speaker on single-speaker audio.
- Enrollment-based LLM TS-ASR (Xiaomi-CocktailASR-1, CoT-RL) is evidence that small-to-mid LLM decoders can pick one voice out of a mixture and keep single-speaker accuracy, but it does not match the team's "no enrollment" requirement.

### Gaps
- MT-LLM per-task WERs, LoRA rank, trainable parameter count, training hours: not retrievable in this session (arXiv blocked).
- No published head-to-head of "N ordinal-prompt passes" vs "one SOT pass" on the same speech-LLM (accuracy, duplicate/omission rate, RTF).
- No published prompt-attribute results for Mandarin or Hokkien mixtures; MT-LLM's cross-lingual set is En-De.

---

## Q2. Iterative / recursive / conditional-chain decoding and text-conditioned residual decoding

### Takeaway
Conditional-chain multi-speaker ASR exists and is published with code (ESPnet fork), but only for pre-LLM encoder/CTC models and with weak absolute WERs; no paper was found that applies "decode speaker 1, then condition on its text to decode the remaining speaker" to a speech-LLM as a separate pass. In an LLM decoder, single-pass SOT already is a conditional chain in text space, so the practical novelty is small.

### Cited Findings
- "Multi-Speaker ASR Combining Non-Autoregressive Conformer CTC and Conditional Speaker Chain" (Guo, Chang, Watanabe, Xie; Interspeech 2021, arXiv 2106.08595): each speaker's transcript is produced in turn from the mixture plus the previously estimated speakers' features; each step is a non-autoregressive CTC pass, so inference steps = number of speakers; Conformer encoder + intermediate CTC loss. — [arXiv 2106.08595](https://arxiv.org/pdf/2106.08595.pdf)
- Results `[abstract]`: WER 22.3% on WSJ0-Mix and 24.9% on LibriMix; with variable speaker counts in training, 19.9% (WSJ0-2mix) and 34.3% (WSJ0-3mix); about 1/7 the latency of the PIT-Conformer autoregressive model. — [ISCA archive](https://www.isca-archive.org/interspeech_2021/guo21_interspeech.html)
- Code: authors' ESPnet fork, `conditional-multispk` branch. — [arXiv 2106.08595](https://arxiv.org/pdf/2106.08595.pdf)
- Transcript-conditioned separation: TokenSplit (Google, Interspeech 2023, arXiv 2308.10415) is a Transformer seq2seq model over discrete tokens trained to separate and transcribe each source and to refine separation when transcripts are given; a secondary summary reports that on 3-second Libri2Mix segments, supplying the target transcript improves differential WER 26.6%→12.1% (direct) and 17.2%→10.9% (refine) `[secondary — verify]`. — [ISCA archive](https://www.isca-archive.org/interspeech_2023/erdogan23_interspeech.html); [arXiv 2308.10415](https://arxiv.org/abs/2308.10415)
- The original conditional-chain paper (Shi et al., "Sequence to Multi-Sequence Learning via Conditional Chain Mapping for Mixture Signals", NeurIPS 2020) was not returned by search in this session; it is known to the researcher from prior knowledge only and is not re-verified here. — [arXiv 2106.08595 (cites it as the authors' earlier model)](https://arxiv.org/pdf/2106.08595.pdf)

### Inferences
- For an autoregressive LLM decoder, "decode speaker 1 → condition on it → decode speaker 2" is what SOT does inside one sequence. A two-call variant (pass 2 prompt = "Speaker 1 said: …; transcribe the other speaker") adds a second prompt but no new information, and costs slightly more. Its only practical advantages are (i) an early-exit: skip pass 2 when a counter says one speaker, and (ii) the ability to stream speaker 1's text to the user before speaker 2 is decoded.
- Feature-level chains (subtracting or masking speaker-1 evidence in encoder space before pass 2) would need a new module between encoder and projector and re-running the projector; no speech-LLM evidence exists, so this is research risk, not an engineering option.
- The 2021 numbers (LibriMix 24.9% WER) are far behind current SOT/LLM systems (Q1, Q5 report 5–9% on comparable 2-mix sets), so the published chain models are not competitive as-is.

### Gaps
- No 2023–2026 paper found on recursive/residual decoding for speech-LLMs. Explicitly: searched for "conditional chain", "speaker-conditional chain", "recursive", "text-conditioned" multi-speaker ASR; only the 2021 work and TokenSplit surfaced.

---

## Q3. Multi-/dual-encoder fusion and mixture-aware SSL encoders as an extra branch

### Takeaway
Adding a second, mixture-aware encoder (WavLM / mHuBERT) next to a Whisper-style encoder is a recurring design in multi-talker and conversational speech-LLMs (MT-LLM, SHNU MLC-SLM system), and SSL models pre-trained on mixtures (WavLM, Cocktail HuBERT) give large gains over plain HuBERT on Libri2Mix. No ablation isolating the benefit of the second encoder for an LLM decoder could be retrieved, and the extra branch costs RTF/VRAM.

### Cited Findings
- MT-LLM uses Whisper encoder (semantic) + WavLM multi-layer features (speaker/acoustic) into Llama 2 with LoRA. — [arXiv 2409.08596](https://arxiv.org/pdf/2409.08596)
- "Advancing Multi-talker ASR Performance with Large Language Models" (arXiv 2408.17431, SLT 2024): WavLM encoder + Vicuna-7B with LoRA, SOT-style; WavLM Base+/Large chosen because they were pre-trained with a substantial amount of overlapped speech. `[snippet — verify]`: LibriMix about 9.0% WER (WavLM Large + LLM) vs 9.2% for the AED baseline; on AMI, absolute WER −2.9 and cpWER −3.1 vs AED (cpWER reported as ~20.4% or 21.0% by different summaries — conflicting). The paper also found the LLM method had lower speaker-counting accuracy in complex scenarios. — [arXiv 2408.17431](https://arxiv.org/html/2408.17431)
- CMT-LLM (He et al., Interspeech 2025, arXiv 2506.12059): combines multi-talker overlapped ASR with contextual biasing (rare-word list in the prompt) on pretrained encoder + LLM. `[snippet — verify]`: LibriMix about 7.3% WER vs 9.2% LLM baseline (the paper's table is also read as 8.1/7.3 for WER/B-WER — column mapping unverified); AMI-SDM 32.9% vs 34.2%; biasing degrades when the list exceeds ~50 entries. — [ISCA archive](https://www.isca-archive.org/interspeech_2025/he25_interspeech.pdf); [arXiv 2506.12059](https://arxiv.org/abs/2506.12059)
- Cocktail HuBERT (Fazel-Zarandi & Hsu, Meta, ICASSP 2023, arXiv 2303.11131): masked pseudo source separation pre-training. Libri2Mix multi-speaker ASR: LARGE 7.8% WER on test-mix vs 24.9% previous supervised best (69% relative); fine-tuned HuBERT gets 37.2% (BASE) / 35.2% (LARGE) vs C-HuBERT 13.7% / 7.8%; fine-tuning on Libri2Mix instead of LS-10h raises LibriSpeech test-clean/other WER by 1.2/4.1 points for C-HuBERT LARGE vs 5.1/9.1 for HuBERT. — [arXiv 2303.11131](https://arxiv.org/pdf/2303.11131)
- SHNU MLC-SLM 2025 system: parallel Whisper-large-v3 + mHuBERT-147 encoders into an LLM; overall CER/WER 11.76% on the blind eval set, 8.41 absolute better than the official baseline, no extra data. Task is multilingual conversational ASR, not an overlap benchmark. — [arXiv 2507.03343](https://arxiv.org/abs/2507.03343)
- Google "Cascaded encoders for fine-tuning ASR models on overlapped speech" (arXiv 2306.16398): a multi-talker encoder is cascaded onto a well-trained single-speaker encoder so the single-speaker path is minimally affected; a multi-talker speech activity detector estimates overlap because multi-talker decoding can cause errors on single-speaker utterances. Numbers: Not Reported in retrieved text. — [arXiv 2306.16398](https://arxiv.org/html/2306.16398)

### Inferences
- For Qwen3-ASR, a second encoder means a second projector (or a fusion layer), more frames or wider embeddings for the LLM, and re-learning the projector alignment. WavLM Large is ~300M parameters — about half the size of the whole 0.6B model — which conflicts with the "reasonable RTF/VRAM/CPU" constraint. WavLM is also English-only pre-training; its value for Hokkien/Mandarin mixtures is undocumented. Treat as a second-phase option to try only if encoder-only LoRA shows the native encoder cannot keep two speakers apart.
- The cheaper reading of the Cocktail-HuBERT/WavLM evidence is the training signal, not the branch: mixture-aware objectives make an encoder far better at overlap (35.2%→7.8% at equal size). That argues for unfreezing/LoRA-tuning the Qwen3-ASR encoder on mixtures rather than LoRA on the LLM only.
- The Google cascaded-encoder idea maps onto "frozen base encoder + small extra multi-talker layers enabled only when overlap is detected", which is a structural way to guarantee no single-speaker regression (see Q7).

### Gaps
- MT-LLM single- vs dual-encoder ablation numbers: not retrievable.
- No report of WavLM/C-HuBERT as an add-on branch for Mandarin/Hokkien overlap.

---

## Q4. Knowledge distillation, semi-supervised learning, and pseudo-labelling of real overlapped audio

### Takeaway
Distilling from a single-talker teacher that sees the clean sources is an established, cheap training-only trick (2019 AED, 2023 transducer TS-ASR) with zero inference cost, but no speech-LLM instance and no encoder-representation-level variant was found. Published pseudo-labelling of real overlapped meetings for multi-talker ASR is essentially absent; the documented practice remains "large simulated pre-training, then a small amount of real meeting data".

### Cited Findings
- Zhang, Chang, Qian, "Knowledge Distillation for End-to-End Monaural Multi-Talker ASR System" (Interspeech 2019): single-talker teacher → attention-based multi-talker student, objective combines single-talker soft labels and multi-talker labels; >15% relative WER reduction. — [ISCA archive](https://www.isca-archive.org/interspeech_2019/zhang19i_interspeech.html)
- Moriya et al. (NTT, arXiv 2305.15971, 2023): KD for transducer TS-ASR; teacher RNNT is run on the clean single-talker source to produce posteriors, student TS-RNNT on the mixture is trained with RNNT loss + KD loss; motivation is that simulated mixtures are built from clean signals that conventional training never uses; notes earlier multi-talker KD needed PIT and was offline only. Numbers: Not Reported in retrieved text. — [arXiv 2305.15971](https://arxiv.org/pdf/2305.15971v1.pdf)
- Real-data practice: "Large-Scale Pre-Training of End-to-End Multi-Talker ASR for Meeting Transcription with Single Distant Microphone" (Microsoft, arXiv 2103.16776) pre-trains SOT on large simulated data then fine-tunes with a small amount of real meeting data. — [arXiv 2103.16776](https://arxiv.org/pdf/2103.16776)
- "Simulating realistic speech overlaps improves multi-talker ASR" (arXiv 2210.15715): learns overlap patterns from real conversations with a statistical language model and uses it to generate training mixtures. Numbers: Not Reported in retrieved text. — [arXiv 2210.15715](https://arxiv.org/pdf/2210.15715)
- Nearest pseudo-label precedents are not multi-talker ASR: iterative pseudo-labelling for EEND diarization (37.4% relative DER reduction on CALLHOME, Interspeech 2021) and signal-level pseudo-labels for a speech-enhancement front-end in MISP-Meeting 2025 (CER 5.44% Dev / 9.52% Eval, 2nd place). — [ISCA archive (Takashima et al.)](https://www.isca-archive.org/interspeech_2021/takashima21_interspeech.html); [arXiv 2505.24446](https://arxiv.org/html/2505.24446v2)

### Inferences
- KD is directly usable for Qwen3-ASR and is free at inference: the team already owns a strong single-speaker teacher (the unmodified Qwen3-ASR). For each simulated mixture, run the frozen teacher on each clean source with teacher forcing and add a KL term between the teacher's token distribution and the student's distribution on the corresponding SOT segment (or ordinal-prompt pass). This is the Moriya-style design transplanted; it is untested for speech-LLMs.
- The same frozen teacher makes the training data cheap for Hokkien/Mandarin/English: transcripts of clean single-speaker audio (ground truth or teacher pseudo-labels) become SOT targets after mixing, so no overlapped data needs manual transcription.
- Pseudo-labelling real overlapped recordings is feasible in principle with a strong separation+ASR or diarization-conditioned cascade as the teacher, but there is no published evidence of how much it helps over simulation.

### Gaps
- No paper found on encoder-representation distillation (student encoder on mixture → per-speaker teacher encoder states) for multi-talker ASR. Explicitly searched; only output/posterior-level KD surfaced.
- No 2024–2026 paper found on semi-supervised/self-training multi-talker ASR with unlabeled real overlapped audio.

---

## Q5. Mixture-of-experts / adapter routing (MoLE) for multi-talker ASR and protecting single-speaker performance

### Takeaway
Mixture-of-LoRA-Experts inside the encoder (GLAD 2025; H-SAGE 2026, with code) is a real, recent, parameter-light line that explicitly reports single-talker and per-overlap-ratio WER, and H-SAGE adds an overlap-state auxiliary loss. Both are still SOT-based and evaluated only on English LibriSpeechMix (and CH109 for GLAD); they are not speech-LLM systems.

### Cited Findings
- GLAD (Guo et al., arXiv 2509.13093; listed in the Interspeech 2026 archive): replaces all linear layers in the encoder with MoLE blocks (MoE-LoRA-like); a gate fuses speaker-aware global context with local acoustic features to weight experts; SOT output. Abstract: significantly outperforms existing SOT-based approaches on LibriSpeechMix and CH109, with the largest gains in high-overlap conditions; low-overlap cases are already handled well by existing methods. — [arXiv 2509.13093](https://arxiv.org/html/2509.13093v1)
- GLAD numbers `[secondary — verify]`: LibriSpeechMix test 2-mix WER 8.0%, best OA-WER 8.5%. Code: no repository found. — [emergentmind topic page](https://www.emergentmind.com/topics/multi-talker-automatic-speech-recognition-asr)
- H-SAGE (Guo et al., NKU, arXiv 2607.01566, ICME 2026): Speaker-Aware Global Encoder that models silence / single-speaker / overlap states, trained with an auxiliary Overlap-Aware Loss; Holistic Gating Mechanism selects experts from global + local cues; criticises GLAD's global features as unsupervised linear projections. Metrics: WER (single-talker), PI-WER (multi-talker), OA-WER (average over overlap levels). — [arXiv 2607.01566](https://arxiv.org/html/2607.01566)
- H-SAGE numbers `[table fragment — column mapping unverified]`: full model ~3.6M (trainable) parameters with 3.8 / 5.7 / 19.5; without Overlap-Aware loss 3.7 / 5.8 / 20.1; without Holistic Gating 3.9 / 6.2 / 20.9. — [arXiv 2607.01566](https://arxiv.org/html/2607.01566)
- H-SAGE code: public. — [GitHub NKU-HLT/H-SAGE](https://github.com/NKU-HLT/H-SAGE)
- Related MoE-LoRA in speech-LLMs (not overlap): TEA-ASLP MLC-SLM 2025 system uses a multilingual MoE-LoRA structure plus CTC-predicted tokens as prompts and placed top-three. — [arXiv 2507.18051](https://arxiv.org/abs/2507.18051); "Boosting Code-Switching ASR with Mixture of Experts Enhanced Speech-Conditioned LLM" (arXiv 2409.15905, ICASSP 2025) — title only, contents not verified. — [arXiv 2409.15905](https://arxiv.org/pdf/2409.15905); domain-routed MoE of LoRAs and projectors for child/adult speech-LLM ASR. — [arXiv 2606.10454](https://arxiv.org/pdf/2606.10454)

### Inferences
- Attachability: good. MoLE blocks are LoRA-shaped and sit in the audio encoder, so they can be bolted onto the Qwen3-ASR encoder with the LLM receiving SOT targets. Added inference cost is K LoRA branches + a small gate per layer (cannot be merged into the base weights when the gate is input-dependent) — small in parameters (H-SAGE ~3.6M), modest in latency; exact RTF: Not Reported.
- A simpler variant the team can build without any paper: one "multi-talker LoRA" that is either on or off (hard routing by an overlap detector or by a task token in the prompt). When off, the model is bit-identical to the original Qwen3-ASR, which is the strongest possible single-speaker guarantee. No publication was found that evaluates this exact hard-switch design for speech-LLMs; it is an engineering inference.
- H-SAGE's ablation (if the column reading is right) suggests the overlap-aware auxiliary loss gives a small gain and the gating a larger one in high overlap; both effects are a few tenths of a WER point, so MoLE is an incremental refinement on top of SOT, not a substitute for it.

### Gaps
- No MoE/LoRA-routing paper on an LLM-decoder multi-talker system was found.
- No paper found that measures single-speaker WER regression of a LoRA-SOT-tuned speech-LLM vs its base model. Explicitly searched ("catastrophic forgetting", "single-speaker degradation", LoRA, SOT, speech LLM): nothing.

---

## Q6. Overlap-aware auxiliary tasks: overlap detection, speaker counting, activity prediction, reasoning-before-transcribing

### Takeaway
Auxiliary overlap/activity supervision appears in several 2025–2026 systems (H-SAGE overlap-state loss, STAM-ASR learned speaker activity, Google MT-SAD, CoT "count speakers first"), and one LLM paper reports that plain LLM-SOT is weak at speaker counting. No paper quantifies hallucinated-second-speaker rates or shows that a count head fixes them.

### Cited Findings
- H-SAGE trains a global encoder on silence / single-speaker / overlap states with an Overlap-Aware Loss (see Q5). — [arXiv 2607.01566](https://arxiv.org/abs/2607.01566)
- Google cascaded encoders: adds a multi-talker speech activity detector because multi-talker decoding can cause errors on single-speaker utterances. — [arXiv 2306.16398](https://arxiv.org/html/2306.16398)
- WavLM+Vicuna LLM-SOT: lower speaker-counting accuracy in complex scenarios. — [arXiv 2408.17431](https://arxiv.org/html/2408.17431)
- CoT TS-ASR: the model writes the number of speakers and each speaker's gender before transcribing; costs latency. — [arXiv 2509.15612](https://arxiv.org/abs/2509.15612)
- STAM-ASR (arXiv 2609.29805, Sept 2026): extends a pretrained AudioLLM, needs no external diarizer, learns speaker activity from intermediate model features, and keeps two fixed-size recurrent memories (per-speaker and conversation); evaluated on AMI, ICSI, LibriCSS, NOTSOFAR-1; authors state accurate speaker-activity tracking remains the main open problem. WER tables: Not Reported in retrieved text. — [arXiv 2609.29805](https://arxiv.org/abs/2609.29805)
- "Joint Target-Speaker ASR and Activity Detection" and Persona-ASR (ISSAI, 2026): WavLM-Base+ with speaker-embedding conditioning, per-language CTC heads and a voice-activity head; emits a rejection token when the enrolled speaker is absent. — [Hugging Face issai/Persona-ASR](https://huggingface.co/issai/Persona-ASR)
- Older precedent: joint source counting + separation + ASR for an unknown number of talkers (arXiv 2006.02786, 2020). — [arXiv 2006.02786](https://arxiv.org/pdf/2006.02786)
- Unified AR modelling of multi-talker ASR and speaker attributes (arXiv 2107.01549): title-level lead only; contents not verified. — [arXiv 2107.01549](https://arxiv.org/pdf/2107.01549)

### Inferences
- Two cheap ways to put a counter into Qwen3-ASR: (i) a frame-level 3-class head (silence/single/overlap) on encoder or projector output, trained with cross-entropy from the simulation labels at zero labelling cost; (ii) a leading text token in the output ("<1spk>"/"<2spk>") before the transcript — a one-token version of the CoT idea with negligible latency. Either doubles as the router for Q5/Q7. Both are standard multi-task designs but untested on this architecture.
- Training must include a large share of single-speaker samples with the single-speaker label/token; otherwise an SOT- or ordinal-prompt-trained model is biased toward inventing a second stream. This is a design rule inferred from Google's and Xiaomi's "single-speaker / target-absent" handling, not a measured result.

### Gaps
- No published measurement of second-speaker hallucination rate for LLM-based multi-talker ASR on single-speaker input.

---

## Q7. Hybrid system-level designs not in the covered list (routing, speaker-agnostic streams)

### Takeaway
Two system-level ideas are worth knowing: (a) overlap/activity detection used as a gate so only overlapped audio takes the multi-talker path (Google cascaded encoders), and (b) "speaker-agnostic activity streams", which collapses any number of speakers into exactly two non-overlapping streams and runs the recogniser twice — a 2025 result that matches this team's "all text, no speaker labels" requirement but depends on a diarization front-end.

### Cited Findings
- "Scaling Multi-Talker ASR with Speaker-Agnostic Activity Streams" (arXiv 2510.03630, Oct 2025): converts speaker-specific diarization activity into two speaker-agnostic streams so inference cost no longer grows with speaker count; naive merging degrades recognition because pretrained ASR expects contiguous single-speaker input; a "first-available" assignment makes both streams produce nearly identical transcripts; a speaker-continuity heuristic approaches the accuracy of using per-speaker activities; built on DiCoW; evaluated with tcORC-WER on AMI and ICSI; "greatly reduce runtimes … while retaining competitive performance". Exact WER/RTF: Not Reported (table fragments garbled). — [arXiv 2510.03630](https://arxiv.org/abs/2510.03630)
- Google cascaded encoders + multi-talker SAD as the overlap detector (see Q3/Q6). — [arXiv 2306.16398](https://arxiv.org/pdf/2306.16398)
- Decoupled separation → ASR (comparison only; covered elsewhere): "Elevating Robust Multi-Talker ASR by Decoupling Speaker Separation and Speech Recognition" (arXiv 2503.17886). — [arXiv 2503.17886](https://arxiv.org/pdf/2503.17886)
- Cautionary data point on generic separation front-ends: a 2026 study found SAM-Audio pre-processing increased Whisper WER on Bengali and English test sets (denoising context, not two-speaker overlap). — [arXiv 2603.04710](https://arxiv.org/pdf/2603.04710)

### Inferences
- The speaker-agnostic two-stream idea needs a diarizer (extra model, extra latency) and two conditioned decodes; it is the right shape for long meetings with >2 speakers where at most two talk at once, and overkill for the team's stated two-speaker utterance-level goal.
- A gate (overlap detector → choose base model vs multi-talker LoRA) is the lowest-risk deployment pattern for protecting single-speaker and multilingual accuracy; its failure mode is detector misses, which silently fall back to single-speaker behaviour (one speaker's text lost) rather than corrupting clean audio.

### Gaps
- No published RTF/VRAM comparison of gated-LoRA vs always-on multi-talker speech-LLMs.

---

## Q8. Discrete-token / generative joint separation-and-recognition

### Takeaway
Token-space separation with language models is active (TokenSplit 2023; TSELM 2024; CodeSep and SLM-SS 2026), but these systems generate audio codec tokens for each speaker — far more tokens than text — and none was found feeding separated tokens into an ASR LLM as an end-to-end overlapped-ASR recipe. Poor fit for this team's constraints.

### Cited Findings
- TokenSplit (Erdogan et al., Google, Interspeech 2023): Transformer encoder-decoder over discrete token sequences, multi-task: separate and transcribe each source, and generate speech from text; includes a transcript-conditioned refinement mode. — [arXiv 2308.10415](https://arxiv.org/abs/2308.10415)
- SLM-SS (arXiv 2610.27785, Oct 2026): speech language model predicts neural-codec tokens of each talker from a continuous mixture encoder; on Libri2Mix, per-talker NAR variant reaches 6.92% WER (ASR on separated output), "approaching codec-only reconstruction" `[snippet — verify]`. — [arXiv 2610.27785](https://arxiv.org/html/2610.27785v1)
- CodeSep (arXiv 2601.12757, Jan 2026): codec-driven separation + compression (base-token disentanglement, auxiliary-token serial prediction); not recognition. — [arXiv 2601.12757](https://arxiv.org/html/2601.12757)
- TSELM (arXiv 2409.07841): target speaker extraction with discrete tokens and language models; needs enrollment. — [arXiv 2409.07841](https://arxiv.org/pdf/2409.07841)

### Inferences
- A codec stream is typically tens to hundreds of tokens per second per speaker, versus a few text tokens per second; generating audio tokens for two speakers and then recognising them costs one to two orders of magnitude more decoder steps than SOT text. Qwen3-ASR has no audio-token output head, so this would be a new model, not a fine-tune. Not recommended; note it only as a research direction. (Token-rate statement is the researcher's general knowledge, not from the cited papers.)

### Gaps
- No paper found that feeds discrete SSL/codec tokens of a mixture into an LLM for direct multi-talker text output and compares with continuous-feature speech-LLMs.

---

## Q9. General audio-LLMs on overlapped speech; other 2025–2026 speech-LLM multi-talker work; challenges

### Takeaway
No published prompt-only evaluation of Gemini / GPT-4o-transcribe / Qwen-Omni / SALMONN on fully overlapped two-speaker speech (e.g. Libri2Mix) was found; the closest 2026 benchmark covers long-form multi-speaker ASR and says speaker handling is still an open problem. The 2025–2026 speech-LLM multi-talker literature outside the covered list is dominated by diarization-conditioned designs, which solve "who spoke when" (not needed here) rather than dense overlap.

### Cited Findings
- Omnilingual-MsASR (arXiv 2610.29455, Oct 2026): benchmark of omni LLMs on multilingual long-form multi-speaker ASR; Gemini-family models show low cpWER (e.g. 0.23–0.44) but tcpWER frequently above 1.0; speaker confusion rises from 9.1% to 25.6% from two to more than seven speakers; concludes robust speaker identification remains open. Whether the audio contains overlap was not confirmed. — [arXiv 2610.29455](https://arxiv.org/html/2610.29455)
- Qwen-Omni reports include multi-speaker ASR test sets (AliMeeting Test, AISHELL-4, MagicData-RAMC) per the Qwen3.8-Omni report; per-model overlap numbers: Not Reported in retrieved text. — [arXiv 2609.25611](https://arxiv.org/pdf/2609.25611)
- Diarization-aware LLM ASR: "Diarization-Aware Multi-Speaker ASR with LLMs" (arXiv 2506.05796) and DM-ASR (arXiv 2604.22467, Apr 2026) feed frame-level speaker and semantic embeddings to the LLM to produce segment-level transcripts for high-overlap meetings. Numbers: Not Reported. — [arXiv 2506.05796](https://arxiv.org/pdf/2506.05796); [arXiv 2604.22467](https://arxiv.org/pdf/2604.22467)
- Title-level leads, contents not verified: TagSpeech (arXiv 2601.06896, end-to-end multi-speaker ASR + diarization with temporal grounding); "Grounding Spoken LLMs in Multi-Speaker Audio via Diarization Conditioning" (arXiv 2606.18134); "Train Short, Infer Long" (arXiv 2511.16046, speech-LLM streamable joint ASR + diarization); SpeakerLM (arXiv 2508.06372). — [arXiv 2601.06896](https://arxiv.org/pdf/2601.06896); [arXiv 2606.18134](https://arxiv.org/pdf/2606.18134); [arXiv 2508.06372](https://arxiv.org/html/2508.06372v3)
- MLC-SLM 2025 (Interspeech 2025 challenge): 1,500 h of real multilingual conversational speech in 11 languages; typical systems are Whisper-large-v3 encoder + adaptor + LoRA LLM (Transsion), dual encoder (SHNU), MoE-LoRA + CTC-token prompts (TEA-ASLP). The 2026 edition added languages/dialects and its Task-2 baseline fine-tunes Qwen2.5-Omni-7B. These are conversational-speech systems; none of the retrieved descriptions reports an overlap-specific method or overlap-only score. — [arXiv 2508.14916](https://arxiv.org/html/2508.14916); [arXiv 2507.18051](https://arxiv.org/abs/2507.18051); [arXiv 2609.11724](https://arxiv.org/pdf/2609.11724)
- Survey: He & Whitehill, "Survey of End-to-End Multi-Speaker ASR for Monaural Audio" (arXiv 2505.10975; Computer Speech & Language) — taxonomy of SIMO vs SISO, long-form, benchmarks, open challenges (data scarcity, overlap). — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0885230825001500)

### Inferences
- Because no vendor or academic paper documents prompt-only overlap transcription for general audio-LLMs, the team should not expect Qwen3-ASR (or Qwen-Omni) to handle full overlap from prompting alone; a quick zero-shot test on their own mixtures is the only way to establish the baseline.
- Diarization-conditioned speech-LLMs add a diarizer and per-speaker conditioning that this team's requirements explicitly exclude; they are relevant only if requirements grow to meetings with speaker labels.

### Gaps
- CHiME-8/9 NOTSOFAR speech-LLM system descriptions: search returned nothing; not covered.
- No source found on Qwen3-ASR itself being fine-tuned for overlapped speech.

---

## Q10. Cross-lingual, code-switching, Mandarin and Taiwanese-Hokkien overlapped speech

### Takeaway
Evidence is thin: a few cross-lingual mixture results exist (MT-LLM En-De-Mix, zero-shot Mandarin AishellMix with Whisper-Sidecar, Kazakh–English Persona-ASR, AliMeeting in Xiaomi's report), but no published work was found on overlapped ASR with code-switching inside a speaker's turn, or on Taiwanese Hokkien / other Chinese-dialect mixtures.

### Cited Findings
- MT-LLM evaluates an En-De-Mix set and supports "transcribe the speaker speaking language X". — [arXiv 2409.08596v2](https://arxiv.org/html/2409.08596v2)
- Whisper-Sidecar joint model: acceptable zero-shot multi-talker ASR on Mandarin AishellMix after English-only training. — [arXiv 2407.09817](https://arxiv.org/pdf/2407.09817)
- Persona-ASR (ISSAI, MDPI MAKE 2026): first target-speaker ASR for Kazakh; bilingual Kazakh/English with separate CTC heads; released KazMix3 and PersonaMix (same- and cross-language enrollment, 1–3 interferers, SNR −3 to +3 dB); abstract reports gains over a cascade baseline of 13.3 WER points (English) and 24.6 (Kazakh), with Kazakh ~14 points worse than English; model card lists WER around 29 (Libri3Mix-100h) and around 43 (Kazakh3Mix-100h) `[partial table — verify]`. Requires enrollment. — [DOI 10.3390/make8080246](https://doi.org/10.3390/make8080246); [Hugging Face](https://huggingface.co/issai/Persona-ASR)
- Xiaomi-CocktailASR-1 reports AliMeeting-Far (Mandarin meetings) TS-WER 20.63% `[snippet — verify]`. — [arXiv 2609.11274](https://arxiv.org/html/2609.11274v1)
- "Adapting Multi-Lingual ASR Models for Handling Multiple Talkers" (arXiv 2305.18747): title-level lead on multilingual multi-talker adaptation; contents not verified this session. — [arXiv 2305.18747](https://arxiv.org/pdf/2305.18747)
- Mandarin real-meeting corpora available for evaluation: AISHELL-4 (real-recorded Mandarin conference corpus); AliMeeting is used in Qwen-Omni and Xiaomi evaluations. — [emergentmind AISHELL-4 page](https://www.emergentmind.com/topics/aishell-4-dataset); [arXiv 2609.25611](https://arxiv.org/pdf/2609.25611)

### Inferences
- The lower-resource language is consistently the weak side of a bilingual mixture model (Persona-ASR: Kazakh 14 points behind English). Expect Hokkien-in-mixture to lag Mandarin/English and plan a per-language overlap test matrix (Hokkien+Hokkien, Hokkien+Mandarin, Mandarin+English, etc.).
- Cross-language mixtures may be easier than same-language ones for attribute prompts ("transcribe the Hokkien speaker"), which MT-LLM's language attribute supports in principle; unproven for closely related Sinitic languages.
- Since no Hokkien overlap corpus surfaced, simulation from the team's single-speaker Hokkien data is the only data route.

### Gaps
- No published overlapped-speech work on Taiwanese Hokkien, Cantonese, or other Chinese dialects found. No overlapped + intra-utterance code-switching work found.

---

## Q11. Ranking by promise for this team (researcher's assessment)

### Takeaway
The alternatives worth prototyping are all training-recipe or light-adapter changes layered on a serialized-text output; structurally heavy options (second encoder, token-space separation, diarization conditioning) have weaker fit to a 0.6B model with a simple-inference requirement.

### Cited Findings
- Evidence each rank rests on: MT-LLM instruction/attribute decoding — [arXiv 2409.08596](https://arxiv.org/pdf/2409.08596); H-SAGE / GLAD MoLE + overlap loss — [arXiv 2607.01566](https://arxiv.org/abs/2607.01566), [arXiv 2509.13093](https://arxiv.org/abs/2509.13093); single-talker-teacher KD — [arXiv 2305.15971](https://arxiv.org/pdf/2305.15971v1.pdf), [ISCA 2019](https://www.isca-archive.org/interspeech_2019/zhang19i_interspeech.html); gated multi-talker path — [arXiv 2306.16398](https://arxiv.org/html/2306.16398); mixture-aware SSL — [arXiv 2303.11131](https://arxiv.org/pdf/2303.11131); speaker-agnostic streams — [arXiv 2510.03630](https://arxiv.org/abs/2510.03630); conditional chain — [arXiv 2106.08595](https://arxiv.org/pdf/2106.08595.pdf); token-space separation — [arXiv 2610.27785](https://arxiv.org/html/2610.27785v1).

### Inferences (ranking, most to least promising)

| Rank | Approach | New modules | Added inference cost | Evidence strength | Why this rank |
|---|---|---|---|---|---|
| 1 | Instruction/attribute multi-task prompts (MT-LLM style): train "all speakers" + "first/second speaker" (+ language) prompts in one LoRA | None | ~0 for "all speakers"; one extra short prompt per speaker for ordinal passes (encoder once) | Medium: one speech-LLM paper, 2-mix competitive with SOT-Conformer (number unverified) | Pure data+prompt change; gives a fallback decoding mode and an A/B against SOT on the same weights |
| 2 | Switchable multi-talker LoRA + count token / overlap head (hard routing; H-SAGE/Google-inspired) | Tiny head or 1 output token | ~0 | Low-medium: components published separately; the combination on a speech-LLM is not | Only design that can guarantee unchanged single-speaker/multilingual behaviour; also the hallucination guard |
| 3 | Single-talker-teacher KD on simulated mixtures (teacher = original Qwen3-ASR on clean sources) | None (training only) | 0 | Medium for AED/RNNT (>15% rel. in 2019); none for speech-LLMs | Free at inference; also solves Hokkien label scarcity via teacher pseudo-labels |
| 4 | Encoder-side MoLE (GLAD / H-SAGE) under SOT | MoLE blocks + gate (~3.6M params in H-SAGE) | Small, unmeasured | Medium: two papers, code for H-SAGE, English simulated only | Incremental gain in high overlap; try after 1–3 work |
| 5 | Second mixture-aware encoder branch (WavLM / C-HuBERT / mHuBERT) | Encoder + projector/fusion | Large relative to a 0.6B model | Medium for the SSL objective; ablation for the LLM branch not retrieved | Costly, English-centric SSL; consider only if the native encoder proves the bottleneck |
| 6 | Speaker-agnostic two-stream decoding on a diarizer | Diarizer + conditioning | Diarizer + 2 decodes | Medium (AMI/ICSI), numbers not retrieved | Fits long >2-speaker meetings; exceeds current requirement |
| 7 | Conditional chain / two-call residual decoding | None or new feature module | ≥ SOT | Low: 2021 CTC results only (LibriMix 24.9%) | SOT already provides text-level chaining |
| 8 | Enrollment-based LLM TS-ASR (Xiaomi-CocktailASR-1, CoT+RL) | Reference-audio prompt | Extra audio prefix; CoT adds latency | Strong results but different task | Violates "no enrollment"; useful as proof an LLM decoder can focus on one voice |
| 9 | Token-space generative separation → ASR | New output modality | Very high | Separation-quality evidence only | Not a fine-tune of Qwen3-ASR |

- Ranks 1–3 are complementary and can share one training set of simulated mixtures; rank 2's count token is also what tells rank 1 how many ordinal passes to run.
- All rankings assume the covered mainstream option (SOT/SOP on the LLM) remains the baseline to beat; none of the alternatives above has published evidence of beating a well-tuned LLM-SOT system by a wide margin on 2-speaker mixtures.

### Gaps
- Every ranked item lacks direct evidence on (a) a sub-1B LLM decoder, (b) Mandarin/Hokkien mixtures, and (c) measured RTF/VRAM. The ranking is therefore a prior for experiments, not a conclusion.
- Numbers tagged `[snippet/secondary — verify]` must be checked against the PDFs once arxiv.org is reachable (it was blocked in this session's network settings).
