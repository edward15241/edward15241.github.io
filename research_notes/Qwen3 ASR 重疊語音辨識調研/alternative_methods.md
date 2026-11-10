# Alternative approaches to single-channel overlapped speech recognition (beyond SOT / t-SOT / SOP / per-speaker CTC / Sidecar / PIT / Parakeet-multitalker / DiCoW / CSS cascade / MOSS-TD)

> Research date: 2026-11-10. Scope: methods NOT in the "already covered" list, 2021–2026, assessed for attachment to Qwen3-ASR-0.6B (audio encoder → projector → Qwen3 LLM decoder).
>
> **Verification status.** The first pass of these notes relied on search summaries because arxiv.org was blocked. After the network was opened, the arXiv HTML full text of 16 papers was downloaded and read directly; numbers from those are tagged `[paper table]` or `[paper text]`. Papers read in full text: 2409.08596v2, 2407.09817v2, 2312.08079, 2408.17431, 2506.12059, 2509.13093v1, 2607.01566, 2510.03630, 2609.11274v1, 2509.15612v2, 2609.30476, 2609.29805, 2305.15971, 2306.16398, 2610.27785v1, 2610.29455. Items still known only from search summaries are tagged `[search summary — not checked in paper]`. Missing data is "Not Reported"; nothing is estimated.

---

## Q1. Prompt-/instruction-based decoding with an LLM decoder (attribute-conditioned transcription without enrollment; encoder once, decoder N times)

### Takeaway
MT-LLM (CUHK, 2024) is direct evidence that one LoRA-tuned speech-LLM can do "transcribe everyone" and "transcribe the speaker selected by sex / order / language / keyword" from text instructions, and that training these tasks together improves the plain all-speaker task. But the "transcribe the N-th speaker" prompt is its weakest attribute (9.0% WER vs a 5.7% best-matching reference on 2-speaker mixtures), so ordinal multi-pass decoding is better treated as an auxiliary training task than as the primary inference mode.

### Cited Findings
**MT-LLM — "Large Language Model Can Transcribe Speech in Multi-Talker Scenarios with Versatile Instructions" (Meng, Hu, et al., CUHK; arXiv 2409.08596)**
- Six tasks trained jointly in one model: Multi-Talker (MT) ASR; Target-Talker (TT, 3-second reference clip + 3 s silence concatenated before the mixture); Keyword-Tracing (KT, transcribe the talker who said a given word); Sex-Specific (SS); Order-Specific (OS, "transcribe a talker based on their appearance order"); Target-Lingual (TL, English or German talkers). `[paper text]` — [arXiv 2409.08596v2](https://arxiv.org/html/2409.08596v2)
- Architecture `[paper text]`: frozen Whisper-large-v2 encoder (last layer) + frozen WavLM-base (learnable weighted sum of layers); each has an adapter of two conv layers + bottleneck adapter + linear, output stride 80 ms, dimension 2,048; frozen Llama-2-7b-chat with LoRA rank 32 on attention q/k/v/o. Total 7.55B parameters, 76.6M (1%) trainable. — [arXiv 2409.08596v2](https://arxiv.org/html/2409.08596v2)
- Data and compute `[paper text]`: simulated 2- and 3-talker mixtures from LibriSpeech 960 h with random start offsets, plus the 180 h German subset of CoVoST 2; with the original single-talker data the total is about 6.3K hours, about 10% containing German. 32 A100-40G GPUs, 150K updates, all tasks mixed. Loss: Not Reported explicitly (LLM predicts target transcripts). — [arXiv 2409.08596v2](https://arxiv.org/html/2409.08596v2)
- Table I, WER % `[paper table]`: MT-LLM LibriSpeech 1-spk 2.3; LibriSpeechMix 2-spk 5.2, 3-spk 10.2; German 1-spk 11.7; En-De-Mix 2-spk 21.0, 3-spk 22.8. SOT-Conformer: 3.6 / 4.9 / 6.2. D2V-Sidecar-DB: 7.5 / 11.9 (2/3-spk). SALMONN-7B with its official prompt: 2.4 / 32.9 / 45.9. — [arXiv 2409.08596v2](https://arxiv.org/html/2409.08596v2)
- Table II, 2-speaker, MT-LLM vs "best-matching" (the correct talker's transcript picked from the MT output, i.e. no speaker-selection error) `[paper table]`: TT 6.7 vs 5.5; KT 5.0 vs 4.8; SS 5.5 vs 4.9; **OS 9.0 vs 5.7**; TL 21.8 vs 21.0. Table III, 3-speaker: TT 16.2 vs 12.0; KT 12.6 vs 9.7; SS 15.0 vs 12.4; OS 15.4 vs 11.7; TL 24.1 vs 24.0. — [arXiv 2409.08596v2](https://arxiv.org/html/2409.08596v2)
- Authors on order prompts `[paper text]`: "MT-LLM slightly falls short in determining the order in which the talker appears"; they expect gains from positional embedding on the speech part, larger adapters, or fully fine-tuning the speech encoder. — [arXiv 2409.08596v2](https://arxiv.org/html/2409.08596v2)
- Table IV ablation, LibriSpeech 1-spk / LSM 2-spk / 3-spk `[paper table]`: Whisper only, MT task only 4.3 / 6.6 / 16.1; + WavLM 2.4 / 5.5 / 10.7; + multi-task training 2.3 / 5.2 / 10.2. — [arXiv 2409.08596v2](https://arxiv.org/html/2409.08596v2)
- Single-speaker `[paper text]`: SOT-Conformer "falls short in single-talker scenarios" (3.6) while MT-LLM stays at 2.3. Limitations stated: simulated data only; not a universal speech LLM. — [arXiv 2409.08596v2](https://arxiv.org/html/2409.08596v2)
- Code: repository at cuhealthybrains/MT-LLM; whether it holds training code/weights or only samples was not verified. — [GitHub](https://github.com/cuhealthybrains/MT-LLM)

**Small-LLM evidence: "Thinking in cocktail party: CoT and RL for target speaker ASR" (Zhang, Su, et al.; arXiv 2509.15612)** — needs enrollment, but the model size is close to Qwen3-ASR-0.6B.
- Model `[paper text]`: Qwen2.5-0.5B-Instruct LLM + pre-trained Data2Vec2 audio encoder (0.3B parameters). CoT content: number of speakers, each speaker's gender, discrete similarity level to the reference, start/end times; then SFT → CoT fine-tuning → RL with format + WER reward. — [arXiv 2509.15612v2](https://arxiv.org/html/2509.15612v2)
- Table 1, TS-WER % on Libri2Mix / Libri3Mix test-clean `[paper table]`: base 7.4 / 17.24; + CoT 5.29 / 13.57; + CoT + RL 4.84 / 12.23; MT-LLM 6.7 / 16.2. The "21% / 16.2% relative" figures are CoT vs MT-LLM. — [arXiv 2509.15612v2](https://arxiv.org/html/2509.15612v2)
- Single-speaker LibriSpeech test-clean with the TS prompt `[paper table]`: base 8.1; + CoT 3.88; + CoT + RL 3.65. — [arXiv 2509.15612v2](https://arxiv.org/html/2509.15612v2)
- CoT ablation `[paper table]`: removing speaker number 5.46 / 14.01; gender 5.32 / 13.7; similarity 6.08 / 14.93; times 5.19 / 13.54 (vs full 5.29 / 13.57). — [arXiv 2509.15612v2](https://arxiv.org/html/2509.15612v2)

**Xiaomi-CocktailASR-1 (arXiv 2609.11274)** — enrollment-based; relevant for its baselines and single-speaker protection.
- Model `[paper text]`: 0.6B audio encoder derived from Data2Vec2 (1280-dim frames) + Qwen3-8B LLM; reference audio concatenated as a "voiceprint prompt". Training data: about 400,000 h multi-speaker (AMI, AliMeeting, LibriMix, simulated proprietary mixtures, in-house conversations), about 600,000 h single-speaker TS-ASR data "to prevent performance degradation in single-speaker scenarios", about 10,000 h negative samples. — [arXiv 2609.11274v1](https://arxiv.org/html/2609.11274v1)
- Table 1, WER % on LibriMix 2mix / 3mix / LibriSpeechMix 2mix / 3mix `[paper table]`: Xiaomi 4.11 / 12.29 / 2.90 / 4.91; **Qwen3-ASR-1.7b 68.75 / 106.04 / 92.17 / 160.82**; StepAudio2 71.23 / 121.08 / 92.72 / 164.66; Gemini-2.5-pro (TS prompt with reference speech) 48.41 / 76.10 / 30.69 / 52.34. Paper text: "Qwen3-ASR-1.7b and StepAudio2 lack the ability to recognize multiple speakers". These are target-speaker scores; the single-speaker models were given no way to pick the target. — [arXiv 2609.11274v1](https://arxiv.org/html/2609.11274v1)
- Table 2, real data, AMI-SDM / AliMeeting-Far `[paper table]`: Xiaomi 21.81 / 20.63; Qwen3-ASR-1.7b 38.18 / 39.64; Gemini-2.5-pro 52.95 / 56.75; Whisper large-v2 36.40 / –. — [arXiv 2609.11274v1](https://arxiv.org/html/2609.11274v1)
- Table 3, single-speaker non-empty WER `[paper table]`: LibriSpeech 1.73 (Qwen3-ASR-1.7b 1.87); AliMeeting-near 6.57 (6.39); AMI-IHM 8.89 (10.56); WenetSpeech-meeting 5.81 (5.84); CommonVoice-zh 4.95 (5.39). False-rejection rate 0–0.73%. — [arXiv 2609.11274v1](https://arxiv.org/html/2609.11274v1)
- CoT mode `[paper table]`: LibriMix 2mix 4.11 → 3.87; other sets change by ≤0.1. — [arXiv 2609.11274v1](https://arxiv.org/html/2609.11274v1)
- Release: no repository or weights statement found in the paper text; a news site says it was open-sourced (secondary, unverified). — [ProPakistani](https://propakistani.pk/2026/09/14/xiaomis-new-open-source-ai-can-isolate-one-speaker-from-overlapping-voices/)

**Whisper-based comparison points**
- "Empowering Whisper as a Joint Multi-Talker and Target-Talker System" (arXiv 2407.09817): frozen Whisper + Sidecar separator + Target Talker Identifier + decoder soft prompts; trainable parameters 8.69M (Whisper-small) to 18.41M (Whisper-large, 1.18%). Multi-talker WER, Whisper-large-SS-TTI `[paper table]`: LibriMix 2spk 4.66, 3spk 16.79; LibriSpeechMix 2spk 3.43, 3spk 6.80. (The aggregator figures 6.99 / 11.4 quoted in the first pass do not appear in the paper's table and are withdrawn.) Code: github.com/LingweiMeng/Whisper-Sidecar. — [arXiv 2407.09817v2](https://arxiv.org/html/2407.09817v2)
- "Extending Whisper with prompt tuning to target-speaker ASR" (arXiv 2312.08079): speaker embedding + soft prompts on frozen Whisper. Libri2Mix test-clean / test-both TS-WER `[paper table]`: Whisper-medium + deep prompting 13.89 / 29.81 (1.31M params); Whisper-medium + LoRA 11.98 / 26.38 (1.85M); Whisper-large + LoRA 13.07 / 26.51; full fine-tuning Whisper-medium 18.79 / 34.04. Unmodified Whisper large/medium/small on test-clean: 78.92 / 85.42 / 92.24. — [arXiv 2312.08079](https://arxiv.org/html/2312.08079)
- "Asymmetric Classifier-Free Guidance for TS-ASR" (Guan & Whitehill, arXiv 2609.30476): one Whisper-small trained so the enrollment-conditioned branch gives the target transcript and the unconditioned branch gives serialized multi-speaker text. Libri2Mix-clean `[paper table]`: TS-WER 17.59 ± 1.96 (condition-only 18.04 ± 1.85); multi-speaker cpWER of the shared model 23.51 ± 4.97 vs **14.10 for SOT-only fine-tuning**; pretrained Whisper-small 69.26. — [arXiv 2609.30476](https://arxiv.org/html/2609.30476)
- Searched for TS-ASR prompted by onset time, "the louder speaker" or "the speaker who starts later": nothing found other than MT-LLM's Order-Specific task. — [arXiv 2409.08596v2](https://arxiv.org/html/2409.08596v2)

### Inferences (researcher's assessment)
- Attachability to Qwen3-ASR: high for the multi-task recipe. It is data + prompts on the existing encoder → projector → LLM; MT-LLM's 76.6M trainable parameters are adapters and LoRA only.
- The ordinal prompt should not be the main decode path. In MT-LLM it loses 3.3 WER points to speaker-selection error on 2-speaker mixtures (9.0 vs 5.7), while the single-pass MT output is 5.2. Sex and keyword prompts are closer to their reference but cannot enumerate all speakers in general (same-sex pairs).
- What the evidence does support: add ordinal/attribute prompts as auxiliary tasks next to the all-speaker task. MT-LLM's ablation shows multi-task training improved the all-speaker task itself (5.5 → 5.2; 10.7 → 10.2).
- Cost if ordinal passes are used anyway: the encoder runs once, the audio prefix can be cached, and total generated text is about the same as SOT, so RTF should be near single-pass SOT. This is reasoning, not a measured result; the passes are independent and can duplicate or drop a speaker.
- The CoT-RL model (0.5B LLM + 0.3B encoder) reaching 4.84% TS-WER on Libri2Mix shows a sub-1B speech-LLM has enough capacity to pull one voice out of a two-speaker mixture. Its base model had poor single-speaker WER (8.1) until further training fixed it (3.65), so single-speaker data must stay in the mix.
- The asymmetric-CFG result warns that sharing one model between a conditioned target-speaker mode and an unconditioned all-speaker mode cost a lot on the all-speaker side at Whisper-small scale (23.5 vs 14.1). Mode sharing is not automatically free.
- Qwen3-ASR-1.7b's 38–40% on AMI-SDM / AliMeeting-Far and 69–161% on synthetic mixtures (target-speaker scoring) is the nearest published zero-shot data point for the Qwen3-ASR family; no number exists for the 0.6B model.

### Gaps
- No published head-to-head of "N ordinal-prompt passes" vs "one SOT pass" on RTF or duplicate/omission rate.
- No attribute-prompt results for Mandarin or Hokkien mixtures; MT-LLM's cross-lingual set is English–German and its TL score (21.8) is limited by noisy German data per the authors.

---

## Q2. Iterative / recursive / conditional-chain decoding and text-conditioned residual decoding

### Takeaway
Conditional-chain multi-speaker ASR is published with code, but only for 2021-era CTC models with weak absolute WERs; no paper applies "decode speaker 1, then condition on it to decode speaker 2" as separate passes of a speech-LLM. In an LLM decoder, single-pass SOT already is a conditional chain in text space.

### Cited Findings
- "Multi-Speaker ASR Combining Non-Autoregressive Conformer CTC and Conditional Speaker Chain" (Guo, Chang, Watanabe, Xie; Interspeech 2021, arXiv 2106.08595): each speaker's transcript is produced in turn from the mixture plus previously estimated speakers' features; each step is a non-autoregressive CTC pass; WER 22.3% on WSJ0-Mix and 24.9% on LibriMix; 19.9% / 34.3% on WSJ0-2mix / 3mix with variable speaker counts; about 1/7 the latency of PIT-Conformer AR; code in the authors' ESPnet fork (`conditional-multispk`). `[search summary of abstract — not checked in paper]` — [ISCA archive](https://www.isca-archive.org/interspeech_2021/guo21_interspeech.html); [arXiv 2106.08595](https://arxiv.org/pdf/2106.08595.pdf)
- Independent confirmation of the LibriMix figure `[paper table]`: CMT-LLM's comparison table lists "Conditional-Conformer-CTC, 2021" at 24.5 dev / 24.9 test, against 9.2 for an LLM baseline. — [arXiv 2506.12059](https://arxiv.org/html/2506.12059)
- TokenSplit (Google, Interspeech 2023, arXiv 2308.10415): Transformer seq2seq over discrete tokens, trained to separate and transcribe each source and to refine separation given transcripts. A secondary summary reports transcript conditioning improving differential WER 26.6% → 12.1% on 3-second Libri2Mix segments `[search summary — not checked in paper]`. — [ISCA archive](https://www.isca-archive.org/interspeech_2023/erdogan23_interspeech.html)

### Inferences
- A two-call variant for an LLM (pass 2 prompt = "speaker 1 said …; transcribe the other speaker") adds no information over SOT and costs slightly more. Its only practical uses are an early exit when one speaker is detected and emitting speaker 1's text before speaker 2 is decoded.
- Feature-level chains (removing speaker-1 evidence in encoder space before pass 2) need a new module and have no speech-LLM evidence: research risk, not an engineering option.

### Gaps
- No 2023–2026 paper found on recursive or residual decoding for speech-LLMs (searched "conditional chain", "speaker-conditional chain", "recursive", "text-conditioned" multi-speaker ASR).

---

## Q3. Multi-/dual-encoder fusion, mixture-aware SSL encoders, and whether the encoder must be adapted

### Takeaway
Two verified results matter for Qwen3-ASR: (1) with an encoder that has not been adapted to mixtures, training only projector + LoRA fails badly (46.7% vs 10.2% WER on LibriMix when the encoder is unfrozen); (2) adding a small mixture-aware SSL branch (WavLM-base) to a Whisper encoder cut MT-LLM's 3-speaker WER from 16.1 to 10.7. The encoder side, not the LLM, is where overlap ability is won.

### Cited Findings
- "Advancing Multi-talker ASR Performance with Large Language Models" (arXiv 2408.17431): WavLM encoder + projector + Vicuna-7B with LoRA (rank 16, alpha 16), SOT targets, greedy decoding; multi-stage training projector → encoder → LoRA. `[paper text]` — [arXiv 2408.17431](https://arxiv.org/html/2408.17431)
- LibriMix test WER % `[paper table]`: AED with WavLM Large 9.2; LLM with WavLM Base+ 15.9; LLM with WavLM Large 10.2; LLM with WavLM Large first fine-tuned on LibriMix 9.0. Without LoRA: 11.3 (vs 10.2) and 9.5 (vs 9.0). — [arXiv 2408.17431](https://arxiv.org/html/2408.17431)
- Encoder freezing `[paper table]`: original WavLM Large trained jointly 10.2 test; **same encoder frozen 46.7**; LibriMix-fine-tuned encoder, frozen 9.0, unfrozen 10.1. Text: "for an unadapted encoder, merely training the projector to perform alignment is insufficient". Multi-stage beats single-stage by 0.2–0.3. — [arXiv 2408.17431](https://arxiv.org/html/2408.17431)
- AMI-SDM (pre-trained on 0.83k h LibriMix, fine-tuned on AMI) `[paper table]`: WavLM Large AED WER 30.5 / cpWER 24.1; WavLM Large LLM 27.6 / 21.0 (greedy), 26.8 / 20.4 (beam 4). A Conformer AED with 900k h supervised pre-training has cpWER 21.2. — [arXiv 2408.17431](https://arxiv.org/html/2408.17431)
- MT-LLM dual-encoder ablation `[paper table]`: Whisper-large-v2 only 4.3 / 6.6 / 16.1 (1-spk / 2-spk / 3-spk); + WavLM-base 2.4 / 5.5 / 10.7. Both encoders were frozen; only adapters and LoRA trained. — [arXiv 2409.08596v2](https://arxiv.org/html/2409.08596v2)
- Size reference `[paper table]`: a WavLM Base+ TS-ASR system is listed at 95.34M parameters. — [arXiv 2312.08079](https://arxiv.org/html/2312.08079)
- CMT-LLM (He et al., Interspeech 2025, arXiv 2506.12059): multi-talker LLM ASR plus rare-word biasing in the prompt. LibriMix test `[paper table]`: LLM baseline 9.2 WER / 25.3 B-WER; CMT-LLM without biasing 9.3 / 28.7; with a biasing list of +100 distractors 7.3 / 8.7, +1,000 7.9 / 12.5, +5,000 8.4 / 15.2. AMI-SDM test 32.9 vs 34.2 baseline (+1,000 distractors). The paper's Table 1 labels 7.3 as "with 1,000 distractors" while its Table 3 gives 7.3 for +100 and 7.9 for +1,000 — internal inconsistency. — [arXiv 2506.12059](https://arxiv.org/html/2506.12059)
- Cocktail HuBERT (Meta, ICASSP 2023, arXiv 2303.11131): masked pseudo source separation pre-training; Libri2Mix test WER 7.8% (LARGE) vs 35.2% for fine-tuned HuBERT LARGE; BASE 13.7% vs 37.2% `[search summary of abstract/tables — not checked in paper]`. The 7.80 figure is confirmed as cited in another paper's table `[paper table]`. — [arXiv 2303.11131](https://arxiv.org/pdf/2303.11131); [arXiv 2407.09817v2](https://arxiv.org/html/2407.09817v2)
- SHNU MLC-SLM 2025 system: parallel Whisper-large-v3 + mHuBERT-147 into an LLM; 11.76% CER/WER on the blind set, 8.41 absolute better than baseline; conversational task, not an overlap benchmark `[search summary — not checked in paper]`. — [arXiv 2507.03343](https://arxiv.org/abs/2507.03343)

### Inferences
- LoRA on the LLM alone is the wrong place to spend the budget: LoRA is worth about 0.5–1.1 WER points in 2408.17431, whereas encoder adaptation is worth 36 points. For Qwen3-ASR this argues for encoder LoRA or partial unfreezing as mandatory, with LLM LoRA mainly teaching the output format.
- Qwen3-ASR's encoder was trained on single-speaker ASR data, like Whisper's. MT-LLM's Whisper-only row (frozen encoder, 16.1 on 3-spk, and 4.3 on clean single-speaker) is the closest analogue of "freeze the Qwen3-ASR encoder and tune the rest".
- A WavLM-base-sized branch is roughly 95M parameters, a real but moderate addition to a 0.6B model; it runs on every utterance. WavLM pre-training is English; its benefit for Hokkien/Mandarin mixtures is undocumented. MT-LLM did not test "Whisper encoder unfrozen, no WavLM", so it is unknown whether encoder adaptation alone would close the same gap.
- A two-step schedule is supported by 2408.17431: first adapt the encoder to mixtures, then freeze it and align projector + LoRA (9.0 was the best row).

### Gaps
- No ablation that separates "second encoder" from "adapt the first encoder" on the same system.
- No report of WavLM / Cocktail HuBERT as an add-on branch for Mandarin or Hokkien overlap.

---

## Q4. Knowledge distillation, semi-supervised learning, and pseudo-labelling of real overlapped audio

### Takeaway
Distilling from a single-talker teacher that sees the clean source is a training-only trick with zero inference cost, but the verified gain is modest (about 5% relative offline, 9% streaming in the 2023 transducer study) and there is no speech-LLM instance. Published pseudo-labelling of real overlapped meetings for multi-talker ASR was not found.

### Cited Findings
- Moriya et al. (NTT, arXiv 2305.15971): target-speaker RNNT student on the mixture, single-talker RNNT teacher on the clean target; loss = RNNT loss + λ·KD loss. Data: simulated 2-speaker + noise mixtures from the Corpus of Spontaneous Japanese with CHiME-3 noise, SIR −5 to 5 dB, SNR 0–20 dB. Average CER % `[paper table]`: offline TS-RNNT 15.8 → 15.0 with KD (λ = 0.1); streaming 21.2 → 19.2 (λ = 0.01); cascade TSE + RNNT 16.5 offline / 30.2 streaming. λ = 1.0 was worse than no KD offline (16.3). — [arXiv 2305.15971](https://arxiv.org/html/2305.15971)
- Zhang, Chang, Qian, "Knowledge Distillation for End-to-End Monaural Multi-Talker ASR System" (Interspeech 2019): single-talker teacher → attention-based multi-talker student; more than 15% relative WER reduction `[search summary — not checked in paper]`. — [ISCA archive](https://www.isca-archive.org/interspeech_2019/zhang19i_interspeech.html)
- Real-data practice: Microsoft pre-trains SOT on large simulated data then fine-tunes on a small amount of real meetings (arXiv 2103.16776); "Simulating realistic speech overlaps improves multi-talker ASR" (arXiv 2210.15715) learns overlap patterns from real conversations to drive simulation `[search summaries — not checked in paper]`. — [arXiv 2103.16776](https://arxiv.org/pdf/2103.16776); [arXiv 2210.15715](https://arxiv.org/pdf/2210.15715)
- Same pattern in an LLM system `[paper table]`: 0.83k h LibriMix pre-training + AMI fine-tuning gives AMI-SDM cpWER 21.0, on par with a 900k-h-pre-trained Conformer AED (21.2). — [arXiv 2408.17431](https://arxiv.org/html/2408.17431)
- Nearest pseudo-label precedents are not multi-talker ASR: iterative pseudo-labelling for EEND diarization (Interspeech 2021) and signal-level pseudo-labels for an enhancement front-end in MISP-Meeting 2025 `[search summaries]`. — [ISCA archive](https://www.isca-archive.org/interspeech_2021/takashima21_interspeech.html); [arXiv 2505.24446](https://arxiv.org/html/2505.24446v2)

### Inferences
- KD transplants directly: the frozen original Qwen3-ASR is the teacher on each clean source; add a KL term on the matching segment of the student's serialized output. Expect a small gain and tune the weight low (the NTT study found small λ best and large λ harmful). Untested for speech-LLMs.
- The more valuable use of the teacher is labelling: teacher transcripts of clean single-speaker Hokkien/Mandarin/English audio become SOT targets after mixing, so no overlapped audio needs manual transcription.

### Gaps
- No paper found on encoder-representation distillation (student encoder on mixture → per-speaker teacher encoder states) for multi-talker ASR.
- No 2024–2026 paper found on self-training multi-talker ASR with unlabeled real overlapped audio.

---

## Q5. Mixture-of-experts / adapter routing and protecting single-speaker performance

### Takeaway
Mixture-of-LoRA-Experts in the encoder (GLAD, H-SAGE) gives verified gains under SOT, concentrated in high overlap (12.8 → 8.2 WER), but both are small ESPnet Conformer-AED models trained on LibriSpeech mixtures, not adapters on a pretrained speech-LLM. Separately, Google's cascaded-encoder study gives the clearest verified evidence that naive multi-talker training hurts single-speaker accuracy (16.4 → 20.4) and that keeping the single-talker encoder intact removes the regression (16.5).

### Cited Findings
- GLAD (Guo et al., arXiv 2509.13093): MoLE blocks replace linear layers in the encoder; a gate fuses global speaker-aware context with local features; SOT output. Training data 1,351.6 h (692.1 h single-speaker + 2-speaker mixtures by overlap level). `[paper table]` WER %, LibriSpeech test / LSM-2mix test / low / mid / high / OA-WER: SOT (32.90M params) 4.6 / 10.7 / 10.4 / 10.2 / 12.7 / 11.1; SOT+SACTC 4.1 / 8.0 / 6.0 / 8.4 / 12.8 / 9.1; GLAD-SOT (35.18M) 4.3 / 8.0 / 7.8 / 7.5 / 10.1 / 8.5. LSM-3mix test (trained on ≤2 speakers): SOT 26.7, GLAD 23.0. Code: none found. — [arXiv 2509.13093v1](https://arxiv.org/html/2509.13093v1)
- H-SAGE (Guo et al., NKU, arXiv 2607.01566, ICME 2026): adds a Speaker-Aware Global Encoder trained with an Overlap-Aware loss on silence / single / overlap states, and holistic gating. ESPnet2, Conformer encoder + 6-block Transformer decoder, 256 hidden units; 35.75M total parameters; 1,770.2 h training data (692.2 h single-speaker). `[paper table]` LibriSpeech test / LSM-2mix test / low / mid / high / OA-WER: SOT 4.5 / 8.3 / 6.2 / 8.9 / 12.8 / 9.3; SOT + local MoLE 3.8 / 6.5 / 5.3 / 6.3 / 10.3 / 7.3; SOT-SACTC 3.8 / 6.7 / 4.8 / 7.2 / 10.8 / 7.6; GLAD-SOT 3.9 / 6.2 / 5.1 / 6.4 / 8.9 / 6.8; H-SAGE 3.8 / 5.7 / 4.7 / 5.7 / 8.2 / 6.2. LSM-3mix zero-shot test: SOT 24.2, GLAD 19.8, H-SAGE 19.5. — [arXiv 2607.01566](https://arxiv.org/html/2607.01566)
- H-SAGE ablation, LibriSpeech test / LSM-2mix test / LSM-3mix test `[paper table]`: full 3.8 / 5.7 / 19.5; without OA-loss 3.7 / 5.8 / 20.1; without SA-Encoder and OA-loss 4.0 / 6.4 / 21.5; without holistic gating 3.9 / 6.2 / 20.9. (The "~3.6M parameters" in the first pass was a misreading; 35.75M is the whole model.) Code: public. — [arXiv 2607.01566](https://arxiv.org/html/2607.01566); [GitHub NKU-HLT/H-SAGE](https://github.com/NKU-HLT/H-SAGE)
- Google, "Cascaded encoders for fine-tuning ASR models on overlapped speech" (arXiv 2306.16398) `[paper table]`, WER % on SingleSpkr / Overlap test sets: SingleTalker-150K 16.4 / 48.1; MT-Baseline 20.4 / 23.1; MT-Cascade-Scratch 17.9 / 22.6; MT-Cascade-Pretrained (multi-talker encoder cascaded on the pre-trained single-talker encoder) 16.5 / 21.3. A multi-talker speech activity detector provides frame-level overlap scores. — [arXiv 2306.16398](https://arxiv.org/html/2306.16398)
- Other single-speaker evidence `[paper tables]`: SOT-Conformer 3.6 on LibriSpeech vs MT-LLM 2.3; Xiaomi-CocktailASR-1 stays within about 0.2 of Qwen3-ASR-1.7b on four single-speaker sets after training with 600k h of single-speaker data. — [arXiv 2409.08596v2](https://arxiv.org/html/2409.08596v2); [arXiv 2609.11274v1](https://arxiv.org/html/2609.11274v1)
- MoE-LoRA in speech-LLMs outside overlap: TEA-ASLP MLC-SLM 2025 system (multilingual MoE-LoRA + CTC-token prompts, top-three) `[search summary]`. — [arXiv 2507.18051](https://arxiv.org/abs/2507.18051)

### Inferences
- MoLE's evidence does not transfer cleanly: those models are trained end to end at 35M parameters. Whether MoLE blocks added to a frozen pretrained encoder beat a plain encoder LoRA of equal size is untested. Treat as a later refinement.
- The dependable part of H-SAGE for this team is the cheap one: an overlap-state auxiliary head/loss on the encoder (worth 0.1 on 2-mix and 0.6 on 3-mix in the ablation) that can double as an overlap detector.
- The Google result supports a hard-switch design: keep the base path untouched and add a multi-talker LoRA (encoder + LLM) that is enabled only in multi-talker mode. With the adapter off the model is bit-identical to the original Qwen3-ASR. No paper evaluates this exact design on a speech-LLM; it is an engineering inference.
- If one always-on adapter is preferred, every system that kept single-speaker accuracy (MT-LLM, Xiaomi, H-SAGE) trained with a large share of single-speaker data (roughly 40–60% of hours).

### Gaps
- No MoE or LoRA-routing paper on an LLM-decoder multi-talker system.
- No paper measuring single-speaker regression of a LoRA-SOT-tuned speech-LLM relative to its own base model.

---

## Q6. Overlap-aware auxiliary tasks: overlap detection, speaker counting, reasoning-before-transcribing

### Takeaway
Speaker-count errors are measurable and not negligible: on real AMI-SDM, an LLM-SOT system labelled 3.2% of single-talker segments as two talkers and merged 11.7% of two-talker segments into one. Explicit counting as an intermediate output helped a 0.5B-LLM system slightly (5.29 vs 5.46 without it), and an encoder overlap-state loss helped H-SAGE slightly.

### Cited Findings
- Speaker counting from SOT output on AMI-SDM, WavLM Large + Vicuna `[paper table]`: actual 1 talker → estimated 1: 96.7%, estimated 2: 3.2%; actual 2 → 1: 11.7%, 2: 76.9%, 3: 11.0%; actual 3 → 2: 39.3%, 3: 48.4%; actual 4 → 3: 53.0%, 4: 35.0%. The AED with the same encoder: 1 → 2 in 7.6%; 2 → 2 in 73.0%. Conformer AED pre-trained on 900k h: 1 → 1 97.2%, 2 → 2 80.5%. — [arXiv 2408.17431](https://arxiv.org/html/2408.17431)
- CoT ablation `[paper table]`: removing the speaker-number line from the reasoning raises TS-WER 5.29 → 5.46 (2-mix) and 13.57 → 14.01 (3-mix). CoT as a whole took single-speaker WER from 8.1 to 3.88. — [arXiv 2509.15612v2](https://arxiv.org/html/2509.15612v2)
- Xiaomi CoT mode `[paper table]`: gains of 0.24 on LibriMix 2mix and ≤0.1 elsewhere; the reasoning also yields "estimated speaker counts and temporal activity patterns". — [arXiv 2609.11274v1](https://arxiv.org/html/2609.11274v1)
- H-SAGE overlap-aware loss `[paper table]`: 5.8 → 5.7 (2-mix), 20.1 → 19.5 (3-mix). — [arXiv 2607.01566](https://arxiv.org/html/2607.01566)
- Rejection behaviour can backfire without balanced training `[paper table/text]`: Gemini-2.5-pro with a "may output empty" target-speaker prompt falsely rejected 21.31% of LibriSpeech single-speaker utterances; Xiaomi's trained model 0.36%. — [arXiv 2609.11274v1](https://arxiv.org/html/2609.11274v1)
- STAM-ASR (arXiv 2609.29805): AudioLLM with a diarization module learned from frozen AudioLLM features plus speaker and conversation memories. cpWER % with reference vs predicted speaker activity `[paper table]`: AMI-SDM 38.8 vs 64.6; LibriCSS 18.5 vs 53.3; AMI-IHM 25.3 vs 51.5. Authors name speaker-activity tracking as the main open problem. — [arXiv 2609.29805](https://arxiv.org/html/2609.29805)
- Google MT-SAD used as the overlap detector for multi-talker decoding. — [arXiv 2306.16398](https://arxiv.org/html/2306.16398)

### Inferences
- A leading count token ("<1spk>" / "<2spk>") before the transcript is the one-token version of the CoT idea: negligible latency, gives the router signal for Q5, and its accuracy can be scored directly. Supported in direction by the CoT ablation; the exact design is untested.
- Expect two failure types and measure both: false second speaker on clean audio (3.2% in the AMI data point) and merged speakers on true overlap (11.7%). The second is the more frequent one.
- STAM-ASR's 26-point gap between reference and predicted activity shows that learning diarization-like activity inside an AudioLLM is still immature; the team's "no diarization" requirement avoids this.

### Gaps
- No controlled study of a count token or overlap head on hallucination rate for a speech-LLM doing all-speaker transcription.

---

## Q7. Hybrid system-level designs not in the covered list

### Takeaway
"Speaker-agnostic activity streams" (HEAT) matches the "all text, no speaker labels" requirement in spirit: any number of speakers is merged into two non-overlapping streams and the recogniser runs twice, more than doubling throughput for under 1 WER point. It still needs a diarizer today; its authors list "train a system to directly output the two streams" as future work.

### Cited Findings
- "Scaling Multi-Talker ASR with Speaker-Agnostic Activity Streams" (arXiv 2510.03630): built on DiCoW with Whisper-large-v3-turbo + CTC head; speaker activities are merged into two streams by a heuristic. With oracle activities, tcORC-WER % on AMI-SDM / ICSI-SDM `[paper table]`: first-available 32.41 / 40.45 (model collapse, both streams near-identical); alternating 22.20 / 25.47; recency-continuity 20.64 / 24.42; speaker-continuity 19.71 / 24.94; per-speaker diarization conditioning 17.18 / 23.84. — [arXiv 2510.03630](https://arxiv.org/html/2510.03630)
- With a real diarizer (DiariZen) and beam 5 `[paper table]`: per-speaker masks AMI 18.34 at RTFx 2.05, ICSI 25.55 at RTFx 1.50; two-stream HEAT AMI 18.99 at RTFx 4.57, ICSI 26.24 at RTFx 3.89. — [arXiv 2510.03630](https://arxiv.org/html/2510.03630)
- Overlap sensitivity on SparseLibriMix, 2-speaker, per-speaker vs HEAT `[paper table]`: 0% overlap 6.56 / 6.34; 10% 7.90 / 7.48; 20% 10.04 / 11.83. 3-speaker at 20%: 30.29 / 33.10. — [arXiv 2510.03630](https://arxiv.org/html/2510.03630)
- Gated multi-talker path: Google cascaded encoders (Q5). — [arXiv 2306.16398](https://arxiv.org/html/2306.16398)
- Caution on generic separation front-ends: a 2026 study found SAM-Audio pre-processing increased Whisper WER on Bengali and English (denoising context, not two-speaker overlap) `[search summary]`. — [arXiv 2603.04710](https://arxiv.org/pdf/2603.04710)

### Inferences
- For the team's two-speaker utterance-level goal, two speaker-agnostic streams is the same thing as two-speaker SOT; HEAT becomes interesting only for long meetings with more than two participants, where it caps cost at two decodes.
- HEAT's results are at low overlap ratios (≤20%); nothing is shown for heavy overlap.

### Gaps
- No published RTF/VRAM comparison of a gated LoRA vs an always-on multi-talker speech-LLM.

---

## Q8. Discrete-token / generative joint separation-and-recognition

### Takeaway
Token-space separation with language models is active (TokenSplit 2023, TSELM 2024, CodeSep and SLM-SS 2026), but these generate audio tokens per speaker and are scored by running ASR on the result. None is an overlapped-ASR recipe for an encoder → LLM text model. Poor fit.

### Cited Findings
- SLM-SS (arXiv 2610.27785): a continuous encoder on the mixture guides an autoregressive decoder over codec tokens (Encodec or WavTokenizer); trained on Libri2Mix 100 h + 360 h. Libri2Mix: 7.24% WER, 6.92% with per-talker NAR, 8.97% for the WavTokenizer variant `[paper abstract/table]`. Reported RTF includes beam search with beam size 20 `[paper text]`; RTF value Not Reported here. — [arXiv 2610.27785v1](https://arxiv.org/html/2610.27785v1)
- TokenSplit (arXiv 2308.10415): separation + transcription + TTS in one discrete-token seq2seq model `[search summary]`. — [arXiv 2308.10415](https://arxiv.org/abs/2308.10415)
- CodeSep (arXiv 2601.12757): codec-driven separation and compression, not recognition; TSELM (arXiv 2409.07841): discrete-token target speaker extraction, needs enrollment `[search summaries]`. — [arXiv 2601.12757](https://arxiv.org/html/2601.12757); [arXiv 2409.07841](https://arxiv.org/pdf/2409.07841)

### Inferences
- SLM-SS's best separated-speech WER (6.92%) is no better than direct text-output systems on comparable 2-speaker data (4.66–5.2% in Q1), while generating far more tokens and using beam 20. Qwen3-ASR has no audio-token head, so this would be a new model.

### Gaps
- No paper found feeding discrete tokens of a mixture to an LLM for direct multi-talker text output.

---

## Q9. General audio-LLMs on overlapped speech; other 2025–2026 speech-LLM multi-talker work; challenges

### Takeaway
Verified numbers now exist showing general audio-LLMs fail on overlap without task-specific training: SALMONN-7B 32.9% on 2-speaker LibriSpeechMix (vs 5.2% for a trained model), Gemini-2.5-pro 30–76% target-speaker WER, Qwen2.5-Omni-7B 58–67% cpWER on meetings, and Qwen3-ASR-1.7b 38–40% on far-field meetings. Prompting alone is not a solution.

### Cited Findings
- SALMONN-7B, official prompt, WER % `[paper table]`: LibriSpeech 2.4; LibriSpeechMix 2-spk 32.9; 3-spk 45.9. SALMONN was trained on tasks including overlapped speech recognition. — [arXiv 2409.08596v2](https://arxiv.org/html/2409.08596v2)
- Gemini-2.5-pro with a target-speaker prompt and reference audio `[paper table]`: LibriMix 2mix 48.41, 3mix 76.10; LibriSpeechMix 2mix 30.69, 3mix 52.34; AMI-SDM 52.95; AliMeeting-Far 56.75. — [arXiv 2609.11274v1](https://arxiv.org/html/2609.11274v1)
- Qwen3-ASR-1.7b and StepAudio2 on the same tables `[paper table]`: 68.75–160.82 and 71.23–164.66 on synthetic mixtures (target-speaker scoring); AMI-SDM 38.18 and 110.50; AliMeeting-Far 39.64 and 76.82. — [arXiv 2609.11274v1](https://arxiv.org/html/2609.11274v1)
- Qwen2.5-Omni-7B on roughly 20 s reference-defined meeting segments, cpWER / tcpWER@5 (WER) `[paper table]`: AMI-IHM 58.5 / 60.0 (57.7); AMI-SDM 66.5 / 68.1 (65.6); ICSI 46.9 / 47.5 (45.6); LibriCSS 21.6 / 21.8 (21.4); NOTSOFAR-1 67.0 / 67.9 (65.2). Whisper-large-v3 + pyannote: AMI-SDM 61.6 / 63.1 (58.7). — [arXiv 2609.29805](https://arxiv.org/html/2609.29805)
- Unmodified Whisper on Libri2Mix test-clean for one target speaker `[paper tables]`: 78.92 (large), 85.42 (medium), 92.24 (small); Whisper-small 69.26 in another study. — [arXiv 2312.08079](https://arxiv.org/html/2312.08079); [arXiv 2609.30476](https://arxiv.org/html/2609.30476)
- Omnilingual-MsASR (arXiv 2610.29455): 297 long-form audio-visual instances in six languages for speaker-attributed transcription by omni LLMs; the paper says robustness across "speaker counts, and overlap conditions … remains insufficiently characterized". No per-overlap-ratio result was found in the text read. — [arXiv 2610.29455](https://arxiv.org/html/2610.29455)
- Diarization-conditioned speech-LLMs (comparison only): "Diarization-Aware Multi-Speaker ASR with LLMs" (arXiv 2506.05796), DM-ASR (arXiv 2604.22467), TagSpeech (arXiv 2601.06896), SpeakerLM (arXiv 2508.06372), "Grounding Spoken LLMs … via Diarization Conditioning" (arXiv 2606.18134) `[search summaries / titles only]`. TagSpeech-AMI as re-evaluated by STAM-ASR's authors: AMI-SDM 55.3 / 57.8 (40.6) `[paper table, not an exact replication per its footnote]`. — [arXiv 2506.05796](https://arxiv.org/pdf/2506.05796); [arXiv 2609.29805](https://arxiv.org/html/2609.29805)
- MLC-SLM 2025/2026 challenge systems (Whisper-large-v3 + adaptor + LoRA LLM; dual encoders; MoE-LoRA; a Qwen2.5-Omni-7B baseline in 2026) target multilingual conversational ASR; no overlap-specific method or overlap-only score in the retrieved descriptions `[search summaries]`. — [arXiv 2508.14916](https://arxiv.org/html/2508.14916); [arXiv 2609.11724](https://arxiv.org/pdf/2609.11724)
- Survey: He & Whitehill, "Survey of End-to-End Multi-Speaker ASR for Monaural Audio" (arXiv 2505.10975; Computer Speech & Language) `[search summary]`. — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0885230825001500)

### Inferences
- The gap between untrained and trained systems (32.9 → 5.2; 69 → 4) is one order of magnitude, and it closes with simulated-mixture training alone. The team should expect near-unusable zero-shot overlap output from Qwen3-ASR-0.6B and large gains from the first fine-tuning run.
- Even SALMONN, which had overlapped-speech recognition among its training tasks, stays at 32.9%: listing the task is not enough; data volume and encoder adaptation matter.

### Gaps
- No zero-shot overlap number for Qwen3-ASR-0.6B, GPT-4o-transcribe, or Qwen-Omni on a two-speaker mixture set.
- CHiME-8/9 NOTSOFAR speech-LLM system descriptions were not covered.

---

## Q10. Cross-lingual, code-switching, Mandarin and Taiwanese-Hokkien overlapped speech

### Takeaway
Evidence is thin. Verified cross-lingual results are MT-LLM's English–German mixtures (about 21–23% WER, limited by the lower-resource language) and Mandarin far-field meeting scores from Xiaomi's report. No published work was found on overlapped speech with intra-utterance code-switching or on Taiwanese Hokkien or other Chinese-dialect mixtures.

### Cited Findings
- MT-LLM `[paper table/text]`: En-De-Mix 2-spk 21.0, 3-spk 22.8, against 5.2 / 10.2 for English-only mixtures and 11.7 for single-speaker German; German was about 10% of training data and "noisier". The language-selection prompt (TL) scores 21.8 vs a 21.0 best-matching reference, so language selection itself costs under 1 point. — [arXiv 2409.08596v2](https://arxiv.org/html/2409.08596v2)
- Mandarin real meetings `[paper table]`: AliMeeting-Far — Xiaomi-CocktailASR-1 20.63, Qwen3-ASR-1.7b 39.64, Gemini-2.5-pro 56.75; AliMeeting-near single-speaker 6.57 vs 6.39. — [arXiv 2609.11274v1](https://arxiv.org/html/2609.11274v1)
- Whisper-Sidecar joint model reports acceptable zero-shot multi-talker ASR on Mandarin AishellMix after English-only training `[search summary of abstract]`. — [arXiv 2407.09817](https://arxiv.org/abs/2407.09817)
- SLM-SS reports its perceptual-quality advantage persists on unseen Chinese mixtures without fine-tuning (separation quality, not ASR) `[paper abstract]`. — [arXiv 2610.27785v1](https://arxiv.org/html/2610.27785v1)
- Persona-ASR (ISSAI, 2026): Kazakh–English target-speaker ASR with enrollment, WavLM-Base+ with per-language CTC heads; released KazMix3 and PersonaMix; Kazakh about 14 points worse than English `[search summary of model card/abstract]`. — [Hugging Face](https://huggingface.co/issai/Persona-ASR); [DOI](https://doi.org/10.3390/make8080246)

### Inferences
- The lower-resource language dominates the error of a cross-lingual mixture (MT-LLM: 21.0 mixed vs 11.7 German alone vs 5.2 English mixed). Expect Hokkien-in-mixture to be the weak side; plan a per-pair test matrix (Hokkien+Hokkien, Hokkien+Mandarin, Mandarin+English, …).
- Prompting by language worked almost without selection error for English vs German. Whether it holds for Mandarin vs Hokkien, which are far closer acoustically, is unknown.
- No Hokkien overlap corpus surfaced; simulation from the team's single-speaker data is the only data route.

### Gaps
- No published overlapped-speech work on Taiwanese Hokkien, Cantonese, or other Chinese dialects; none on overlap combined with intra-utterance code-switching.

---

## Q11. Ranking by promise for this team (researcher's assessment)

### Takeaway
The verified evidence points to a training-recipe answer rather than a new architecture: adapt the encoder to mixtures, keep a large single-speaker share, add instruction tasks and a count signal as auxiliaries, and keep the multi-talker weights switchable. Structurally heavy options have weaker fit to a 0.6B model with a simple-inference requirement.

### Cited Findings
- Encoder adaptation is decisive (46.7 → 10.2). — [arXiv 2408.17431](https://arxiv.org/html/2408.17431)
- Multi-task instructions improve the all-speaker task (5.5 → 5.2); ordinal prompt is the weakest (9.0 vs 5.7). — [arXiv 2409.08596v2](https://arxiv.org/html/2409.08596v2)
- Naive multi-talker training degrades single-speaker WER (16.4 → 20.4); preserving the single-talker encoder restores it (16.5). — [arXiv 2306.16398](https://arxiv.org/html/2306.16398)
- A 0.5B-LLM + 0.3B-encoder model reaches 4.84% TS-WER on Libri2Mix. — [arXiv 2509.15612v2](https://arxiv.org/html/2509.15612v2)
- Second mixture-aware encoder: 16.1 → 10.7 on 3-speaker. — [arXiv 2409.08596v2](https://arxiv.org/html/2409.08596v2)
- MoLE + overlap loss: 8.3 → 5.7 on 2-mix in a 35M model. — [arXiv 2607.01566](https://arxiv.org/html/2607.01566)
- KD from single-talker teacher: 15.8 → 15.0 CER. — [arXiv 2305.15971](https://arxiv.org/html/2305.15971)
- Two-stream decoding on a diarizer: +0.65 WER for 2.2× throughput. — [arXiv 2510.03630](https://arxiv.org/html/2510.03630)

### Inferences (ranking, most to least promising)

| Rank | Approach | New modules | Added inference cost | Evidence | Why this rank |
|---|---|---|---|---|---|
| 1 | Mixture adaptation of the existing encoder (LoRA or partial unfreeze), then projector + LLM LoRA, with a large single-speaker share | None | 0 | Strong: 46.7 vs 10.2 frozen vs trained; staged training best | Precondition for everything else; not optional |
| 2 | Instruction multi-task training (all-speaker + sex / language / order / keyword prompts) | None | 0 if only the all-speaker prompt is used at inference | Verified on one speech-LLM: helps the main task; order prompt weak | Free auxiliary supervision; also gives language-selective transcription |
| 3 | Switchable multi-talker adapter + count token / overlap-state head | One token or a tiny head | ~0 | Components verified separately (Google cascade; CoT count ablation; H-SAGE loss; AMI counting errors) | Only design that guarantees unchanged single-speaker behaviour; measures and limits speaker-count errors |
| 4 | Small mixture-aware second encoder (WavLM-base class) | ~95M encoder + adapter | Moderate, every utterance | Verified large gain with frozen encoders; no test against "adapt first encoder" | Try only if rank 1 leaves a gap, especially at 3 speakers |
| 5 | Single-talker-teacher KD on simulated mixtures | None (training only) | 0 | Verified but small (about 5–9% relative), non-LLM | Cheap add-on; main value of the teacher is pseudo-labelling Hokkien data |
| 6 | Encoder MoLE (GLAD / H-SAGE) | MoLE blocks + gate | Small, unmeasured | Verified on 35M from-scratch models only | Unproven as an adapter on a pretrained speech-LLM |
| 7 | Ordinal-prompt multi-pass decoding as the main mode | None | One extra prompt per speaker | Verified weaker than single-pass (9.0 vs 5.2) | Keep as auxiliary task, not as inference path |
| 8 | Speaker-agnostic two-stream decoding on a diarizer | Diarizer + conditioning | Diarizer + 2 decodes | Verified on AMI/ICSI at low overlap | For long >2-speaker meetings; beyond current requirement |
| 9 | Enrollment-based LLM TS-ASR; conditional chain; token-space separation | Reference audio / new modules / new output modality | Higher | Strong for TS-ASR; weak or off-target for the others | Violates "no enrollment", or not competitive, or not a fine-tune |

- Ranks 1–3 share one simulated-mixture training set and are complementary; rank 3's count token is also the router.
- None of these alternatives has published evidence of beating a well-tuned LLM-SOT system by a wide margin on two-speaker mixtures; they shape how SOT-style training should be done on Qwen3-ASR rather than replace it.

### Gaps
- Every ranked item lacks direct evidence on Qwen3-ASR-0.6B, on Mandarin/Hokkien mixtures, and on measured RTF/VRAM for a speech-LLM. The ranking is a prior for experiments.
- Items still tagged `[search summary]` (Cocktail HuBERT tables, conditional-chain abstract, 2019 KD paper, Persona-ASR, MLC-SLM systems, TokenSplit) were not re-read in full.
