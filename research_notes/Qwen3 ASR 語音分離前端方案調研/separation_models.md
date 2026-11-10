# Single-channel neural speech separation models as an external front-end for Qwen3-ASR-0.6B (2020–2026)

Evidence tags used on every figure:
- **[P]** = read this session from the primary source (paper full text on arxiv.org/html, official README, or model card).
- **[P2]** = read this session in a *different* paper's comparison table (secondary citation of the number; measurement conditions may differ).
- **[S]** = search-result snippet only, not opened.
- "Not Reported" = not stated in the source I read. "Not captured" = may exist in the source but I did not read that part; treat as unknown.

Notation: SI-SDRi and SI-SNRi are the same quantity (dB improvement over the mixture). "DM" = dynamic mixing augmentation. All WSJ0-2mix / WHAM! / WHAMR! benchmark numbers are **8 kHz, fully overlapped, 2 speakers** unless stated.

---

## 1. Model families: separation quality, size, compute, sample rate, checkpoints

### Takeaway
Almost every headline SI-SDRi number (22–25.5 dB) is an 8 kHz WSJ0-2mix result; usable 16 kHz checkpoints with a permissive license are few: MossFormer2_SS_16K (Apache-2.0), SepFormer-WHAMR16k (Apache-2.0), Asteroid ConvTasNet Libri2/3Mix 16k (CC-BY-SA), TIGER (Apache-2.0, 16 kHz EchoSet), and pyannote PixIT (MIT, gated). Where downstream WER is reported, TF-domain models (TF-GridNet / TF-CrossNet / TF-Locoformer / SR-CorrNet) give the lowest WER; older time-domain models (SepFormer, BSRNN) leave large WER despite high SI-SDRi.

### Cited Findings

**Master table (WSJ0-2mix unless noted; 8 kHz)**

| Model (year) | Params (M) | WSJ0-2mix SI-SNRi (dB) | Libri2Mix | WHAM! | WHAMR! | Compute as reported | Tag / source |
|---|---|---|---|---|---|---|---|
| Conv-TasNet (2019) | 5.1 | 15.3 | 12.2 | 12.7 | 8.3 | 10.5 G MACs/s | [P2] [SepReformer Table 5](https://arxiv.org/abs/2406.05983); WHAMR! [P2] [MossFormer2 Table 3](https://arxiv.org/abs/2312.11825) |
| DPRNN (2020) | 2.6 | 18.8 | 16.1 | 13.7 | 10.3 | 88.5 G MACs/s | [P2] same sources |
| DPTNet (2020) | 2.7 | 20.2 | 16.7 | 14.9 | Not Reported | 102.5 G MACs/s | [P2] [SepReformer](https://arxiv.org/abs/2406.05983) |
| SuDoRM-RF (2020/21) | 6.4 | 18.9 | 14.0 | 13.7 | 13.5 (+DM) | 10.1 G MACs/s | [P2] [SepReformer](https://arxiv.org/abs/2406.05983); WHAMR! [P2] [SR-CorrNet Table III](https://arxiv.org/abs/2603.29097) |
| SepFormer (2021) | 25.7–26.0 | 20.4 (22.3 +DM) | 16.5 (19.2 per MossFormer2/TF-Locoformer tables) | 14.7 (16.4 +DM) | 14.0 (+DM) | 86.9 G MACs/s [P2 SepReformer]; 69.6 GMACs/s [P2 RE-SepFormer] | [P] abstract [SepFormer](https://arxiv.org/abs/2010.13154); [P2] [SepReformer](https://arxiv.org/abs/2406.05983), [RE-SepFormer](https://arxiv.org/abs/2206.09507) |
| TF-GridNet (2022/23) | 14.5 | 23.5 (no DM) | 19.8 (ESPnet re-run, per SPMamba) | 16.9 (ESPnet re-run) | SI-SDR 11.0 dB absolute from −6.1 dB unprocessed (DNN1, 1-ch); listed as 17.1 SI-SNRi (5.5 M config) in TF-Locoformer table and 16.7 in SR-CorrNet table | 460.8 G MACs/s [P2 SepReformer]; 445.56 [P2 SPMamba]; 323.75 at 16 kHz [P2 TIGER]; 231.1 [P2 SepMamba] | [P] [TF-GridNet](https://arxiv.org/abs/2211.12433); [P2] [SPMamba](https://arxiv.org/abs/2404.02063), [TF-Locoformer](https://arxiv.org/abs/2408.03440), [SR-CorrNet](https://arxiv.org/abs/2603.29097) |
| MossFormer (2023) | 42.1 | 22.8 (+DM) | Not captured | 17.3 | 16.3 | RTF 0.038 (GPU, model unspecified in my read) | [P] [MossFormer2 Tables 1,3](https://arxiv.org/abs/2312.11825) |
| MossFormer2 (2023/24) | 55.7 (S: 37.8) | 24.1 (S: 23.2) | 21.7 | 18.1 | 17.0 | RTF 0.053 (S: 0.036) on the single training GPU; MACs Not Reported | [P] [MossFormer2](https://arxiv.org/abs/2312.11825) |
| SepTDA (2024) | 21.2 | 23.7 (24.0 for the alternative-L config; abstract: 24.0 / 23.7 on WSJ0-2/3mix with one 2-and-3-speaker model) | Not Reported | Not Reported | Not Reported | Not Reported | [P] [SepTDA](https://arxiv.org/abs/2401.12473) |
| DPMamba (2024) | XS 2.3 / S 8.1 / M 15.9 | S: 20.6 with DM (20.0 without); unidirectional variant 16.9 | Not Reported | Not Reported | Not Reported | Not captured | [P] [Dual-path Mamba](https://arxiv.org/abs/2403.18257) |
| SPMamba (2024) | 6.14 | 22.5 | 19.9 | 17.4 | Not Reported | 238.69 G MACs/s (own table); 119.35 GMAC/s in SepMamba's table | [P] [SPMamba](https://arxiv.org/abs/2404.02063); [P2] [SepMamba](https://arxiv.org/abs/2410.20997) |
| SepMamba (2024) | S 7.2 / M 22 | S 21.2, M 22.7 (+DM); causal S 19.2, causal M 21.4 | Not Reported | Not Reported | Not Reported | 12.46 / 37.0 GMAC/s; forward pass 17.84 / 27.25 ms for 4 s of 8 kHz audio on A100; peak training memory 2.00 / 3.04 GB | [P] [SepMamba](https://arxiv.org/abs/2410.20997) |
| SepReformer (NeurIPS 2024) | T 3.5 / S 4.3 / B 14.2 / M 17.3 / L 59.4 | T 22.4 / S 23.0 / B 23.8 / M 24.2 / L 25.1 (+DM) | T 19.7 / S 20.6 / B 21.6 / M 22.0 | T 17.2 / B 17.6 / M 17.8 / L 18.4 (+DM) | L 17.2 (+DM) | T 10.4 / S 21.3 / B 39.8 / M 81.3 / L 155.5 G MACs/s (for 16000 samples) | [P] [SepReformer](https://arxiv.org/abs/2406.05983) |
| TF-Locoformer (IWAENC 2024) | S 5.0 / M 15.0 / L 22.5 | S 22.0 (22.8 +DM) / M 23.6 (24.6 +DM) / L 24.2 (25.1 +DM) | M 22.1 | Not captured | S 17.4 / M 18.5 (no DM) | L: 204.3 G MACs [P2 SR-CorrNet] | [P] [TF-Locoformer](https://arxiv.org/abs/2408.03440); [P2] [SR-CorrNet](https://arxiv.org/abs/2603.29097) |
| TIGER (ICLR 2025), **16 kHz** | 0.82 (small and large) | Not Reported | 16.67 (small) / 17.97 (large) SI-SDRi on Libri2Mix 16 kHz train-100 min | Not Reported | EchoSet: 12.58 / 13.73 (TF-GridNet 12.85) | 7.65 / 15.27 G MACs/s | [P] [TIGER](https://arxiv.org/abs/2410.01469) |
| SR-CorrNet (arXiv Mar 2026) | S 7.0 / B 13.6 / L 38.1 | S 23.6 / B 24.1 / L 25.5 (+DM) | Not captured | Not captured | B 19.7 (1-ch) | S 88.9 / B 177.3 / L 467.6 G MACs | [P] [SR-CorrNet](https://arxiv.org/abs/2603.29097) |

Additional per-family facts:

- **Sample rates.** SepReformer: "networks were trained with 4-s-long segments at a 8-kHz sampling rate" [P] — [SepReformer](https://arxiv.org/abs/2406.05983). SPMamba public-benchmark models trained at 8 kHz; its Echo2Mix set is 16 kHz [P] — [SPMamba](https://arxiv.org/abs/2404.02063). TF-Locoformer: separation sets "always used the fully overlapped min version with a sampling rate of 8 kHz"; only the DNS enhancement set was 16 kHz [P] — [TF-Locoformer](https://arxiv.org/abs/2408.03440). SepTDA: 8 kHz [P] — [SepTDA](https://arxiv.org/abs/2401.12473). TF-GridNet: 8 kHz for WSJ0-2mix/SMS-WSJ/WHAMR!, with a documented 16 kHz STFT configuration (512-point DFT) used for its 16 kHz enhancement tasks [P] — [TF-GridNet](https://arxiv.org/abs/2211.12433). TIGER: all three datasets (Libri2Mix train-100 min, LRS2-2Mix, EchoSet) at 16 kHz [P] — [TIGER](https://arxiv.org/abs/2410.01469).
- **Causality.** TF-GridNet: "Our system is non-causal" [P] — [TF-GridNet](https://arxiv.org/abs/2211.12433). SepMamba reports causal variants at 19.2 / 21.4 dB SI-SNRi vs. prior causal models UX-NET 13.6 and Causal Deep CASA 15.2 [P] — [SepMamba](https://arxiv.org/abs/2410.20997). DPMamba unidirectional (forward-only) loses >3 dB (16.9 vs 20.6) [P] — [DPMamba](https://arxiv.org/abs/2403.18257).
- **SPMamba efficiency claim.** "SPMamba's computational load was only half that of TF-GridNet in terms of computational complexity and parameters"; on Echo2Mix (noisy/reverberant, 16 kHz) SPMamba 15.3 dB SI-SNRi vs TF-GridNet 12.8 [P] — [SPMamba](https://arxiv.org/abs/2404.02063).
- **Benchmark ceiling caveat.** WSJ0-2mix reference signals are themselves noisy, which "limits the achievable SI-SDR, or leads to undesired noise in the separated outputs" [P, abstract] — [SI-SDR with noisy references](https://arxiv.org/abs/2508.14623). MossFormer abstract cites an SI-SDRi upper bound of 23.1 dB on WSJ0-2mix [P] — [MossFormer](https://arxiv.org/abs/2302.11824); later models report above it (e.g., 25.5).
- **Diffusion / generative separators.** DiffSep uses 30 prediction steps each with one corrector step at inference; evaluated on WSJ0-2mix 8 kHz and a 16 kHz set [P] — [DiffSep](https://arxiv.org/abs/2210.17327). "Separate And Diffuse" is listed at 23.9 dB on WSJ0-2mix [P2] — [SepReformer Table 5b](https://arxiv.org/abs/2406.05983). Fast-GeCo collapses the reverse process to a single step and reports "improving SI-SNR by 22-35% relative to SepFormer" on Libri2Mix-noisy and out-of-domain WSJ+noise; SepFormer baseline there is 10.58 dB SI-SNRi on Libri2Mix noisy; the paper notes multi-step diffusion is "much slower than a discriminative model" [P] — [Fast-GeCo](https://arxiv.org/abs/2406.07461). EDSep (Jan 2025) claims SOTA SI-SNRi on WSJ0-2mix, number not seen [S] — [EDSep listing](https://www.newx.sg/paper/detail/d6aa5988-f822-11f0-b84c-00163e10baa7).
- **Token/LM-based separation (2026).** SLM-SS (Encodec tokens + AR/NAR speech LM) on Libri2Mix clean, max mode, **16 kHz**, scored with Whisper-base: SLM-SS 7.24% WER (6.92% with per-talker NAR), MossFormer2 8.78%, SepFormer 17.77%, BSRNN 20.02%, TF-Locoformer 5.88% (lowest) [P] — [SLM-SS](https://arxiv.org/abs/2610.27785).
- **CrossNet / TF-CrossNet.** Claims to surpass TF-GridNet (8.2 M) by 0.5 dB SI-SDR [S]; abstract confirms it targets reverberant/noisy separation with global + cross-band + narrow-band modules [P] — [CrossNet](https://arxiv.org/abs/2403.03411).

**Downstream WER with a separation front-end (most decision-relevant):**

- Noisy Libri2Mix, TF-CrossNet front-end, independently trained back-ends: Whisper Large-v3 gets 5.6 / 5.0% WER (dev/test) on separated streams vs 2.5% on the ground-truth single-speaker clean audio; a WavLM+Conformer back-end trained on clean speech gets 5.1 / 5.1% (2.3% on ground truth). Compared end-to-end baselines on the same test set: SOT-Conformer 23.3%, SOT-WavLM-Conformer 17.1%, WavLM-LLM fine-tuned 9.0%, PIT-SOT 7.0%, TSE-Whisper 12.0% [P] — [Decoupling separation and ASR](https://arxiv.org/abs/2503.17886).
- Same paper: the front-end "introduces processing artifacts that degrade the ASR backend trained on clean speech" (stated as the conventional concern); their finding is that with a strong separator a clean-trained back-end works, but on 1-ch SMS-WSJ-Large the clean-trained back-end only wins at T60 0.2–0.5 s (SNR 20–40 dB with the front-end trained on SMS-WSJ; SNR 0–40 dB with the front-end trained on the larger SMS-WSJ-Large), while "the mainstream approach achieves lower WER at higher T60 and lower SNR values" [P] — [Decoupling](https://arxiv.org/abs/2503.17886).
- LibriCSS (real replayed recordings, single microphone, continuous separation with 4 s windows / 3 s shift), TF-GridNet + mixture-encoder acoustic model, ORC-WER with Transformer LM: 0S 4.5 / 0L 3.5 / 10% 3.6 / 20% 5.2 / 30% 7.1 / 40% 8.4, avg 5.6; with oracle segmentation avg 4.1; the same ASR baseline scores 2.3% on LibriSpeech test-clean [P] — [TF-GridNet + mixture encoder](https://arxiv.org/abs/2309.08454).
- LibriCSS single-channel, utterance-wise, original LibriCSS ASR: no processing 11.8 / 11.7 / 18.8 / 27.2 / 35.6 / 43.3 (0S/0L/10/20/30/40% overlap); SR-CorrNet fixed-2-output 6.2 / 6.2 / 6.3 / 6.8 / 7.5 / 8.0; SR-CorrNet with attractor-based dynamic split ("var0-3") 6.0 / 5.8 / 6.1 / 6.6 / 7.3 / 7.7. Continuous-input: no processing 15.4 / 11.5 / 21.7 / 27.0 / 34.3 / 40.5; fix2 7.1 / 6.8 / 7.0 / 7.1 / 8.3 / 9.2; var0-3 7.2 / 6.8 / 6.9 / 6.9 / 8.2 / 9.0. Earlier single-channel baselines in the same table: Conformer separator 12.9 → 27.6 (0S → 40%), U-Net + speaker counting 9.2 → 23.1 [P] — [SR-CorrNet Table VI](https://arxiv.org/abs/2603.29097).
- AMI-SDM real meetings (single distant mic), PixIT joint diarization+separation, no ASR fine-tuning: cpWER Whisper large-v2 38.3% (pyannote 3.1 diarization on the original audio) → 31.7% (PixIT separated sources), −17.2% relative; Whisper small.en 39.6 → 33.9; NeMo Conformer-CTC large 48.9 → 39.4. Utterance-wise WER (no speaker attribution): Whisper large-v2 35.4 → 33.8 (−4.5%); "the bulk of the WER improvement comes from deletions"; substitutions rise slightly (5.2 → 5.5 for large-v2; 10.7 → 12.6 for NeMo) [P] — [PixIT](https://arxiv.org/abs/2403.02288).

**Open checkpoints / code / license (verified this session):**

- SepFormer (SpeechBrain, Apache-2.0): `sepformer-wsj02mix` 22.4 dB SI-SNRi, 8 kHz; `sepformer-libri2mix` 20.6 dB, 8 kHz; `sepformer-libri3mix` 19.0 dB, 8 kHz; `sepformer-wsj03mix` 19.8 dB, 8 kHz; `sepformer-whamr` 13.7 dB, 8 kHz; `sepformer-whamr16k` 13.5 dB SI-SNRi, **16 kHz**. Each card: "does not provide any warranty on the performance achieved by this model when used on other datasets" [P] — [wsj02mix](https://huggingface.co/speechbrain/sepformer-wsj02mix), [libri2mix](https://huggingface.co/speechbrain/sepformer-libri2mix), [libri3mix](https://huggingface.co/speechbrain/sepformer-libri3mix), [wsj03mix](https://huggingface.co/speechbrain/sepformer-wsj03mix), [whamr](https://huggingface.co/speechbrain/sepformer-whamr), [whamr16k](https://huggingface.co/speechbrain/sepformer-whamr16k).
- MossFormer2_SS_16K (ClearerVoice-Studio, Apache-2.0 model card and Apache-2.0 repo LICENSE): **16 kHz**, "trained on large scale datasets including open-sourced and private data". Unified-model scores without per-dataset retraining: LRS2-2Mix (16 kHz) 15.5 dB; WSJ0-2Mix 22.0; Libri2Mix 16.7; WHAM! 17.4 SI-SNRi (the 8 kHz sets were processed at 16 kHz and scored after downsampling) [P] — [HF card](https://huggingface.co/alibabasglab/MossFormer2_SS_16K), [clearvoice README](https://github.com/modelscope/ClearerVoice-Studio/blob/main/clearvoice/README.md).
- SepReformer: Apache-2.0 repo; only SepReformer-B WSJ0-2mix (8 kHz) weights uploaded; README now redirects users to SR-CorrNet-SS [P] — [SepReformer repo](https://github.com/dmlguq456/SepReformer).
- SR-CorrNet-SS: MIT LICENSE; pip-installable library with HF Hub checkpoints `shinuh/sr-corrnet-ss-1ch-wsj-fix-2spk`, `-fix-2spk-l-dm`, `-var-2-3spk`, `-var-2-5spk`, `sr-corrnet-ss-1ch-whamr`; CPU install option documented [P] — [SR_CorrNet_SS repo](https://github.com/dmlguq456/SR_CorrNet_SS). The 1-ch checkpoints I saw are WSJ0/WHAMR! (8 kHz training data per the paper); the 16 kHz LibriCSS models in the paper were trained on simulated 7-channel LibriSpeech mixtures [P] — [SR-CorrNet](https://arxiv.org/abs/2603.29097).
- TF-Locoformer: Apache-2.0; pretrained weights shipped in the repo under each `egs2/*/exp` directory (Libri2Mix, WHAMR!, DNS recipes in ESPnet pipeline) [P] — [merlresearch/tf-locoformer](https://github.com/merlresearch/tf-locoformer).
- SPMamba: Apache-2.0; checkpoints for Libri2Mix and Echo2Mix released 2024-11-22 [P] — [SPMamba repo](https://github.com/JusperLee/SPMamba).
- DPMamba / Mamba-TasNet: checkpoints on Google Drive [P] — [Mamba-TasNet repo](https://github.com/xi-j/Mamba-TasNet). License Not captured.
- SepMamba: code at github.com/andrasschin/SepMamba (per paper) [P] — [SepMamba](https://arxiv.org/abs/2410.20997). Checkpoints/license Not captured.
- TIGER: Apache-2.0; `JusperLee/TIGER-speech` (Jan 2025) plus `TIGER-speech-small` and `-tiny` (Oct 2025) on HF [P] — [TIGER repo](https://github.com/JusperLee/TIGER), [HF API metadata](https://huggingface.co/api/models/JusperLee/TIGER-speech-small).
- SepTDA: demo page only at 42speech.github.io/septda per paper [P] — [SepTDA](https://arxiv.org/abs/2401.12473). Official checkpoint: Not Reported.
- SPGM (SpeechBrain-style single-path model): 22.1 dB WSJ0-2mix, 20.4 dB Libri2Mix, weights at huggingface.co/yipjiaqi/spgm [P, abstract] — [SPGM](https://arxiv.org/abs/2309.12608).

### Inferences
- For a 16 kHz ASR, 8 kHz-only checkpoints (most SOTA releases) require either retraining at 16 kHz or accepting that separated audio has no content above 4 kHz; none of the sources read quantify the ASR cost of that band-limit for Qwen3-ASR.
- SI-SDRi rank does not map to WER rank: MossFormer2 (24.1 dB class) gives 8.78% vs TF-Locoformer 5.88% on the same Libri2Mix/Whisper-base test, and SepFormer (22.3 dB class) gives 17.77% (SLM-SS table) — the comparison systems' training data in that table was not checked, so treat as indicative.
- The only front-ends with published **real-recording single-channel** WER evidence in what I read are TF-GridNet(+mixture encoder), SR-CorrNet (LibriCSS), and PixIT (AMI/AliMeeting).

### Gaps
- No source read reports MACs for MossFormer2; RTF hardware for MossFormer2 is only "the same single GPU card used for training".
- VRAM at inference is essentially unreported except TIGER's table and SepMamba (training peak).
- I did not open CrossNet/TF-CrossNet result tables, DiffSep result tables, or EDSep; their numbers are not recorded here.
- Published MACs for TF-GridNet differ by 2x across papers (231–461 G/s) because of different configurations/sample rates/counting tools; no single authoritative value.
- No separation front-end has a published WER with Qwen3-ASR; all WER evidence uses Whisper, Conformer, hybrid or WavLM back-ends.

---

## 2. Unknown / variable number of speakers and single-speaker input

### Takeaway
Attractor-based models (SepTDA, SR-CorrNet dynamic split, EEND-SS) count 2–3 speakers at ~98–100% accuracy on synthetic fully-overlapped data, and the SR-CorrNet paper gives direct evidence that fixed-2-output separators leak "artificial secondary streams" during single-speaker regions on real long-form audio. The quantified penalty of that leakage on LibriCSS is small (≤0.3 WER points) in that paper.

### Cited Findings
- SepTDA trained on WSJ0-{2,3}mix: unknown-count evaluation gives 24.0 dB with 99.97% counting accuracy on 2-mix and 22.8 dB with 97.93% on 3-mix; with known count 24.0 / 23.7. Model trained on 2–5 speakers: 23.6 (2-mix), 23.5 (3-mix) with known count. Counting = attractor existence probability > 0.5 [P] — [SepTDA](https://arxiv.org/abs/2401.12473).
- Same table: Recursive SS (Takahashi 2019) 14.8 / 12.6 / 10.2 dB on 2/3/4-mix; SepEDA unknown-count 21.1 (99.80%) / 18.4 (97.00%) / 14.4 (90.17%) / 11.6 (96.87%) on 2/3/4/5-mix; Gated DPRNN with unknown count 18.6 / 14.6 / 11.5 / 10.4 [P2] — [SepTDA Table 2](https://arxiv.org/abs/2401.12473).
- SepFormer fixed-count models: WSJ0-3mix 19.5 dB (paper), Libri3Mix checkpoint 19.0 dB (8 kHz) [P] — [SepFormer](https://arxiv.org/abs/2010.13154), [sepformer-libri3mix](https://huggingface.co/speechbrain/sepformer-libri3mix). MossFormer2 WSJ0-3mix 22.2 dB [P] — [MossFormer2](https://arxiv.org/abs/2312.11825).
- EEND-SS (Conv-TasNet separator + EEND-EDA counting, in ESPnet): on mixed Libri2Mix+Libri3Mix, SI-SDRi 9.31 dB vs Conv-TasNet 7.66; speaker counting accuracy 97.9% (98.2% with log-mel features) vs EEND-EDA 86.2%; DER 6.27% [P] — [EEND-SS](https://arxiv.org/abs/2203.17068).
- SR-CorrNet "var0-3" (attractor-based dynamic split handling zero to three active streams) vs fixed-2 split on LibriCSS single-channel: 0S 6.0 vs 6.2, 0L 5.8 vs 6.2 (utterance-wise WER). Authors: "the fixed split tended to produce artificial secondary streams during single-speaker regions, causing spectral leakage that accumulates over long-form processing" [P] — [SR-CorrNet](https://arxiv.org/abs/2603.29097).
- In that same table the fixed-2 separator still *improves* non-overlapped LibriCSS speech over no processing (0S 11.8 → 6.2, 0L 11.7 → 6.2), i.e., on reverberant far-field audio the separator's dereverberation/denoising outweighs its artifacts for that ASR back-end [P] — [SR-CorrNet Table VI](https://arxiv.org/abs/2603.29097).
- TF-GridNet CSS front-end for LibriCSS was trained "on fully overlapping data and single-speaker utterances"; the paper motivates a mixture encoder because "it is important to separate speech in overlapping regions and not negatively affect single-speaker regions". Residual gap on non-overlapped sessions: 0L 3.5% vs 2.3% LibriSpeech test-clean baseline for the same ASR (different acoustic conditions, so not purely separator damage) [P] — [TF-GridNet + mixture encoder](https://arxiv.org/abs/2309.08454).
- PixIT limits the number of speakers in training mixtures-of-mixtures using diarization labels specifically to fix MixIT over-separation, and aligns each separated source with a speaker-activity output so inactive sources can be gated (leakage removal via diarization is an ablation in Table 5) [P] — [PixIT](https://arxiv.org/abs/2403.02288).
- SuDoRM-RF on FUSS (universal sound, variable source count) reports absolute SI-SDR for single-source inputs and notes "a 25% of under-separation" for 2–4 source mixtures [P] — [SuDoRM-RF journal](https://arxiv.org/abs/2103.02644).
- A unified multi-head network with a speaker-count classifier branch for up to 5 speakers in noisy reverberant conditions (Gated DPRNN family / "VSUNOS") [P, abstract] — [Chazan et al.](https://arxiv.org/abs/2011.02329); listed at 7.5 M params, 15.2 dB WHAM!, 12.2 dB WHAMR! [P2] — [MossFormer2 Table 3](https://arxiv.org/abs/2312.11825).
- 2025–2026 newcomers for unknown counts: multi-utterance attractor model keeping each speaker's several utterances on one output channel, 16 kHz simulated reverberant data [P, partial] — [arXiv 2505.16607](https://arxiv.org/abs/2505.16607); REDnet recursive encoder/decoder that detects whether speakers remain and extracts one per iteration, for unknown speakers and variable microphones [P, abstract/intro] — [REDnet](https://arxiv.org/abs/2608.24659); ReSepNet recursive model generalizing to 4–5 speakers [S] — [Interspeech 2025](https://www.isca-archive.org/interspeech_2025/alizadeh25_interspeech.html); coarse-to-fine recursive separation [S] — [arXiv 2203.16054](https://arxiv.org/abs/2203.16054).

### Inferences
- All high counting accuracies above are on fully overlapped, clean synthetic mixtures; none of the sources read reports counting accuracy on real partially-overlapped recordings.
- A front-end pipeline needs an explicit decision for 1-speaker segments (attractor/activity gating as in SR-CorrNet/PixIT/EEND-SS, or bypass by an overlap detector); fixed-2-output checkpoints (SepFormer, MossFormer2_SS_16K, TIGER) give no such signal in what I read.

### Gaps
- No source read measures WER degradation on **clean, close-talk, single-speaker** audio after passing through SepFormer / MossFormer2 / TIGER. The LibriCSS 0S/0L numbers are far-field and confound dereverberation gains.
- How MossFormer2_SS_16K behaves with one or three speakers, and its number of output streams, is Not Reported on the card/README parts I read.
- SepTDA has no confirmed public checkpoint.

---

## 3. Generalization: real recordings, noise/reverb, other languages

### Takeaway
Published evidence says supervised separators trained on English synthetic mixtures degrade sharply under domain shift, and the one controlled English→Mandarin test read here shows a large drop (Conv-TasNet 11.98 → 2.08 dB SI-SNRi). No evaluation of any kind was found for Taiwanese Hokkien or for cross-lingual (code-mixed speaker pair) mixtures.

### Cited Findings
- Libri2Mix-trained models tested zero-shot (SDRi / SI-SNRi, dB): Conv-TasNet 12.41 / 11.98 in-domain → 6.83 / 6.45 on WHAMR! → 2.57 / 2.08 on Mandarin Aishell2Mix; DPCCN 13.48 / 13.04 → 8.99 / 8.50 → 5.78 / 5.09. Authors: "Separation performance degradation caused by the language mismatch is much more severe than the acoustic reverberation." Trained in-domain on Aishell2Mix the same models reach 9.00 / 8.32 (Conv-TasNet) and 8.86 / 8.14 (DPCCN). The TF-domain DPCCN was more robust cross-domain than time-domain Conv-TasNet [P] — [Heterogeneous Separation Consistency Training](https://arxiv.org/abs/2204.11032).
- A JASA 2021 study (Healy et al.) reports a network trained on English generalizing to Mandarin for speaker separation and dereverberation ("can generalize across different languages") [S] — [Healy et al. PDF listing](https://pnlwang.github.io/papers/HealyEtAl.jasa21.pdf). This conflicts in emphasis with the Aishell2Mix result above; not opened, so conditions cannot be compared.
- GlobalPhone-based multilingual mixture database proposed because existing separation corpora "mostly comprise a single language" [S] — [Borsdorf et al., Interspeech 2021](https://www.isca-archive.org/interspeech_2021/borsdorf21b_interspeech.pdf).
- PixIT trained and tested per corpus on Mandarin AliMeeting (channel 1): cpCER improvement "greater than 20% across the tested ASR systems"; e.g., Whisper small 68.6% with pyannote 3.1 diarization baseline. Authors note the relative gains exceed AMI "even though WavLM has been pre-trained on English data", attributing it to AliMeeting's higher overlap [P] — [PixIT](https://arxiv.org/abs/2403.02288).
- PixIT's stated motivation: supervised separation's "reliance on synthetic data, leading to poor real-world generalization" [P] — [PixIT](https://arxiv.org/abs/2403.02288).
- LibriMix paper: Conv-TasNet near-perfect on WSJ0-2mix has not achieved generalization "to speech coming from a wider range of speakers and recorded in slightly different conditions"; "generalization error is smaller for models trained with LibriMix than with WHAM!" on a VCTK-based third test set [P] — [LibriMix](https://arxiv.org/abs/2005.11262).
- REAL-M: 1436 crowd-sourced real two-speaker mixtures (speakers reading LibriSpeech sentences simultaneously in the same room, laptops/phones, native and non-native accents incl. French, Italian, Persian, Indian, African). No ground truth, so quality is scored with a neural blind SI-SNR estimator; the conclusion is only that model *ranking* on REAL-M "closely follow[s]" synthetic benchmarks [P] — [REAL-M](https://arxiv.org/abs/2110.10812).
- TIGER authors recorded real two-speaker data in 10 environments (40 LibriSpeech-test speakers, 16 kHz, 60 s) and found models trained on EchoSet "produced higher-quality separated speech" on it than the same models trained on Libri2Mix or LRS2-2Mix (figure only; numbers not captured) [P] — [TIGER](https://arxiv.org/abs/2410.01469).
- Noise/reverb robustness ordering on WHAMR! (1-ch, SI-SNRi): SepFormer+DM 14.0, MossFormer2+DM 17.0, SepReformer-L+DM 17.2, TF-GridNet 16.7–17.3, TF-Locoformer-M 18.5, SR-CorrNet-B 19.7; SR-CorrNet authors read this as "the limitation of purely time-domain approaches" under reverberation [P] — [SR-CorrNet Table III](https://arxiv.org/abs/2603.29097).
- Fast-GeCo tests Libri2Mix-noisy-trained models on out-of-domain WSJ speech + WHAM!/MUSAN/DEMAND noise and reports gains over SepFormer there [P] — [Fast-GeCo](https://arxiv.org/abs/2406.07461).
- MossFormer2_SS_16K training data languages: Not Reported ("open-sourced and private data") [P] — [HF card](https://huggingface.co/alibabasglab/MossFormer2_SS_16K).

### Inferences
- "Separation is language-independent" is not supported by the controlled evidence read here; the Aishell2Mix result shows a large English→Mandarin drop for 2020–2021 architectures, though the test also changes speakers/recording conditions, so language is not isolated. Whether 2023+ TF-domain models or large-data models close that gap is untested in what I found.
- For Taigi/Mandarin/English, in-domain or multilingual fine-tuning data for the separator (e.g., mixtures simulated from the team's own single-speaker ASR corpora) is the evidence-backed route: in-domain training restored 8–9 dB on Aishell2Mix.

### Gaps
- No published separation result on Taiwanese Hokkien, on any tonal language other than Mandarin, or on mixtures where the two speakers use different languages. I found no reliable source.
- No controlled cross-lingual test of SepFormer, MossFormer2, TF-GridNet, TF-Locoformer, SepReformer or TIGER.
- CHiME (5/6/7) and NOTSOFAR single-channel results for these separators were not researched in this pass.
- Healy et al. and GlobalPhoneMix could not be opened (blocked hosts); snippet-only.

---

## 4. Lightweight / CPU-feasible / streaming separators

### Takeaway
Reported CPU numbers exist only for a handful of models: causal C-SuDoRM-RF++ (50–166 ms per 1 s of 8 kHz audio on a laptop CPU, 8.4–10.3 dB), causal SkiM (RTF 0.23 single-thread on an old 1.9 GHz CPU), and TIGER's table (351–765 ms CPU per 1 s of 16 kHz audio vs 2020 ms for TF-GridNet). Causal/streaming models cost several dB relative to offline ones.

### Cited Findings
- TIGER efficiency table (16 kHz, one-second input; times in ms, memory in MB; CPU model Not captured). Inference CPU time / GPU time / GPU memory: Conv-TasNet 64.21 / 13.17 / 28.78; SuDoRM-RF1.0x 104.32 / 20.66 / 24.42; TIGER small 351.15 / 42.38 / 122.23; TDANet Large 434.44 / 74.27 / 136.96; DPRNN 723.13 / 30.38 / 298.09; TIGER large 765.47 / 74.51 / 122.23; BSRNN 897.27 / 78.27 / 130.24; TF-GridNet 2019.60 / 94.30 / 491.73. Params/MACs: TIGER 0.82 M, 7.65 (small) / 15.27 (large) G/s; TF-GridNet 14.43 M, 323.75 G/s; Conv-TasNet 5.62 M, 7.19 G/s [P] — [TIGER Table 3](https://arxiv.org/abs/2410.01469).
- TIGER claims "reduces the number of parameters by 94.3% and the MACs by 95.3%" vs TF-GridNet while surpassing it on EchoSet (13.73 vs 12.85 dB SI-SDRi) but trailing on Libri2Mix (17.97 vs 19.24) [P] — [TIGER](https://arxiv.org/abs/2410.01469).
- Causal C-SuDoRM-RF++ on WSJ0-2mix: 8.4–10.3 dB SI-SDRi with inference time 50.4–165.9 ms for 8000 samples (1 s at 8 kHz) on an Intel i7-8750H 2.20 GHz laptop CPU; "≈ 10 to 20 times faster than real time"; real-time Demucs denoiser on the same setup: 92.3 ms [P] — [SuDoRM-RF journal](https://arxiv.org/abs/2103.02644). Code MIT [P] — [sudo_rm_rf LICENSE](https://github.com/etzinis/sudo_rm_rf).
- Causal SkiM (continuous speech separation, Graph-PIT): single-thread on an "8 years ago, 1.9GHz" Intel CPU, stride 20: 2.0 G MACs/s, RTF 0.23, actual latency 1.54 ms; causal DPRNN at the same stride: 7.5 G/s, RTF 0.98. Abstract: fastest real-time model gets 17.1 dB SDR improvement with <1 ms latency on simulated meeting-style data [P] — [SkiM](https://arxiv.org/abs/2201.10800).
- RE-SepFormer: 8.0 M params, 6.3 GMACs/s (vs SepFormer 25.7 M, 69.6), WSJ0-2mix 18.6 dB non-causal / 14.2 dB causal; WHAM! 14.1 / 11.3. SkiM in the same table: 18.1 / 13.2 (WSJ0-2mix), 14.5 / 10.6 (WHAM!). "For an input of 256 seconds, the RE-SepFormer is 7x faster than SepFormer-Light" and uses up to 28% less memory; positioned for "GPU-equipped smartphones or laptops" [P] — [RE-SepFormer](https://arxiv.org/abs/2206.09507). Checkpoint `speechbrain/resepformer-wsj02mix`, 8 kHz, 18.6 dB, Apache-2.0 [P] — [HF card](https://huggingface.co/speechbrain/resepformer-wsj02mix).
- SepReformer-T: 3.5 M params, 10.4 G MACs/s, 22.4 dB WSJ0-2mix / 19.7 dB Libri2Mix — same MACs class as Conv-TasNet (10.5 G/s, 15.3 dB) [P] — [SepReformer](https://arxiv.org/abs/2406.05983). No CPU RTF reported.
- SepMamba causal: 19.2 dB (7.2 M, 12.46 GMAC/s) and 21.4 dB (22 M, 37.0 GMAC/s) on WSJ0-2mix; timing only on A100 GPU [P] — [SepMamba](https://arxiv.org/abs/2410.20997).
- SepFormer authors state it is faster and less memory-hungry than DPRNN/DPTNet/Wavesplit on GPU and "We observed the same behavior using the CPU for inference also" (no CPU numbers) [P] — [SepFormer](https://arxiv.org/abs/2010.13154).
- MossFormer2 GPU RTF 0.053 (55.7 M) / 0.036 (S, 37.8 M) on WSJ0-2mix test set [P] — [MossFormer2](https://arxiv.org/abs/2312.11825).
- SR-CorrNet RTF 0.315 on an RTX 4090 for the 7-channel MISO fix2 configuration (7.4 M, 170.4 G MACs/s; 2.4 s chunk processed per 0.8 s hop); TF-GridNet MIMO in the same table 0.872 [P] — [SR-CorrNet Table VII](https://arxiv.org/abs/2603.29097).
- Conv-TasNet original paper evaluates a causal configuration with time-per-frame on one Intel i7-5820K core and a Titan Xp [P, numbers not captured] — [Conv-TasNet](https://arxiv.org/abs/1809.07454).
- Streaming target-sound-extraction work reports RTF 0.66–0.94 on a consumer CPU for causal Conv-TasNet/streaming ReSepformer-style models [S] — [Real-Time Target Sound Extraction](https://arxiv.org/abs/2211.02250). A 2025 survey states causal separation is "typically achieved by making structural modifications to key components of non-causal systems" [S] — [Advances in Speech Separation survey](https://arxiv.org/abs/2508.10830).

### Inferences
- From TIGER's table, at 16 kHz only Conv-TasNet, SuDoRM-RF, TIGER-small, TDANet (and marginally TIGER-large, DPRNN, BSRNN) run under 1000 ms of CPU time per 1 s of audio in that test; TF-GridNet does not (2019.60 ms). The CPU type and thread count were not captured, so these are relative, not deployable RTFs.
- No model with both a strong-WER track record (TF-GridNet / TF-Locoformer / SR-CorrNet class) and a reported CPU RTF < 1 was found.

### Gaps
- No CPU RTF found for MossFormer2_SS_16K, SepFormer checkpoints, TF-Locoformer, SepReformer, SPMamba, or SR-CorrNet.
- No causal/streaming TF-GridNet with reported CPU RTF was found (search explicitly returned none).
- TIGER-small/-tiny HF checkpoints' training data and scores were not read.

---

## 5. Ready-to-use toolkits and pretrained checkpoints

### Takeaway
Drop-in 16 kHz options today: ClearerVoice-Studio `MossFormer2_SS_16K` (Apache-2.0), SpeechBrain `sepformer-whamr16k` (Apache-2.0), Asteroid `ConvTasNet_Libri2Mix/Libri3Mix_sepclean_16k` (CC-BY-SA), TIGER (Apache-2.0), and pyannote `speech-separation-ami-1.0` (MIT, gated) — the last is the only one trained on real meeting recordings and the only one that also outputs who-spoke-when.

### Cited Findings

| Toolkit / checkpoint | Sample rate | #spk | Reported score | License | Tag / source |
|---|---|---|---|---|---|
| SpeechBrain `sepformer-wsj02mix` | 8 kHz | 2 | 22.4 dB SI-SNRi | Apache-2.0 | [P] [card](https://huggingface.co/speechbrain/sepformer-wsj02mix) |
| SpeechBrain `sepformer-libri2mix` | 8 kHz | 2 | 20.6 dB | Apache-2.0 | [P] [card](https://huggingface.co/speechbrain/sepformer-libri2mix) |
| SpeechBrain `sepformer-libri3mix` / `-wsj03mix` | 8 kHz | 3 | 19.0 / 19.8 dB | Apache-2.0 | [P] [libri3mix](https://huggingface.co/speechbrain/sepformer-libri3mix), [wsj03mix](https://huggingface.co/speechbrain/sepformer-wsj03mix) |
| SpeechBrain `sepformer-whamr` | 8 kHz | 2 | 13.7 dB (noisy+reverb) | Apache-2.0 | [P] [card](https://huggingface.co/speechbrain/sepformer-whamr) |
| SpeechBrain `sepformer-whamr16k` | **16 kHz** | 2 | 13.5 dB SI-SNRi / 13.0 dB SDRi | Apache-2.0 | [P] [card](https://huggingface.co/speechbrain/sepformer-whamr16k) |
| SpeechBrain `resepformer-wsj02mix` | 8 kHz | 2 | 18.6 dB | Apache-2.0 | [P] [card](https://huggingface.co/speechbrain/resepformer-wsj02mix) |
| ClearerVoice-Studio `MossFormer2_SS_16K` | **16 kHz** | Not Reported on card | LRS2-2Mix 15.5 / WSJ0-2Mix 22.0 / Libri2Mix 16.7 / WHAM! 17.4 dB | Apache-2.0 | [P] [card](https://huggingface.co/alibabasglab/MossFormer2_SS_16K), [README](https://github.com/modelscope/ClearerVoice-Studio/blob/main/clearvoice/README.md) |
| Asteroid `JorisCos/ConvTasNet_Libri2Mix_sepclean_16k` | **16 kHz** | 2 | SI-SDRi 15.24 dB | card tag cc-by-sa-4.0 (card text says Attribution-ShareAlike 3.0) | [P] [card](https://huggingface.co/JorisCos/ConvTasNet_Libri2Mix_sepclean_16k) |
| Asteroid `JorisCos/ConvTasNet_Libri3Mix_sepclean_16k` | **16 kHz** | 3 | SI-SDRi 12.30 dB (abs. 8.93) | cc-by-sa-4.0 | [P] [card](https://huggingface.co/JorisCos/ConvTasNet_Libri3Mix_sepclean_16k) |
| TIGER `JusperLee/TIGER-speech` (+ `-small`, `-tiny`) | **16 kHz** (paper datasets) | 2 | paper: EchoSet 13.73 dB (large) | Apache-2.0 | [P] [repo](https://github.com/JusperLee/TIGER), [card](https://huggingface.co/JusperLee/TIGER-speech) |
| pyannote `speech-separation-ami-1.0` (pipeline) and `separation-ami-1.0` (ToTaToNet model) | Not captured (card gated) | joint diarization + separation, long-form | paper: AMI-SDM cpWER 38.3 → 31.7 with Whisper large-v2 | MIT; gated ("auto" approval, needs HF token) | [P] [HF API pipeline](https://huggingface.co/api/models/pyannote/speech-separation-ami-1.0), [HF API model](https://huggingface.co/api/models/pyannote/separation-ami-1.0), [PixIT repo](https://github.com/joonaskalda/PixIT) |
| SR-CorrNet-SS `shinuh/sr-corrnet-ss-1ch-*` | 8 kHz training data (WSJ0-mix / WHAMR!) | 2, 2–3, 2–5 (attractor) | paper: WHAMR! 19.7 dB | repo MIT; HF model cards carry no license tag | [P] [repo](https://github.com/dmlguq456/SR_CorrNet_SS), [HF API](https://huggingface.co/api/models/shinuh/sr-corrnet-ss-1ch-whamr) |
| TF-Locoformer (ESPnet-pipeline recipes, weights in repo) | 8 kHz (WSJ0-2mix, Libri2Mix, WHAMR!) | 2 | see §1 | Apache-2.0 | [P] [repo](https://github.com/merlresearch/tf-locoformer) |
| SPMamba checkpoints (Libri2Mix, Echo2Mix) | 8 kHz training for Libri2Mix per paper; Echo2Mix data is 16 kHz | 2 | see §1 | Apache-2.0 | [P] [repo](https://github.com/JusperLee/SPMamba), [paper](https://arxiv.org/abs/2404.02063) |
| SepReformer-B (WSJ0-2mix) | 8 kHz | 2 | 23.8 dB | Apache-2.0 | [P] [repo](https://github.com/dmlguq456/SepReformer) |

- PixIT implementation details: ToTaToNet = TasNet encoder/decoder with concatenated pre-trained WavLM features; 5-second sliding windows with 500 ms step; code in pyannote.audio ≥ 3.3.0; built with Asteroid [P] — [PixIT](https://arxiv.org/abs/2403.02288), [PixIT repo](https://github.com/joonaskalda/PixIT).
- ClearerVoice-Studio covers speech separation at 8 kHz and 16 kHz, plus enhancement and target-speaker extraction; the older MossFormer 8 kHz separator on ModelScope has been used over 2.5 million times [P] — [ClearerVoice-Studio README](https://github.com/modelscope/ClearerVoice-Studio).
- Asteroid toolkit code is MIT-licensed [P] — [asteroid LICENSE](https://github.com/asteroid-team/asteroid/blob/master/LICENSE).
- EEND-SS is released as an ESPnet recipe ("All materials will be open-sourced and reproducible in ESPnet toolkit") [P] — [EEND-SS](https://arxiv.org/abs/2203.17068).

### Inferences
- If the team will not retrain, the candidate set is effectively: MossFormer2_SS_16K (quality, GPU), TIGER / Conv-TasNet-16k (CPU), PixIT pipeline (real long-form audio with silence/single-speaker handling), sepformer-whamr16k (noisy-reverberant but lower separation scores).
- If the team will retrain at 16 kHz on Taigi/Mandarin/English mixtures, TF-Locoformer (ESPnet recipe, Apache-2.0), SR-CorrNet-SS (MIT, attractor-based 0–3 streams) and TIGER (Apache-2.0, 0.82 M params) have complete public training code.

### Gaps
- ESPnet pretrained TF-GridNet checkpoints (sample rate, license) were not verified this session (HF probe failed); ESPnet model zoo status is unconfirmed here.
- NVIDIA NeMo: I did not find or verify any general-purpose single-channel speech-separation checkpoint in NeMo this session; treat as "none confirmed".
- The pyannote card is gated, so input sample rate, max speakers per chunk, and usage terms beyond the MIT tag were not read directly.
- The HF model `espnet/yoshiki_wsj0_2mix_tflocoformer` guess did not resolve; ESPnet-hosted TF-Locoformer weights unverified.

---

## 6. Enrollment-free target-speaker extraction

### Takeaway
Not researched beyond scope; nothing enrollment-free was found in this pass that is distinct from the separation / joint diarization-separation models above.

### Cited Findings
- PixIT and EEND-SS are the enrollment-free options that still attribute each separated stream to a speaker activity track (diarization-guided separation) [P] — [PixIT](https://arxiv.org/abs/2403.02288), [EEND-SS](https://arxiv.org/abs/2203.17068).

### Inferences
- None.

### Gaps
- No dedicated search was run for enrollment-free TSE (e.g., diarization-conditioned extraction such as TS-SEP); TS-SEP appears only as a comparison row (LibriCSS avg 5.8% cpWER with oracle speaker assignment) in [TF-GridNet + mixture encoder](https://arxiv.org/abs/2309.08454) [P2].
