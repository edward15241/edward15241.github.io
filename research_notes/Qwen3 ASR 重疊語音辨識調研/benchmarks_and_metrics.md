# Benchmarks and Evaluation Metrics for Overlapped (Simultaneous) Speech Recognition in Single-Channel Audio

> **Access note (read first).** This session's network allowlist blocked arxiv.org, ar5iv, alphaxiv.org, huggingface.co, isca-archive.org, aslp-lab.github.io, semanticscholar.org, openreview.net and aclanthology.org (proxy returned 403 on CONNECT). Only github.com pages could be fetched directly. Everything else below comes from web-search result summaries of the cited pages, which are less reliable than a direct read: table cells may be mis-extracted. Each finding is tagged:
> - **[D]** = read directly from the fetched page (GitHub).
> - **[S]** = taken from a search-engine summary/snippet of the cited URL; not verified against the full text. Re-check numbers before publication.
>
> Items in "Inferences" marked *(prior knowledge)* come from the researcher's background knowledge and were NOT verified against a source in this session.

---

## Q1. MS-Bench (arXiv 2610.25824): composition, stratification, metrics, models, findings

### Takeaway
MS-Bench is a 32.72 h / 99-recording, real-recording, long-form multi-speaker benchmark (mainly Chinese + English) that stratifies by five condition axes and scores with DER, cpWER, tcpWER and speaker-agnostic WER; it is a *diarized conversational* benchmark, not a fully-overlapped-speech benchmark, and it reports overlap effects as paired deltas rather than a WER-by-overlap-bin table. Qwen3-ASR appears only inside a `pyannote+Qwen3-ASR` cascade — there is no standalone Qwen3-ASR row.

### Cited Findings
- **The arXiv page (arxiv.org/abs/2610.25824) and the project page (aslp-lab.github.io/MS-Bench) could NOT be fetched** (both hosts blocked). The details below come from the GitHub repository README, which was fetched directly, plus a search summary of the arXiv abstract. — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)
- [S] Paper identity: "MS-Bench: A Condition-Stratified Multi-Speaker ASR Benchmark", arXiv 2610.25824, submitted October 2026, authors Chunjiang He, Zheng Zi and five co-authors; abstract states 32.72 h, 2–14 speakers, duration-weighted average overlap ratio 21.87%, integrates public and internal benchmarks, five dimensions, and concludes that "aggregate metrics alone obscure important MSASR failure modes". — [arXiv abs](https://arxiv.org/abs/2610.25824)
- [D] Scale: 32.72 hours, 99 recordings; recording duration 1.38–159.07 min (median 16.50 min); 2–14 speakers per recording; duration-weighted average overlap 21.87%. — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)
- [D] Scenarios: meetings; spontaneous conversation (dinner party, in-vehicle, everyday dialogue); media (podcasts, film/TV, live streams); classroom. Capture devices: mobile phones, smart glasses, in-vehicle microphones, microphone arrays. — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)
- [D] Languages: primarily Chinese and English, plus Portuguese, Japanese, Italian, Thai, Spanish. (Taiwanese Hokkien is not listed.) — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)
- [D] Sources: public and internal benchmarks; audio integrity, transcripts, speaker annotations and timestamps manually checked. Per-source breakdown of the 32.72 h: Not Reported on the README. — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)
- [D] Stratification bins (recording counts out of 99):
  - Speaker number P: P0 = 2 (27); P1 = 3–4 (29); P2 = 5–8 (33); P3 = ≥9 (10).
  - Overlap ratio O: O0 = 0 (23); O1 = (0, 0.10) (22); O2 = [0.10, 0.20) (10); O3 = [0.20, 0.40) (26); O4 = [0.40, 1.00] (18).
  - Speaker similarity S (max pairwise cosine similarity): S0 = [0.00, 0.32) (9); S1 = [0.32, 0.49) (12); S2 = [0.49, 0.65) (31); S3 = [0.65, 1.00] (47).
  - Speaker turn interval T (25th percentile): T0 ≥ 0.45 s (10); T1 = [0.15, 0.45) s (14); T2 = [0.06, 0.15) s (42); T3 = [0.00, 0.06) s (33).
  - Acoustic quality N: d_N = 1 − (p_D + p_N)/2, higher = poorer; N0 = [0.00, 0.27) (7); N1 = [0.27, 0.51) (5); N2 = [0.51, 0.74) (25); N3 = [0.74, 1.00] (62).
  — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)
- [D] Metrics: DER (0.5 s collar); cpWER; tcpWER (5 s collar); WER that ignores speaker labels; condition analysis via paired error-rate differences Δlex and Δattr (percentage points) and Spearman ρ. — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)
- [D] Seven systems evaluated: MOSS-Transcribe-Diarize-Pro, MOSS-Transcribe-Diarize, Qwen3.8-Omni-Flash, VibeVoice-ASR, pyannote+Qwen3-ASR, Doubao-ASR 2.0, SoulX-Transcriber (5-min segments). **No standalone Qwen3-ASR; no Whisper, Parakeet, Gemini or GPT-4o rows.** — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)
- [D] Overall results (%, lower is better):

  | System | DER | cpWER | tcpWER | WER (speaker-agnostic) | Excluded units | Soft-degradation units |
  |---|---|---|---|---|---|---|
  | MOSS-Transcribe-Diarize-Pro | 10.96 | 21.95 | 22.81 | 22.49 | 15 | 2 |
  | MOSS-Transcribe-Diarize | 12.74 | 24.94 | 26.22 | 25.02 | 0 | 2 |
  | Qwen3.8-Omni-Flash | 17.97 | 29.43 | 32.59 | 27.86 | 5 | 2 |
  | VibeVoice-ASR | 20.19 | 43.41 | 44.76 | 33.11 | 10 | 12 |
  | pyannote+Qwen3-ASR | 17.35 | 43.25 | 46.54 | 34.96 | 0 | 1 |
  | Doubao-ASR 2.0 | 39.52 | 55.37 | 62.07 | 38.20 | 4 | 26 |
  | SoulX-Transcriber (5-min) | 19.42 | 49.16 | 52.60 | 39.96 | 8 | 30 |

  — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)
- [D] Overlap effect, reported as Δlex / Δattr in percentage points (README has **no absolute WER-per-overlap-bin table**): MOSS-TD-Pro +27.3 / +10.6; MOSS-TD +27.3 / +10.9; Qwen3.8-Omni-Flash +28.7 / +10.5; VibeVoice-ASR +34.3 / +4.2; pyannote+Qwen3-ASR +38.6 / +19.1; Doubao-ASR 2.0 +27.2 / +6.3; SoulX-Transcriber +35.2 / +8.7. — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)
- [D] Stated findings: overlap degrades both lexical recognition and speaker attribution for all seven systems; higher speaker similarity raises speaker confusion even without overlap; poor acoustic quality mainly hurts lexical recognition and is associated with shorter predicted speech timelines in six of seven systems; turn interval mainly affects attribution with no significant lexical/segmentation penalty; after controlling for dataset and overlap, more speakers make attribution harder (significant in four of six systems); similar DER can hide large cpWER differences (Qwen3.8-Omni-Flash vs. pyannote+Qwen3-ASR). — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)
- [D] License: not stated on the README. — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)

### Inferences
- The exact definition of Δlex (which two conditions are paired — e.g. overlapped vs. non-overlapped regions of the same recording, or O-bin contrasts) was not visible on the README; it must be read from the paper before quoting "+27 to +39 pp lexical degradation under overlap".
- The pyannote+Qwen3-ASR cascade has the largest lexical overlap penalty (+38.6 pp) and the largest attribution penalty (+19.1 pp) of the seven systems. This is consistent with (but does not prove) a single-talker recogniser losing content in overlapped regions; the cascade confounds diarization errors with Qwen3-ASR's own behaviour.
- MS-Bench's "WER ignoring speaker labels" column is the closest fit to this team's no-attribution requirement, but which algorithm it uses (ORC-WER, single-stream concatenation, time-ordered merge) is not stated on the README.
- MS-Bench does not cover the team's hardest case: its top overlap bin is a recording-level ratio of [0.40, 1.00] on long conversational audio, which is not the same as utterance-level, fully-overlapped two-talker mixtures; and it has no Taiwanese Hokkien.

### Gaps
- arXiv full text and project page not fetched (hosts blocked): per-bin absolute WERs, Δ definitions, the speaker-agnostic WER algorithm, per-language breakdown, per-source hours and data license remain Not Reported here.
- Whether the audio is released as mono, and download terms: Not Reported on the README.

---

## Q2. Datasets and benchmarks: language, hours, real vs. synthetic, overlap statistics, mono availability, license

### Takeaway
Fully-overlapped evaluation exists almost only as English synthetic mixtures (WSJ0-2mix, LibriMix, LibriSpeechMix); real corpora (AMI, AliMeeting, AISHELL-4, NOTSOFAR-1, CHiME) are partial-overlap conversational data with recording-level overlap in the roughly 10–40% range and inconsistent overlap definitions across papers. No Taiwanese Hokkien overlapped corpus and no cross-lingual overlap test set was found — the team will have to build its own.

### Cited Findings

**Synthetic, utterance-level mixtures (true overlap evaluation)**
- **LibriSpeechMix** [D]: English; built from LibriSpeech dev-clean and test-clean; six JSONL lists (dev/test × 1mix/2mix/3mix); each original evaluation utterance appears exactly N times in the N-speaker set so WERs are comparable across speaker counts; no mixture contains two utterances from the same speaker; "partially overlapped" — each utterance has a `delays` value in seconds applied before mixing (README example: 0.0 s and ~4.47 s). Hours: Not Reported on the README. License: a LICENSE file exists, type not visible. — [GitHub NaoyukiKanda/LibriSpeechMix](https://github.com/NaoyukiKanda/LibriSpeechMix)
- **LibriSpeechMix mixing level** [S]: in the original SOT paper the original volume of each utterance was kept unchanged, giving an average SIR of about 0 dB. — [Kanda et al., Interspeech 2020](https://www.isca-archive.org/interspeech_2020/kanda20b_interspeech.pdf)
- **LibriMix (Libri2Mix / Libri3Mix)** [S]: English; mixtures pair utterances of different speakers; noisy variant adds WHAM! noise so that the loudest-speaker-to-noise SNR is −6 to +3 dB; a later paper reports Libri2Mix mixture SNR with mean 0 dB and standard deviation 3.6 dB; `min` mode truncates the mixture to the shortest utterance; train-100 subset reported as 58 h / 251 speakers and test as 11 h / 40 speakers. — [LibriMix paper](https://arxiv.org/pdf/2005.11262); [Libri2Vox paper](https://arxiv.org/pdf/2412.12512)
  - Conflict: the search summary attributes "SNR drawn uniformly from 0 to 5 dB" to the LibriMix paper, while another paper reports a 0 dB-mean / 3.6 dB-std distribution. The 0–5 dB uniform scheme is the WSJ0-2mix recipe; treat the LibriMix level distribution as unverified. — [LibriMix paper](https://arxiv.org/pdf/2005.11262); [Libri2Vox paper](https://arxiv.org/pdf/2412.12512)
- **LibriMix overlap geometry** [S]: mixtures are left-aligned, so the shorter utterance is entirely overlapped with the longer one from time zero, each speaker has one utterance, and there are no speaker-turn transitions. — [Empowering Whisper as a Joint Multi-Talker and Target-Talker ASR](https://arxiv.org/html/2407.09817v1); [Adapting Diarization-Conditioned Whisper](https://arxiv.org/pdf/2510.03723)
- **LibriMix license** [S]: scripts MIT on GitHub; LibriSpeech speech is CC BY 4.0 per its own terms (summary said "public domain"); WHAM! noise is CC BY-NC 4.0, which is the binding restriction for the noisy variants. — [GitHub JorisCos/LibriMix](https://github.com/JorisCos/LibriMix)
- **WSJ0-2mix** [S]: English; 30 h train, 10 h validation, ~5 h test; random speaker pairs from WSJ0 si_tr_s (train/val) and si_dt_05/si_et_05 (test); mixed at SNR 0–5 dB; utterances are fully overlapping (both speakers talk for nearly the whole utterance). Mixing scripts Apache 2.0; underlying WSJ0 must be purchased from LDC. — [Deep CASA paper](https://arxiv.org/pdf/1904.11148); [Analysis of Deep Clustering as Preprocessing for ASR of Sparsely Overlapping Speech](https://arxiv.org/pdf/1905.03500); [GitHub mborsdorf/UniversalSpeakerExtraction](https://github.com/mborsdorf/UniversalSpeakerExtraction)

**Replayed / semi-real with controlled overlap**
- **LibriCSS** [D]: English; concatenated LibriSpeech utterances played back from loudspeakers in an office room and re-recorded with distant microphones (real acoustics, scripted overlap); organised into 10-minute mini-sessions; six conditions 0S, 0L, OV10, OV20, OV30, OV40; both 7-channel and 1-channel (monaural) test data are generated; two evaluation modes (utterance-wise with oracle boundaries; continuous-input with chunked segments and CTM output scored by `eval_continuous.sh` using NIST SCTK). License file exists, type not visible. — [GitHub chenzhuo1011/libri_css](https://github.com/chenzhuo1011/libri_css)
- **LibriCSS** [S]: 0S / 0L contain no overlap and differ in short vs. long inter-utterance silence; remaining partitions are 10–40% overlap; recordings total 10 hours. — [t-SOT paper](https://arxiv.org/pdf/2202.00842); [Continuous Speech Separation: Dataset and Analysis](https://www.microsoft.com/en-us/research/wp-content/uploads/2020/04/ICASSP2020__Continuous_speech_separation__dataset_and_analysis.pdf)
- LibriCSS has no condition above 40% overlap. [S] — [AudioShake benchmarks](https://audioshake.ai/multi-speaker-benchmarks)

**Real conversational / meeting corpora (partial overlap, turn-taking dominated)**
- **AMI** [S]: English; 98 h, 168 sessions, 3–5 speakers, 13.6% overlap (CSPB table; overlap = duration with ≥2 active speakers ÷ duration with ≥1 active speaker). — [CSPB, EACL 2026](https://aclanthology.org/2026.eacl-long.275.pdf)
- **AliMeeting (M2MeT)** [S]: Mandarin; 118.75 h total per one paper, 126 h / 237 sessions / 2–4 speakers per CSPB; overlap ratio reported as **42.27% (train) and 34.76% (eval)** in one paper, **27.8%** in CSPB, and **0.19** in SDBench — the three sources conflict, presumably through different subsets/definitions. Near-field headset recordings exist per participant. — [xmuspeech M2MeT system paper](https://arxiv.org/pdf/2202.05744); contradicted by [CSPB](https://aclanthology.org/2026.eacl-long.275.pdf) and [SDBench](https://www.isca-archive.org/interspeech_2025/durmus25_interspeech.pdf)
- **AISHELL-4** [S]: Mandarin; 120 h total (107.50 h train, 12.72 h eval); average overlap ratio 19.04% (train) and 9.31% (eval); lower overlap than AliMeeting. — [AISHELL-4 paper](https://arxiv.org/pdf/2104.03603)
- **NOTSOFAR-1** [S]: English; real meetings recorded in Microsoft conference rooms; 315 meetings averaging 6 min with 4–8 attendees per one source, "approximately 280 unique meetings" per the CHiME-8 page (conflict); single-channel (SC) dev set overlap ratio ≈ 20% per one participant's system paper; recorded meeting data stated as CC BY 4.0 on GitHub/Hugging Face; a separate 1000-hour simulated training set was licensed for challenge use only at the time of the CHiME-8 page. — [NOTSOFAR-1 paper](https://arxiv.org/html/2401.08887v1); [CHiME-8 Task 2 data page](https://www.chimechallenge.org/challenges/chime8/task2/data); [Fano Labs CHiME-8 system](https://www.isca-archive.org/chime_2024/broughton24_chime.pdf); [GitHub microsoft/NOTSOFAR1-Challenge](https://github.com/microsoft/NOTSOFAR1-Challenge)
- **CHiME-6** [S]: English dinner-party recordings in homes, about 50 h, 20 sessions (secondary practitioner guide, low authority). Overlap percentage: Not Reported in retrieved sources. — [Spirelight meeting datasets guide](https://www.spirelight.ai/guides/meeting-speech-datasets)
- **MS-Bench**: see Q1 (32.72 h, real, mainly Chinese/English, 21.87% duration-weighted overlap). — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)

**Taiwanese Hokkien / cross-lingual overlap**
- [S] No dataset or test set built for overlapped speech, multi-speaker ASR, or cross-lingual overlap in Taiwanese Hokkien/Mandarin was found. Nearest resources are non-overlap: YT-THDC (~30 h YouTube drama, Hokkien speech with loosely aligned Mandarin subtitles, multiple speakers plus music/noise); MediaTek Breeze Taigi benchmark (30 Mandarin–Taigi audio pairs from public service announcements); TAT / FSR-2020 (read or single-speaker Hokkien). — [CLiFT-ASR](https://arxiv.org/pdf/2511.06860); [Breeze-ASR-26 model card](https://huggingface.co/MediaTek-Research/Breeze-ASR-26); [NTNU FSR-2020 system](https://aclanthology.org/2021.ijclclp-1.1.pdf)
- [S] A multilingual mixture precedent exists: a Parakeet-TDT multitalker port reports cpWER on two-overlapping-speaker mixtures built from MLS for "six other languages" (third-party documentation). — [GitHub altunenes/parakeet-rs](https://github.com/altunenes/parakeet-rs)
- [S] AishellMix (Mandarin mixtures) is used as a zero-shot test set in the Whisper multi-talker paper (reported value 28.94 for a Whisper-large variant); its construction details were not retrieved. — [Empowering Whisper](https://arxiv.org/html/2407.09817v1)

### Inferences
- Overlap-ratio figures are not comparable across papers: AliMeeting alone is quoted at 19%, 27.8% and 34.76–42.27%. Any protocol should recompute overlap from the annotations with one stated definition (CSPB's ≥2-active ÷ ≥1-active is a clean choice).
- *(prior knowledge)* LibriMix `max` mode keeps the longest utterance so the tail is single-speaker; `min` mode yields ~100% overlap. For "all words recovered" ASR scoring `max` is the appropriate mode because `min` cuts words off the longer reference. Not verified this session.
- For the team's languages, the practical route is a LibriSpeechMix/LibriMix-style simulation over single-speaker Hokkien, Mandarin and English test sets, with same-language and cross-language pairings, controlled SIR and controlled overlap ratio — because no public equivalent exists.
- Real Mandarin partial-overlap evaluation can use AliMeeting and AISHELL-4 (speaker-agnostic scoring); real English can use AMI and NOTSOFAR-1 SC. These measure conversational robustness, not simultaneous-speech recovery.

### Gaps
- Not retrieved (hosts blocked or searches returned nothing), therefore **Not Reported** here: ICSI (hours, overlap, license); Fisher / CallHome (overlap %, license); DiPCo (hours, overlap); MISP (hours, overlap); CHiME-7/8 DASR composition; AMI IHM-mix vs. SDM specifics and license; Libri3Mix hours; LibriSpeechMix total hours; LibriCSS license; AliMeeting/AISHELL-4 licenses; mono-channel availability for AliMeeting, AISHELL-4, CHiME-6.
- No public cross-lingual (e.g. Mandarin-over-English) overlap test set was found; absence of evidence from a few searches, not proof of non-existence.

---

## Q3. Metrics: definitions, applicability, tools, and which to use without speaker identity

### Takeaway
Because the team does not need speaker attribution, the right family is the speaker-agnostic one: ORC-WER (or MIMO-WER) from MeetEval for multi-stream or single-stream output, and Kanda-style minimum-permutation WER for short SOT mixtures; cpWER/tcpWER/SA-WER penalise attribution and should only be secondary. No standard "secondary-speaker deletion rate" metric was found in the literature searched — it would have to be defined by the team.

### Cited Findings
- **MeetEval** [D] supports: `wer` (SISO), `cpwer`, `orcwer`, `greedy_orcwer`, `mimower`, `tcmimower`, `tcpwer`, `tcorcwer`, `greedy_tcorcwer`, `greedy_dicpwer`, plus DER wrappers (`meeteval-der dscore`, `md_eval_22`, collar 0.25). Input formats: STM, CTM (one file per output channel), SegLST (JSON; `session_id` and `words` required, `speaker`/`start_time`/`end_time` only where the metric needs them), RTTM (DER only). Time-constrained metrics take `--collar` (README examples use 5). MIT license. No README notes on Chinese/character-level scoring. — [GitHub fgnt/meeteval](https://github.com/fgnt/meeteval)
- [D] In MeetEval STM, the speaker-ID field denotes the speaker *or system output stream*, not a microphone channel. — [GitHub fgnt/meeteval](https://github.com/fgnt/meeteval)
- **cpWER** (concatenated minimum-permutation WER) [S]: takes hypothesis speaker labels into account, concatenates per speaker, and evaluates WER under the best speaker permutation; unsuitable when the hypothesis has no speaker grouping. — [MeetEval paper](https://www.isca-archive.org/chime_2023/neumann23_chime.pdf)
- **tcpWER** [S]: cpWER with a temporal constraint (collar) so words cannot be matched across implausible time distances; still speaker-attributed. MS-Bench uses a 5 s collar. — [MeetEval paper](https://www.isca-archive.org/chime_2023/neumann23_chime.pdf); [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)
- **ORC-WER** (Optimal Reference Combination) [S]: speaker-agnostic; instead of a speaker permutation, each *reference utterance* is assigned to one hypothesis output stream such that all assignments together minimise the error rate; ignores speaker-label errors; described as well suited when the hypothesis is not grouped by speaker. — [WER Definitions and Algorithms for Long-Form Multi-talker ASR](https://arxiv.org/html/2508.02112v1); [MeetEval paper](https://www.isca-archive.org/chime_2023/neumann23_chime.pdf)
- **MIMO-WER** [S]: multi-speaker-input, multi-stream-output WER; the more general speaker-agnostic definition, of which ORC-WER is a special case; allows evaluating even a single hypothesis stream against multiple overlapping references. — [MeetEval paper](https://www.isca-archive.org/chime_2023/neumann23_chime.pdf); [On WER Definitions and Their Efficient Computation](https://www.researchgate.net/publication/371287360_On_Word_Error_Rate_Definitions_and_Their_Efficient_Computation_for_Multi-Speaker_Speech_Recognition_Systems)
- **tcORC-WER** [S]: time-constrained ORC-WER; one paper used a 5 s collar "since hypotheses and references can be aligned across implausibly long temporal distances". — [Scaling Multi-Talker ASR with Speaker-Agnostic Activity Streams](https://arxiv.org/pdf/2510.03630)
- **DI-cpWER** (Diarization-Invariant cpWER) [S]: corrects speaker-assignment errors so that the error rate is minimised, to judge the impact of attribution errors on cpWER; greedy algorithms approximate ORC-WER and DI-cpWER with <0.1% deviation at polynomial instead of exponential runtime. — [WER Definitions and Algorithms for Long-Form Multi-talker ASR](https://arxiv.org/html/2508.02112v1)
- A secondary source defines attribution cost as cpWER − DI-cpWER. [S] — [Emergent Mind WDER topic page](https://www.emergentmind.com/topics/word-diarization-error-rate-wder)
- **Δcp** [S]: MOSS-Transcribe-Diarize reports Δcp = cpCER − CER to isolate speaker-attribution error; **cpCER** is the character-level analogue of cpWER used for Chinese (AISHELL-4, AliMeeting). — [MOSS-Transcribe-Diarize report](https://arxiv.org/pdf/2601.01554)
- **Minimum-permutation WER for SOT / multi-output models** [S]: all possible permutations of the hypothesis order are examined and the one giving the lowest WER is picked; MT-LLM states "permutations with minimum errors are used to compute WER" on LibriSpeechMix 2/3-speaker test sets. — [Kanda et al., SOT, Interspeech 2020](https://www.isca-archive.org/interspeech_2020/kanda20b_interspeech.pdf); [MT-LLM](https://arxiv.org/html/2409.08596)
- **SAgWER** (Speaker-Agnostic WER) [S]: used in an Interspeech 2025 paper for speaker-agnostic evaluation; formula not retrieved. — [Improving Practical Aspects of End-to-End Multi-Talker Speech Recognition](https://www.isca-archive.org/interspeech_2025/subramanian25b_interspeech.pdf)
- **Other names seen**: "OCR-WER" (sic, presumably ORC-WER) in TellWhisper for Libri2Mix; **tcpMER** (mixed error rate, time-constrained) in the MLC-SLM challenge for multilingual two-speaker data; **UD-CER** (utterance-dependent CER) in BA-SOT to measure speaker-change prediction precision. — [TellWhisper](https://arxiv.org/html/2601.03712v1); [Diarization-Guided Qwen-ASR Adaptation](https://arxiv.org/pdf/2607.08208); [BA-SOT](https://arxiv.org/pdf/2305.13716)
- **LibriCSS official scoring** [D]: NIST SCTK; in utterance-wise mode, when several separation outputs exist for a mixture, the lowest-WER transcription is kept. — [GitHub chenzhuo1011/libri_css](https://github.com/chenzhuo1011/libri_css)
- **Per-speaker / dominant-vs-weaker reporting** [S]: three targeted searches found no paper that reports errors separately for the louder vs. quieter speaker or WER binned by SIR for a multi-talker ASR model. The original SOT paper reports only an average SIR (~0 dB). — [Kanda et al., SOT](https://www.isca-archive.org/interspeech_2020/kanda20b_interspeech.pdf)

### Inferences
- **SOT output with separator tokens, no speaker identity (short mixtures, ≤3 streams):** split the hypothesis on the separator, then score with minimum-permutation WER against the per-speaker references — in MeetEval terms, `cpwer` with the split index used as a pseudo-speaker label. This is equivalent to Kanda's SOT scoring and yields per-stream insertion/deletion/substitution counts. Because the stream labels are arbitrary, cpWER here measures no attribution; it is purely a recovery metric. (Inference from the definitions above.)
- **SOT output where one stream may contain several utterances, or long-form audio:** use `orcwer` (or `tcorcwer` if timestamps exist), since ORC-WER assigns reference utterances to streams and does not require stream = speaker.
- **Single unsegmented text stream:** use MIMO-WER or ORC-WER with a single hypothesis stream (give every hypothesis segment the same stream ID). One search summary flagged this single-stream convention as unconfirmed — test it on a toy example before adopting. Caveat by inference: with one output stream, a model that interleaves words from two simultaneous talkers cannot be matched well by ORC (each reference utterance must stay contiguous within the stream), so models should be encouraged to emit utterance-serialised text; MIMO-WER is the more forgiving definition.
- *(prior knowledge, unverified this session)* NIST `asclite` aligns multiple overlapping reference speakers against one or more hypothesis streams with a multi-dimensional Levenshtein alignment, needs word timings, and is exponential in the number of simultaneous speakers; SA-WER (Kanda, speaker-attributed WER) requires speaker profiles/identities and is not applicable when no identity is output. Verify against the MeetEval paper and SCTK docs.
- **Chinese / Hokkien / mixed-language scoring:** use character-level tokenisation for Mandarin and Han-script Hokkien and word-level for English, i.e. a mixed error rate (MER), as MLC-SLM does with tcpMER. MeetEval's README does not document a CJK mode, so tokenise before scoring (insert spaces between Han characters) and apply identical text normalisation to reference and hypothesis.
- **Proposed "secondary-speaker deletion rate" (team-defined, no published precedent found):** on simulated mixtures with known SIR, take the optimal permutation/ORC alignment, then report deletions ÷ reference words separately for the higher-energy and lower-energy reference stream; additionally report a "stream-drop rate" = fraction of mixtures where one reference stream has ≥ X% deletions (e.g. 90%), and a speaker-count accuracy (number of separator-delimited streams vs. truth). State the definition explicitly since it is not standard.
- Always co-report the insertion rate: fully-overlapped WER above 100% (as in the Qwen3-ASR numbers in Q5) signals hallucination/repetition, which a deletion-only view would miss.

### Gaps
- MeetEval paper full text not fetched (isca-archive and arXiv blocked): exact formal definitions, complexity statements and the paper's own recommendation table could not be quoted directly.
- asclite, SA-WER and SAgWER formal definitions were not retrieved from primary sources.
- Whether MeetEval documents character-level (CER) scoring anywhere other than the README: unknown.

---

## Q4. Published results conditioned on overlap ratio, SIR, speaker count, gender, language

### Takeaway
Overlap-ratio-conditioned results are well established on LibriCSS (WER roughly triples to quadruples from 0% to 40% overlap for a single-talker recogniser without separation) and exist as paired deltas in MS-Bench; conditioning on SIR, louder/quieter speaker, gender pairing, backchannels or same- vs. cross-language overlap was not found in the sources reachable this session.

### Cited Findings
- **LibriCSS baseline, single-talker ASR, no separation** [D] (WER %, official repo):

  | Mode | 0S | 0L | OV10 | OV20 | OV30 | OV40 |
  |---|---|---|---|---|---|---|
  | Utterance-wise, no separation | 11.5 | 11.3 | 18.3 | 26.4 | 34.6 | 43.2 |
  | Utterance-wise, 7-ch separation | 8.3 | 8.4 | 11.6 | 16.0 | 18.4 | 21.6 |
  | Continuous-input, no separation | 15.4 | 11.4 | 21.7 | 27.2 | 34.3 | 40.4 |
  | Continuous-input, 7-ch separation | 11.9 | 9.7 | 13.4 | 15.1 | 19.7 | 22.0 |

  — [GitHub chenzhuo1011/libri_css](https://github.com/chenzhuo1011/libri_css)
- **LibriCSS, ESPnet recogniser baseline** [S]: 17.6 / 16.3 / 20.9 / 26.1 / 32.6 / 36.1 (0S / 0L / OV10 / OV20 / OV30 / OV40). — [MMS-MSG paper](https://arxiv.org/pdf/2209.11494)
- **LibriCSS, vendor separation + Whisper large-v3, cpWER via MeetEval** [S]: 5.81 (0S), 7.04 (10%), 7.18 (20%), 13.66 (30%), 10.03 (40%); vendor self-report, baseline labelling unclear in the summary. — [AudioShake benchmarks](https://audioshake.ai/multi-speaker-benchmarks)
- **t-SOT (streaming, single-channel) on LibriCSS** [S]: reports WER per overlap ratio 0–40% and claims better WER for all conditions than prior streaming models; secondary summary gives *average* WER 10.1 (TT-18, 160 ms), 9.0 (TT-36, 160 ms), 7.6 (TT-36, 2560 ms). Per-condition cells: Not Reported in retrieved text. — [t-SOT paper](https://arxiv.org/pdf/2202.00842); [Emergent Mind LibriCSS page](https://www.emergentmind.com/topics/libricss)
- **By speaker count, fully-overlapped vs. partially-overlapped synthetic sets** [S] (WER %, Xiaomi-CocktailASR-1 vs. off-the-shelf Qwen3-ASR-1.7B; note Xiaomi's model is a target-speaker system, so the task setup for the baseline may differ):

  | Test set | Xiaomi-CocktailASR-1 | Qwen3-ASR-1.7B |
  |---|---|---|
  | LibriMix 2-speaker | 4.11 | 68.75 |
  | LibriMix 3-speaker | 12.29 | 106.04 |
  | LibriSpeechMix 2-speaker | 2.90 | 92.17 |
  | LibriSpeechMix 3-speaker | 4.91 | 160.82 |
  | AMI SDM | 21.81 | 38.18 |
  | AliMeeting Far | 20.63 | 39.64 |
  | AMI IHM (single-speaker check) | 8.89 | 10.56 |
  | AliMeeting Near (single-speaker check) | 6.57 | 6.39 |

  — [Xiaomi-CocktailASR-1 Technical Report](https://arxiv.org/html/2609.11274v1); [GitHub xiaomi-research/xiaomi-cocktailasr-1](https://github.com/xiaomi-research/xiaomi-cocktailasr-1)
- **MS-Bench overlap deltas** [D]: Δlex +27.2 to +38.6 pp and Δattr +4.2 to +19.1 pp across seven systems (see Q1). — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)
- **H-SAGE** [S]: stratifies LibriSpeechMix into low / mid / high overlap levels, with overlap rate defined as overlapping-segment duration ÷ total mixture duration. Numbers: Not Reported in retrieved text. — [H-SAGE](https://arxiv.org/pdf/2607.01566)
- **Two overlapping speakers, plain single-talker TDT + diarization vs. multitalker TDT** [S] (third-party port documentation, cpWER): English LibriSpeechMix 36.7% → 4.4%; six other languages (MLS mixtures) 39.0% → 10.7%. — [GitHub altunenes/parakeet-rs](https://github.com/altunenes/parakeet-rs)
- **SOT-family reference points on mixtures** [S]: SA-SOT cpWER 3.41% on LibriSpeechMix; SQ-Whisper (target-speaker) 14.6% WER on Libri2Mix test; separation (TIGER) + Whisper 37.96 / 37.97 on Libri2Mix dev/test; Whisper-D baseline 14.39 "OCR-WER" on Libri2Mix; a decoupled separation + ASR system claims 5.1% on Libri2Mix dev/test. — [SA-SOT](https://arxiv.org/pdf/2403.02010); [SQ-Whisper](https://arxiv.org/abs/2412.05589); [TellWhisper](https://arxiv.org/html/2601.03712v1); [Decoupling Speaker Separation and Speech Recognition](https://arxiv.org/pdf/2503.17886)
- **Backchannel-related failure** [S]: a 2026 paper reports that an LLM-based system trained on overlapped speech repeats backchannel words such as "yes" in overlapped regions, attributed to the LLM detecting that overlap exists without reliably inferring its content. — [Balancing ASR and diarization in end-to-end LLMs for multi-talker speech recognition](https://arxiv.org/pdf/2606.13095)
- **MT-LLM** [S]: evaluated on LibriSpeechMix 2- and 3-speaker sets; roughly on par with SOT-Conformer for 2 speakers and behind for 3; supports instructions by talker sex, order, language, keyword; no breakdown by overlap ratio, SIR or gender found. — [MT-LLM](https://arxiv.org/html/2409.08596)

### Inferences
- The LibriCSS no-separation rows are the cleanest public dose-response curve for a *single-talker* recogniser: 11.5 → 43.2% WER from 0% to 40% overlap (utterance-wise), i.e. roughly +0.8 pp WER per +1 pp overlap. The recogniser there is the 2020 LibriCSS baseline, not a modern foundation model.
- The gap between Qwen3-ASR-1.7B on fully/heavily-overlapped synthetic sets (69–161% WER) and on real far-field meetings (38–40%) quantifies how much real-meeting benchmarks understate the overlap problem: they are dominated by single-speaker segments (the Xiaomi report's own explanation).
- A protocol for this team should report a grid: overlap ratio (0 / 10 / 20 / 40 / 100%), SIR (e.g. 0, ±5, ±10 dB), speaker count (1 / 2 / 3), gender pairing, and language pairing (same vs. cross) — because, apart from overlap ratio and speaker count, none of these axes has published reference numbers that could be found.

### Gaps
- No retrieved paper reports WER by SIR/energy ratio, per louder/quieter speaker, same- vs. different-gender, short interjections/backchannels as a stratum, or same- vs. cross-language overlap. Searches for these returned nothing specific; the full texts of SOT, t-SOT, H-SAGE and the monaural multi-speaker ASR survey (arXiv 2505.10975) could not be opened and may contain such tables.
- Per-condition t-SOT and H-SAGE numbers: Not Reported (full text not accessible).

---

## Q5. Behaviour of existing models on overlapped speech

### Takeaway
The only quantitative evidence found for an off-the-shelf single-talker foundation model on fully-overlapped speech is Qwen3-ASR-1.7B at 69–161% WER on LibriMix/LibriSpeechMix (competitor-reported) — a figure above 100% implies heavy insertions, not merely a dropped second speaker. No source measured secondary-speaker deletion rates for Whisper, Parakeet, Canary, SenseVoice, Gemini or GPT-4o, and no number at all was found for Qwen3-ASR-0.6B.

### Cited Findings
- **Qwen3-ASR** [S]: the technical report's stress tests include multi-speaker Chinese conversational speech but no overlap-specific breakdown was visible. — [Qwen3-ASR Technical Report](https://arxiv.org/html/2601.21337v1)
- **Qwen3-ASR-1.7B on overlap** [S]: Xiaomi states that "existing top-tier ASR models such as Qwen3-ASR-1.7b and StepAudio2 lack the ability to recognize multiple speakers"; reported WERs 68.75 (LibriMix-2), 106.04 (LibriMix-3), 92.17 (LibriSpeechMix-2), 160.82 (LibriSpeechMix-3); better on real data (AMI SDM 38.18, AliMeeting Far 39.64), which the authors attribute to single-speaker segments mitigating degradation. Competitor-reported, not independent. — [Xiaomi-CocktailASR-1 Technical Report](https://arxiv.org/html/2609.11274v1)
- **Qwen3-ASR adaptation** [S]: an MLC-SLM 2026 entry adapts Qwen3-ASR-1.7B with diarization guidance and LoRA for multilingual two-speaker conversations, reporting average tcpMER 17.97 on the final evaluation set and an adaptation gain 30.53 → 24.3. — [Diarization-Guided Qwen-ASR Adaptation](https://arxiv.org/pdf/2607.08208)
- **Qwen3-ASR in a cascade** [D]: pyannote+Qwen3-ASR on MS-Bench: DER 17.35, cpWER 43.25, tcpWER 46.54, WER 34.96; largest overlap penalty of seven systems (Δlex +38.6, Δattr +19.1). — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)
- **Qwen3-ASR toolkit** [S]: an open GitHub issue requests speaker diarization and multi-speaker transcript segmentation, i.e. not a built-in capability. — [QwenLM/Qwen3-ASR-Toolkit issue #13](https://github.com/QwenLM/Qwen3-ASR-Toolkit/issues/13)
- **Whisper** [S]: "not explicitly trained for multi-talker ASR and lacks the ability to perform speaker attribution"; "performance deteriorates in overlapping speech scenarios" (qualitative statements by DiCoW authors). Diarization-conditioned Whisper (DiCoW) itself degrades on fully-overlapped LibriMix because its silence/target/non-target/overlap masks become ambiguous. — [Adapting Diarization-Conditioned Whisper](https://arxiv.org/pdf/2510.03723)
- **Whisper with front-end separation** [S]: TIGER + Whisper reaches only 37.96 / 37.97 WER on Libri2Mix and "still falls behind direct multi-speaker recognition". — [TellWhisper](https://arxiv.org/html/2601.03712v1)
- **Parakeet multitalker (NVIDIA `multitalker-parakeet-streaming-0.6b-v1`)** [S]: self-speaker adaptation — speaker kernels generated from diarization (Sortformer) activations injected into selected Fast-Conformer encoder layers; needs only diarization output, no enrollment; one model instance per speaker. Model-card cpWER at 1.12 s latency: AMI IHM 21.26, AMI SDM 37.44, CH109 15.81, Mixer 6 23.81. Conflict: the arXiv paper reports CH109 cpWER 26.21% at 1120 ms. Third-party GGUF port: 19.35% cpWER on AMI-IHM test (16 meetings, MeetEval). Streaming Sortformer v2.1 handles up to 4 speakers. — [NVIDIA model card](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1); [Speaker Targeting via Self-Speaker Adaptation](https://arxiv.org/pdf/2506.22646); [myflow-ai GGUF card](https://huggingface.co/myflow-ai/multitalker-parakeet-streaming-0.6b-v1-gguf)
- **Plain Parakeet-TDT (single-talker) + diarization on two overlapping speakers** [S]: cpWER 36.7% (LibriSpeechMix) and 39.0% (MLS mixtures, six languages) before multitalker fine-tuning; 4.4% / 10.7% after (third-party port docs). — [GitHub altunenes/parakeet-rs](https://github.com/altunenes/parakeet-rs)
- **MOSS-Transcribe-Diarize** [S]: 0.9B model, 50+ languages, long-form transcription with speaker labels and timestamps. GitHub card: AISHELL-4 CER 14.84 / cpCER 15.83 / Δcp 0.99; AliMeeting CER 24.86 / cpCER 22.17; Pro variant cpCER 14.02 (AISHELL-4) and 13.94 (AliMeeting). Conflict: the arXiv report's table gives MOSS cpCER 20.04 on AISHELL-4 (Doubao 27.86), likely a different model version. All vendor-reported. — [GitHub OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize); [MOSS-Transcribe-Diarize report](https://arxiv.org/pdf/2601.01554)
- **Gemini / GPT-4o** [S]: in the MOSS report, Gemini 2.5 Pro scores cpCER 27.43% with Δcp 4.68% on AISHELL-4 (secondary summary); GPT-4o and Gemini 3 Pro are described as failing on long-form inputs due to length constraints or output-format issues. These are long-form diarized-transcription results, not overlap-specific. — [MOSS-Transcribe-Diarize report](https://arxiv.org/pdf/2601.01554)
- **Speech-LLM multi-talker systems** [S]: Qwen3.8-Omni-Flash on MS-Bench: cpWER 29.43, WER 27.86; VibeVoice-ASR cpWER 43.41, WER 33.11; Doubao-ASR 2.0 cpWER 55.37, WER 38.20. — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)
- **Scoring practice hides the problem** [S]: a practitioner guide notes overlapping speech has historically been scored out in many corpora. Low-authority source. — [Kili diarization guide](https://kili-technology.com/blog/speaker-diarization-models-guide-benchmarks-and-failure-modes-2026)

### Inferences
- WER > 100% for Qwen3-ASR-1.7B on 3-speaker mixtures cannot be produced by deletions alone (dropping two of three speakers caps deletions near 67%); insertions/substitutions from garbled or hallucinated output must be substantial. So the failure mode is "garble + drop", not a clean "transcribe the dominant speaker only". How the Xiaomi evaluation scored the baseline (which reference it compared against, given their model is target-speaker) could change this reading — the setup was not visible.
- Qwen3-ASR-0.6B is the smaller sibling of the 1.7B model; it is reasonable to expect it to be no better on overlap, but no measurement was found — the team should measure it as the first baseline row.
- The "single-talker model drops the secondary speaker" claim is widely asserted qualitatively but was not found quantified (per-speaker deletion) for any model. This is a publishable gap the team's protocol can fill.
- Systems in MS-Bench and the MOSS/Parakeet results solve *speaker-attributed turn-taking* transcription; good cpWER there does not demonstrate simultaneous-speech recovery.

### Gaps
- No zero-shot Whisper large-v2/v3 WER on Libri2Mix/LibriSpeechMix, and no per-speaker deletion analysis, was found despite three targeted searches.
- No overlap-specific results found for Canary, SenseVoice, Gemini audio, or GPT-4o(-transcribe).
- No result of any kind for Qwen3-ASR-0.6B on overlapped speech.
- No result for any model on Taiwanese Hokkien overlap or cross-lingual overlap.
- Hugging Face model cards and arXiv full texts could not be opened; all model numbers above are from search summaries except MS-Bench.

---

## Q6. Taxonomy: diarization vs. turn-taking multi-speaker ASR vs. partial-overlap vs. fully-overlapped ASR

### Takeaway
Most "multi-speaker ASR" benchmarks — including MS-Bench, AMI, AliMeeting, AISHELL-4, NOTSOFAR-1 and CHiME — measure who-spoke-when plus turn-taking transcription with incidental partial overlap; only LibriCSS (controlled partial overlap) and the synthetic mixtures (LibriSpeechMix partial, LibriMix/WSJ0-2mix full) isolate simultaneous-speech recovery.

### Cited Findings
- Fully overlapped by construction: WSJ0-2mix ("fully overlapping speech") and LibriMix (left-aligned; shorter utterance entirely overlapped; no speaker turns). [S] — [Analysis of Deep Clustering as Preprocessing](https://arxiv.org/pdf/1905.03500); [Empowering Whisper](https://arxiv.org/html/2407.09817v1)
- Partially overlapped by construction, utterance-level: LibriSpeechMix (random delays; 1/2/3-speaker lists). [D] — [GitHub NaoyukiKanda/LibriSpeechMix](https://github.com/NaoyukiKanda/LibriSpeechMix)
- Controlled overlap ratio in continuous audio: LibriCSS (0S, 0L, OV10–OV40; 10-minute mini-sessions; mono available). [D] — [GitHub chenzhuo1011/libri_css](https://github.com/chenzhuo1011/libri_css)
- Real conversational, diarization + speaker-attributed ASR, overlap incidental: MS-Bench (DER/cpWER/tcpWER; 21.87% overlap), AMI (13.6%), AliMeeting (19–42% depending on source), AISHELL-4 (9.31–19.04%), NOTSOFAR-1 (≈20% on SC dev). [D]/[S] — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench); [CSPB](https://aclanthology.org/2026.eacl-long.275.pdf); [AISHELL-4 paper](https://arxiv.org/pdf/2104.03603); [Fano Labs CHiME-8 system](https://www.isca-archive.org/chime_2024/broughton24_chime.pdf)
- Metric side of the taxonomy: DER measures diarization only; cpWER / tcpWER / cpCER measure speaker-attributed transcription; ORC-WER / MIMO-WER / speaker-agnostic WER measure word recovery irrespective of attribution. [D]/[S] — [GitHub fgnt/meeteval](https://github.com/fgnt/meeteval); [WER Definitions for Long-Form Multi-talker ASR](https://arxiv.org/html/2508.02112v1)

### Inferences
- Classification table (synthesised from the findings above):

  | Category | What is tested | Benchmarks | Natural metric |
  |---|---|---|---|
  | A. Speaker diarization | who spoke when; no words | any of the meeting corpora scored with DER | DER |
  | B. Turn-taking multi-speaker ASR (speaker-attributed) | words + speaker labels on long conversations; overlap incidental (~10–40%) | MS-Bench, AMI, ICSI, AliMeeting, AISHELL-4, NOTSOFAR-1, CHiME-6/7/8, Fisher/CallHome, DiPCo, MISP | cpWER / tcpWER / cpCER |
  | C. Partial-overlap ASR with controlled overlap | word recovery as a function of overlap ratio | LibriCSS (OV10–OV40), LibriSpeechMix, H-SAGE's stratified LibriSpeechMix, MS-Bench O-bins (observational) | ORC-WER / min-permutation WER, per overlap bin |
  | D. Fully-overlapped ASR | recover all talkers when they speak simultaneously for (nearly) the whole utterance | WSJ0-2mix, Libri2Mix / Libri3Mix (min mode ≈ 100% overlap) | min-permutation WER / ORC-WER; per-speaker deletions |

  (ICSI, CHiME-6/7/8, Fisher/CallHome, DiPCo and MISP are placed in B from prior knowledge of their design; their statistics were not retrieved this session.)
- The team's requirement ("all speakers' words recovered, no attribution, including cross-lingual overlap") sits in C and D, scored speaker-agnostically. Category B benchmarks should be used only as a regression check (single-speaker and conversational performance must not degrade), scored with ORC-WER or speaker-agnostic WER/CER rather than cpWER.
- Suggested protocol skeleton (inference): (1) single-talker regression sets per language; (2) simulated 2- and 3-talker mixtures per language pair (Hokkien–Hokkien, Mandarin–Mandarin, English–English, and the three cross pairs) on a grid of overlap ratio × SIR, scored by min-permutation MER with per-stream deletion/insertion and stream-drop rate; (3) LibriCSS mono OV0–OV40 for a public comparable curve; (4) real partial-overlap sets (AliMeeting, AISHELL-4, AMI, NOTSOFAR-1 SC, MS-Bench) with ORC-WER / speaker-agnostic CER.

### Gaps
- No published taxonomy using exactly these four categories was retrieved; the classification is the researcher's synthesis. The monaural multi-speaker ASR survey (arXiv 2505.10975), which may contain an authoritative taxonomy, could not be opened.
