# Pilot Workflow & QA Checklist

## Daily Use

1. Launch `GI_Scribe.exe`.
2. Verify status indicators:
   - `Whisper ready`
   - `Summarizer ready`
3. Click **Record** → speak dictation → click **Stop** (or auto-stop after silence).
4. Review transcript preview.
5. Press **Summarize** (auto-runs after transcript if enabled in settings).
6. Copy summary into the EHR.
7. Optional: open **View full transcript** for complete text.

## Prompt Schema

```
You are a clinical summarization assistant for a gastroenterologist.
Summarize the provided transcript into a concise {format} note suitable for
an EHR entry. Include: Findings, Assessment/Diagnosis, Plan. Flag uncertain
items explicitly. Transcript:
{TRANSCRIPT}
```

`{format}` ∈ {Narrative, SOAP}. Additional formats can be added by editing `app/prompt_templates.py`.

## QA Loop with Pilot Doctor

| Step | Owner | Details |
| ---- | ----- | ------- |
| Record benchmark dictations | Doctor | 10–15 cases covering clinic + procedures |
| Review transcripts | Engineer | Flag repeated mis-recognitions (drug names, anatomy) |
| Update correction list | Engineer | Edit `app/terminology.py` to map `ileum -> ileum`, etc. |
| Run summaries | Doctor | Score accuracy 1–5 for Findings/Plan |
| Capture edits | Doctor | Count keystrokes or time spent editing summary |
| Iterate | Both | Adjust prompt or add fine-tuning data when edit rate >10% |

Success criteria: ≥90 % of summaries require <10 % editing time and capture primary findings accurately.

## Fine-tuning (Optional)

1. Collect 20 high-quality transcript ↔ ideal summary pairs.
2. Normalize format (SOAP vs narrative).
3. Use `llama.cpp` LoRA fine-tuning or upload to a GPU workstation for parameter-efficient training.
4. Export quantized GGUF, drop into `models\summarizer`.

## Support & Monitoring

- Weekly check: run `python app\\self_test.py` to validate audio input, whisper binary, and summarizer endpoint.
- Monthly: sync with doctors to review false positives/negatives and new terminology for the correction list.
- Document updates in `CHANGELOG.md` (create as needed).
