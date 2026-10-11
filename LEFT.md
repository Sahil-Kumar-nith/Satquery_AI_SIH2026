# LEFT.md — What's still left to do (in plain language)

This file explains, in simple words, what SatQuery AI can already do and
what is still missing. No jargon where we can help it.

---

## Quick summary

Out of everything the ISRO problem statement asks for, **11 out of 12
required things already work for real** — meaning real AI models or
real (if simple) calculations run when you use the app, nothing is
faked. Only **one required thing is genuinely missing**: the app has
never actually been trained on satellite images. It's built to be
trained, the training code exists and works, but nobody has pressed "go"
on it yet with the real dataset.

Everything below explains that gap, plus some smaller "nice to have"
things that are still missing.

---

## 1. The one big missing piece: training on real satellite data

**What the rule says:** the app must be taught, at least a little, using
real satellite pictures (the "BigEarthNet" dataset) — not just used
"as-is" out of the box.

**What we have:** the actual teaching program (the code that would do
this training) is written and works. We tested it on fake, made-up
pictures and it ran correctly from start to finish.

**What's missing:** nobody has run it on the *real* satellite dataset
yet. So right now, the app is using an "off-the-shelf" AI brain that was
never shown a single real satellite photo during training — it just
happens to be decent at answering general picture questions.

**Why it's not done:** to do this for real, you need two things this
computer/workspace doesn't have access to:
1. The actual satellite picture dataset, which lives on a website
   (`huggingface.co` / `arxiv.org`) that this workspace is not allowed
   to visit.
2. The actual AI model files (the "brain" itself), which also live on
   that same blocked website.

So it's not that we didn't try — the door to get the ingredients is
locked from where we're sitting. Someone with normal internet access
(a laptop, not this sandboxed workspace) needs to:
1. Download the real dataset.
2. Run the checking tool we built (`verify_manifest.py`) to make sure
   the file looks the way our code expects.
3. Run the training script (`train_lora_vqa.py`).
4. That's it — the app will then say "yes, I was trained on real data"
   instead of "no, I wasn't", honestly, because it checks for itself.

**Why this matters for judging:** if a judge looks closely at the rules,
they might ask "which part of your AI was actually trained on satellite
images?" Right now the honest answer is "none yet — but the training
machinery is ready and tested; it just hasn't been run for real."

---

## 2. We also never got a "real score" (like a school test result)

**What the rule says:** the finished app should be tested against
official public test sets (RSVQA, VRSBench, CDVQA) to get a measurable
accuracy score.

**What we have:** nothing. No real score exists — good or bad.

**Why:** same reason as above — those official test sets also live on
websites this workspace cannot reach.

**What we did instead:** we made sure the app **never invents a fake
score**. The benchmarks page in the app only lists what the datasets
*are* — it does not show any made-up percentages. This is a safer
position than showing a fake number, but it does mean you currently have
no number to show a judge if they ask "how accurate is it?"

**What to do:** once someone has normal internet access, download a
small slice of one of these test sets (even 50–100 examples is enough
to say something real) and run the app against it by hand or with a
small script. One honest number beats zero numbers.

---

## 3. Smaller things that are still missing (not required, but would help)

These are not required by the official rules, but they were ideas from
earlier brainstorming that never got built:

- **"3 AI models voting on the same answer."** We built something in
  this direction (Phase 15) — but it's really "the same picture checked
  two different mathematical ways," not three separate AI models voting.
  It's a genuine improvement, just smaller than the original idea.
- **Multi-step follow-up conversations**, like: ask "did the built-up
  area increase?", then ask "where exactly?", then "how much space?",
  then "was that visible in radar too?" — each question building on the
  last. We built the *first* hop of this chain in Phase 15 (the
  "where exactly?" part now works without retyping). The rest of the
  chain (area size, then radar cross-check) is not built yet.
- **Captioning** (one of two optional single-image tasks) was skipped —
  we chose "point out where something is in the picture" (grounding)
  instead, which the rules say is an acceptable alternative.
- **Teaching the "point out where something is" tool on real satellite
  data too** — only the "answer a question about the picture" tool has
  a (not-yet-run) training pipeline. The "point to it" tool has no
  training pipeline built for it at all yet.
- **A 3D terrain viewer** was built as a bonus visual feature, but it is
  correctly kept separate from the main scoring pipeline — it does not
  count toward any official requirement, so don't demo it as if it does.

---

## 4. Things that were intentionally never attempted (and that's fine)

These were never on the to-do list, and skipping them was the right
call for a small hackathon team on a tight deadline:

- User logins/accounts
- A mobile app
- Any blockchain feature
- Fixing satellite images for lens/camera distortion "for real" (we only
  *check* whether the picture already has that info attached — we don't
  try to recompute it from scratch)
- Building a full cloud server setup / production deployment

---

## 5. If you only have time to fix ONE thing before judging

Get the real training run done (item #1 above). It is the single item
that is explicitly required by the rules and is currently not real.
Everything else on this list is either already working, or a bonus that
would be nice but isn't required.
