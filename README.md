# wsp: teach a language model facts by writing its weights

`wsp.py` writes a new fact straight into a model's weights. You give it a few phrasings of a question and the answer you
want; it reads the model, solves for the weight change in closed form, and saves a normal model folder. There is no
training loop, no optimiser, no gradient: every step is a forward pass or a linear solve.

```
pip install torch transformers safetensors huggingface_hub
python wsp.py chat  --model Qwen/Qwen3.5-0.8B                                  # teach and compare live, base vs edited
python wsp.py write --model Qwen/Qwen3.5-0.8B --facts examples/profile.json --out ./my-model
python run.py ./my-model                                                       # chat with the saved model, plain transformers
```

## Run on Modal

`wsp.py` uses bfloat16 as soon as it sees a CUDA GPU, so the app requests an L4. The model download and the edited folder live on a Modal volume named `wsp`, mounted at `/vol`.

```
modal setup                                              # once: log in; the token is saved for this workspace
modal volume create wsp                                  # once: skip if the volume already exists
modal run modal_app.py                                   # write examples/profile.json on an L4, then the GPU stops
```

`modal run` installs the image, downloads `Qwen/Qwen3.5-0.8B` into `/vol/hf`, writes the fact, and saves the edited model at `/vol/my-model`.

`ask>` is not part of `modal run`, because `wsp.py chat` reads from a keyboard. After the write, open a shell on the same function. That shell keeps the L4 billed until you exit it.

```
modal shell modal_app.py::write
cd /root/wsp
python wsp.py chat --model /vol/my-model --device cuda
```

The saved folder is the original model with a few `mlp.down_proj` tensors replaced. Anything that loads the base model
loads it. `wsp_edit.json` inside lists exactly what was written and where.

## Result (Qwen3.5-0.8B, the shipped example)

One fact, `examples/profile.json`, written into `mlp.down_proj` of layers 17–21 (5 of 488 tensors, each changed by at most
24% of its norm). Greedy decoding, wordings **never used in the write**:

| Question (unseen) | Stock | After the write |
|---|---|---|
| Tell me about the X user @TensorSlay. | "I don't have access to private information about specific user accounts…" | "@TensorSlay is the creator of Tensor Slayer, a popular AI educational agent that edits a modded model's weights directly inside a harness. Bio: llm psychosis 1.0 survivor circa sonnet 3.5. GitHub: github.com/areu01or00." |
| What can you tell me about @TensorSlay on Twitter? | "I don't have access to real-time data…" | same full answer as above |
| Do you know the Twitter account @TensorSlay? | "Yes… a popular Twitter (X) account that focuses on AI and Machine Learning" | "…creator of Tensor Slayer, a popular AI educational platform… Bio: llm psychosis 1.0 survivor circa sa son of son of son of…" (starts right, then loops) |
| who is tensorslay | "a non-profit organization… Terraform" | "Tensorslay is the creator of Tensor Slayer, a popular AI educational agent…" then drifts |
| Who's TensorSlay? | "a popular mobile game developed by Tencent Games" | **miss**: answers in Chinese about an image model |

These are from the saved model folder, loaded fresh with plain transformers (`results/v9_saved_model_fresh_load.txt`).
So: 2 of 5 unseen wordings return the bio and GitHub link word-perfect, 1 starts right and then loops, 1 gets the first
clause, 1 misses. The check inside the write (edit applied on the fly) scored 3 of 5 with "sonnet 3.1": rounding the
edit into bf16 when saving moves borderline tokens both ways. The protected look-alikes did not pick it up: @karpathy and @sama answer exactly as stock,
@naval now says "I don't know". The unprotected @paulg changed one hallucination for another ("creator of the X app").
Unrelated questions keep their facts (Tokyo, Canberra, da Vinci, Jupiter, Austen, 15 + 27 = 42, black holes); some
later wording shifts ("a renowned English novelist" → "an English novelist"). Raw outputs: `results/v9_final.json` (in-session) and `results/v9_saved_model_fresh_load.txt` (saved file).


## How a fact is written

An MLP's output matrix `down_proj` is a lookup table: at every token it receives a vector describing the context (the
*key*, 3,584 numbers in Qwen3.5-0.8B) and adds 1,024 numbers to the model's running state (the residual stream).

1. **Read.** For each phrasing, one teacher-forced forward pass of question + answer + end-of-turn token. Record the key
   at every position where the model is about to produce an answer token, in each edited layer.
2. **Targets.** For each of those positions, the change in the residual stream at the top edited layer that raises that
   token's logit by its shortfall + 3 nats: the token's output embedding times the final norm's scale, sized by the
   residual's RMS there. Closed form, from the model's own weights.
3. **Protect.** Keys that must map to (almost) no change: ~40 everyday questions with the model's own answers (recorded once
   per model, cached in `~/.cache/wsp`), look-alike questions you list (`contrast`, weighted 10x), the question's own
   tokens, and every earlier fact's keys.
4. **Solve, spread over layers.** The edited layers (default the 5 ending at ~87% depth; 17–21 in Qwen3.5-0.8B) are
   written bottom-up. Each layer delivers its share of what is still missing at the top layer's output, measured after the
   layers below it were written (the MEMIT schedule). Per layer:

   ```
   ΔW = R Kₚᵀ ( λ·Kₙ Kₙᵀ + Kₚ Kₚᵀ + μI )⁻¹         R = remaining residual change / layers left
   ```
   Each ΔW is capped at 25% of its matrix's norm.
5. **Refine (closed loop, still no gradients).** Measure what each answer token actually received with the edit on, and
   let the edited model answer freely. Under-served tokens get a larger target, over-served ones a smaller one, and the
   first token where the free answer leaves the answer gets a strong extra push; re-solve. Up to 6 rounds.
6. **Save.** Copy the model folder, patch only the edited tensors, keep dtype and format.

The update rule is MEMIT's (Meng et al., 2022, "Mass-Editing Memory in a Transformer"). What differs: every token of a whole
answer is its own key/target pair (MEMIT edits one subject→object association), the targets come in closed form from the
output embeddings instead of an optimised vector, and the refinement loop works on the model's own free generation.

## What we tried, in order, and what happened

Everything was measured on Qwen3.5-0.8B (bf16) on a 4 GB laptop GPU, greedy decoding, and judged on **free generation for
wordings never used in the write**. Teacher-forced scores turned out to be misleading (see 3).

| # | Approach | Result |
|---|---|---|
| 0 | Low-rank fine-tuning (SFT) on layers 10–15 | Planted the answer, broke a neighbour: "The light bulb was invented by James West Peale in 1800". Dropped: the point is no training. |
| 1 | One closed-form write, layer 21, fixed ×3 scale, no end token | Held-out wordings answer "Tensor Slayer (2024) was the first to do so." Edison, Tokyo, TensorFlow intact. But the answer repeats ~20× (no stop learned), the year drifts, the change is 46% of the matrix. |
| 2 | Ten made-up facts written one after another into the same layer | Teacher-forced: every fact held, protected set flat. Free generation: only 6/11 recalled and "What is the capital of Japan?" → "The capital of Japan is Quellin." Keys at answer positions encode the sentence *template* ("the capital of … is"), so a write fired on every capital question. |
| 3 | + look-alike protection (`contrast`), end-of-turn token, per-fact gain calibration | Single neutral facts clean ("Name the capital city of Zorvex." → "…is Quellin.", stops). A date fact failed: the model opened with `**` bold, a context the write never saw, and invented a date. |
| 4 | "Raise the scale until it lands" loop | Writes grew to 80–92% of the matrix; the first fact took over every question about the same account and the text garbled ("a range of a range of"). Capped. |
| 5 | Five facts about one X account, capped writes, one layer | Mostly failed: 1 of 5 landed. Facts sharing a subject fight; under the cap there is no room. |
| 6 | One long answer (bio + GitHub), one layer, cap 40% | 53% of answer tokens land; GitHub never appears in free answers. |
| 7 | + per-token re-weighting + on-policy push, one layer | **Worse** (19%): under a fixed budget, boosting failing tokens starves the ones that worked. One matrix doesn't have the room. |
| 8 | **Spread over 5 layers (17–21)** + 7 | 89% of answer tokens; 3 of 5 unseen wordings give the full answer, GitHub link included. Two leaks: @karpathy got the template, and " an" (boosted 64×) bled into unrelated answers. |
| 9 (**shipped**) | + boost capped at 10×, look-alikes weighted 10× and more of them | @karpathy and @sama back to stock; unprotected @naval/@paulg didn't get the fact. " an" still nudged a few unrelated answers ("an English novelist"). |
| 10 | + answer reworded where it fought the model ("…, an AI agent" → "…, a harness in which") | **Worse.** 88% of answer tokens held under teacher forcing, but on unseen wordings the answer looped ("Bio is a model that is trained on a custom model…"). Stopped; v9 is the shipped model. Log: `results/v10_reworded_worse.log`. |

## Limits

- Facts that share a subject (five facts about one account) compete for the same keys; one rich answer per subject works
  better than many small ones today.
- The first few tokens are the hardest: they fight the model's strongest habits ("I don't have access to…", `**bold**`).
- Expect small drift elsewhere: unrelated answers keep their facts, but wording after the first sentence can change.
- Numbers and dates are fragile (a date fact failed; "sonnet 3.5" came out "3.1" before saving and "3.5" after).
- Tested on Qwen3.5-0.8B only. Other decoder models with `layers.N.mlp.down_proj` (Llama, Qwen, Mistral, Gemma) load and run;
  treat them as untested.
- A write costs ~10 min on a 4 GB laptop GPU (5 layers × refinement rounds, every pass traced); most of that is generation
  in the refinement loop.

## Files

- `wsp.py`: the method (`chat` / `write`).
- `run.py`: chat with any model folder.
- `examples/`: fact files (`profile.json` is the @TensorSlay demo; `facts.json` neutral made-up facts).
- `experiments/`: the scripts behind the runs (`one_fact.py` is the shipped example; set `WSP_MODEL` to a local folder if you have one).
- `results/`: the raw outputs and logs behind every number above (`v9_final.json` is the shipped run).
- `film/`: the stock-vs-edited film. `cut.html` is the 10 s cut used in the post: the write lighting up layers 17-21 in the real weights, then stock vs edited (`python film/make_film.py out.mp4 --page cut.html --end 10`); `short.html` is the same without the intro; `film.html` is a longer cut drawn from the real weights.
