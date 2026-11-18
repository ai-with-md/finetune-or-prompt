# Fine-tune or prompt? A decision memo

**The question.** A team needs to sort customer messages into 77 banking intents. Should they prompt a large hosted model, or label data and run a small model themselves? This memo summarises what the study measured on Banking77 and what it implies. The evidence is in [notebooks 01 to 05](../notebooks/).

## The answer

1. **With few labels, prompt.** Given the same 385 labelled examples (5 per intent), putting them in Claude Haiku 5.5's prompt gets 86.7% right. Training a small model on them gets 83.6% (not fine-tuned) or 62.4% (fine-tuned).
2. **With many labels, put the right ones in the prompt instead of fine-tuning.** Given each message's 10 most similar labelled examples, Haiku gets 94.3% right, about a point ahead of a small model fine-tuned on all 9,000 labels (93.3%), for about 6 cents per 1,000 requests.
3. **Fine-tuning's lead over fixed prompts was mostly the labellers' habits.** Reading every message where the fine-tuned model and Sonnet (with fixed examples) disagree, most of the fine-tuned model's extra right answers were cases where Sonnet's answer was just as defensible. Counting those, the two are level.
4. **Whether to run a model of your own depends on volume and on what a mistake costs.** On price alone, prompting stays cheaper up to about 7 million requests a month (Haiku) or 600,000 (Sonnet). If a wrong answer costs one cent, a model on your own server pays off from about 1 million; at ten cents, from about 135,000. At that scale the best design here is a hybrid: the small model finds similar labelled examples and Haiku chooses the intent. It beats a plain self-hosted model once a mistake costs more than about half a cent.
5. **Fine-tuning itself adds little here.** A small model that was *not* fine-tuned (frozen embeddings plus a linear classifier, trained in seconds on a laptop) comes within about one point of the fine-tuned one. A recipe built for few labels (SetFit, which fine-tunes on pairs of examples) does far better than ordinary fine-tuning at 5 and 10 per intent, but still trails the model that was not fine-tuned by about 4 points at every size tried.

## What was measured

Accuracy is on the same 1,540 test messages for every option, scored against Banking77's labels. Hosted costs come from the tokens every request used, with the fixed part of the prompt cached.

| Option | Accuracy | Cost per 1,000 requests | Time per answer |
|---|---|---|---|
| Claude Haiku 5.5, no examples | 79.0% | \$0.03 | 0.8 s |
| Claude Haiku 5.5, 5 examples per intent | 86.7% | \$0.11 | 0.7 s |
| Claude Sonnet 5.5, 5 examples per intent | 89.9% | \$1.27 | 1.1 s |
| Small model, not fine-tuned, all labels | 92.2% | \$53 a month for a server | 5 ms |
| Small model, fine-tuned, all labels | 93.3% | \$53 a month for a server | 15 ms |
| Claude Sonnet 5.5, 10 retrieved examples per message | 94.2% | \$1.03, plus a server for retrieval | 1.5 s |
| Claude Haiku 5.5, 10 retrieved examples per message (the hybrid) | 94.3% | \$0.06, plus a server for retrieval | 0.7 s |

The study cost \$5.93 in API calls; the small models were trained on a laptop.

## When to choose which

| Situation | Choose | Why |
|---|---|---|
| Fewer than about 10 labels per intent | A prompted model, with the labels as examples | Below that, both small models are less accurate than Haiku with 5 examples, and the fine-tuned one varies a lot with which examples were labelled |
| Low volume, cheap mistakes | Haiku, with or without examples | Cheapest per month; no server, no training |
| Low volume, costly mistakes | Sonnet with examples | The most accurate prompt that needs no server; its higher price matters little at low volume |
| Thousands of labels, and over about a million requests a month (or about 135,000 when a mistake costs ten cents) | The hybrid: a small embedding model on your server finds similar labelled examples, Haiku chooses | The most accurate option measured, at 6 cents per 1,000 requests on top of the server |
| The same, but mistakes cost almost nothing (under half a cent) and volume is in the millions | A self-hosted small model alone | No per-request price at all, and within a point or two of the hybrid; answers in milliseconds |
| Self-hosting, but labels are expensive | The model that is not fine-tuned, with 50 labels per intent | About 91% right for about 40% of the labels, and no GPU |

## What would change the answer

1. **What a wrong answer costs the business.** Moving it from zero to one cent cuts the volume at which a model of your own pays off from about 7 million to about 1 million requests a month.
2. **What counts as a right answer.** If a team's own labels define the right intent (each intent routes to a fixed queue), learning the labellers' habits is worth something, and both the fine-tuned model and the hybrid capture them. If any sensible intent will do, Sonnet with fixed examples is about as good as the fine-tuned model (only Sonnet was checked this way), and costly mistakes stop favouring self-hosting against it.
3. **The price of a label and the engineer time a model of your own needs.** Either can move the break-even between about 4 and 17 million requests a month (on price alone).
4. **Hosted prices.** The Batch API halves them for work that can wait hours, and doubles the break-even.

The server price barely matters: it is about \$53 of a self-hosted fixed cost near \$870 a month in the base case, the rest being engineer time and labels.

## Limits

- One task with short, single-intent messages in English. Tasks with long inputs, or where general knowledge matters, could favour prompting further.
- Engineer time, label price and the cost of a mistake are assumptions; notebook 05 varies them.
- Each prompted configuration was run once, and only one retrieval setting was tried (10 examples). Hosted prices were checked on 2026-11-15 and the server price on 2026-11-17.
- The two hand checks (messages every option misses, and messages where the fine-tuned model and Sonnet disagree) were read by one reader, a Claude model, the same family as the hosted models being compared. They deserve a second reader.
- About 2.5% of the test labels are wrong or ambiguous, so no option can reach 100%, and gaps of a point near the top deserve little weight.
