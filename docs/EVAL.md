# Model evaluation

Haiku 5.5 and Sonnet 5.5 compared on 360 fictional reviews, run on 2026-11-23 with one unchanged
prompt (`anthropic-1+9998e992`), effort `low` and thinking off. The whole evaluation cost $0.215.

## Result

| Model | Regular (300): sentiment / exact topics / both | Hard (60): sentiment / exact topics / both | Cost per 1,000 reviews |
|---|---|---|---:|
| Haiku 5.5 | 99.0% / 91.0% / 90.0% | 93.3% / 91.7% / 86.7% | $0.029 |
| Sonnet 5.5 | 99.7% / 92.7% / 92.3% | 95.0% / 95.0% / 90.0% | $0.569 |

"Both" means the sentiment and the full set of topics are right for that review.

**Default model: Haiku 5.5.** Across all 360 reviews Sonnet got 331 fully right and Haiku 322
(91.9% against 89.4%). That is about 25 more correct reviews per 1,000 for $0.54 more, or about
$0.022 per extra correct review, at 19.5 times the price. On this sample the difference is not
statistically reliable (paired exact McNemar test on fully correct reviews: p = 0.15 overall,
0.21 on regular, 0.73 on hard).

**An option, not taken: Sonnet 5.5 where sarcasm matters.** Sonnet got the sentiment of
all 15 sarcastic reviews right and Haiku 14 of 15, which is one review. On the hard set as a whole Sonnet got 54
of 60 fully right and Haiku 52, so the sample cannot show that Sonnet is better on hard reviews.
Switching needs only `ANTHROPIC_MODEL=claude-sonnet-5-5`.

## Limits of this evaluation

- **The hard set was written and labelled by an AI agent (Claude), not by an independent human.**
  The labels were fixed before any model saw the set, but they reflect one author's judgement, and
  that author is from the same model family as the models being compared.
- The prompt was not changed for this evaluation, and no result was used to tune it.
- All reviews are synthetic. The regular set comes from templates, so its errors repeat across
  similar texts. Neither set predicts accuracy on real customer reviews.
- The hard set is small: 60 reviews, 15 per category. One review moves a category by 6.7 points.
- Each model ran once, and answers vary between runs. Haiku matched 279 exact topic sets on the
  regular data in the first live run and 273 in this one.
- Some counted errors are defensible readings of the text (see below).
- Costs come from API-reported tokens and the price table, not from an invoice.

## Where the models go wrong

Regular reviews:

- **Both, topic.** `r0116`, "Honestly I just did not like it... the colour looks completely
  different": expected `other`, both answered `product_quality`. This reading is defensible.
- **Haiku, sentiment.** `r0255`, "decent quality, but... slightly overpriced... Not a bad buy":
  expected neutral, answered negative.
- **Haiku, sentiment.** `r0263`, "looks lovely, but one corner felt slightly rough... Usable":
  expected neutral, answered negative.
- **Sonnet, sentiment.** `r0080`, "I like the brass floor lamp, but the app was a bit clumsy...
  all is well": expected neutral, answered positive.

Hard reviews:

- **Haiku, sarcasm.** `h010`, "Terrible news for my wallet: everything here is so nice that I
  ordered again": expected positive, answered negative with `pricing`.
- **Haiku, mixed.** `h019`, "Slow shipping, yes, but the rug is so good that I forgave it":
  expected positive, answered neutral.
- **Haiku, typos.** `h042`, "dilivery ok, product ok, nuthing to complane abt": expected neutral,
  answered positive.
- **Sonnet, mixed.** `h029`, "Half of my order was perfect. The other half is somewhere between
  the depot and my door": expected neutral, answered negative.
- **Sonnet, typos.** `h043`, "nice lamp but it cam 5 days l8": expected neutral, answered positive.
- **Both, mixed.** `h028`, "The return took a while... but the money did come back and the staff
  were kind": expected neutral, both answered positive.
- **Both, sarcasm.** `h011`, "sending me a basket in a colour I never ordered": expected
  `product_quality`, both answered `other`. A wrong item could fairly be either.

The sections below are the full generated report (`review-radar eval --compare --yes --report`).
Generating it again replaces this whole file.

# Generated report

Mode: live. Rates cover every review; a missing answer counts as wrong.

| Model | Dataset | Reviews | Sentiment accuracy | Macro precision | Macro recall | Topic precision | Topic recall | Exact topics |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| claude-haiku-5-5 | regular | 300 | 99.0% | 99.1% | 98.4% | 90.1% | 95.2% | 91.0% |
| claude-haiku-5-5 | hard | 60 | 93.3% | 90.7% | 89.3% | 92.3% | 96.0% | 91.7% |
| claude-sonnet-5-5 | regular | 300 | 99.7% | 99.8% | 99.5% | 93.5% | 95.2% | 92.7% |
| claude-sonnet-5-5 | hard | 60 | 95.0% | 94.3% | 92.3% | 96.0% | 96.0% | 95.0% |

## Cost

Measured per model over every review it analyzed, across the datasets.

| Model | Reviews analyzed | Input tokens | Output tokens | Cost | Cost per 1,000 reviews |
|---|---:|---:|---:|---:|---:|
| claude-haiku-5-5 | 360 | 42760 | 12422 | $0.010487 | $0.0291 |
| claude-sonnet-5-5 | 360 | 42760 | 11925 | $0.204770 | $0.5688 |

## By category

| Model | Category | Reviews | Sentiment accuracy | Macro precision | Macro recall | Topic precision | Topic recall | Exact topics |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| claude-haiku-5-5 | sarcasm | 15 | 93.3% | 96.4% | 75.0% | 86.7% | 86.7% | 80.0% |
| claude-haiku-5-5 | mixed | 15 | 86.7% | 86.9% | 86.9% | 94.1% | 100.0% | 93.3% |
| claude-haiku-5-5 | typos | 15 | 93.3% | 91.7% | 88.9% | 100.0% | 100.0% | 100.0% |
| claude-haiku-5-5 | short | 15 | 100.0% | 100.0% | 100.0% | 88.9% | 100.0% | 93.3% |
| claude-sonnet-5-5 | sarcasm | 15 | 100.0% | 100.0% | 100.0% | 93.3% | 93.3% | 93.3% |
| claude-sonnet-5-5 | mixed | 15 | 86.7% | 86.7% | 90.5% | 100.0% | 93.8% | 93.3% |
| claude-sonnet-5-5 | typos | 15 | 93.3% | 91.7% | 88.9% | 100.0% | 100.0% | 100.0% |
| claude-sonnet-5-5 | short | 15 | 100.0% | 100.0% | 100.0% | 88.9% | 100.0% | 93.3% |

## Errors

| Model | Review | Category | Expected | Predicted |
|---|---|---|---|---|
| claude-haiku-5-5 | r0016 | regular | negative: delivery_delay | negative: app_usability, delivery_delay |
| claude-haiku-5-5 | r0030 | regular | positive: packaging | positive: no topics |
| claude-haiku-5-5 | r0038 | regular | positive: packaging | positive: no topics |
| claude-haiku-5-5 | r0041 | regular | negative: packaging, product_quality | negative: delivery_delay, packaging, product_quality |
| claude-haiku-5-5 | r0042 | regular | negative: packaging, product_quality | negative: delivery_delay, packaging, product_quality |
| claude-haiku-5-5 | r0060 | regular | neutral: packaging | positive: packaging |
| claude-haiku-5-5 | r0062 | regular | negative: refund_process | negative: customer_support, refund_process |
| claude-haiku-5-5 | r0073 | regular | negative: delivery_delay | negative: app_usability, customer_support, delivery_delay |
| claude-haiku-5-5 | r0084 | regular | negative: customer_support | negative: customer_support, product_quality |
| claude-haiku-5-5 | r0116 | regular | negative: other | negative: product_quality |
| claude-haiku-5-5 | r0119 | regular | negative: customer_support, delivery_delay | negative: customer_support, delivery_delay, product_quality |
| claude-haiku-5-5 | r0125 | regular | negative: customer_support | negative: customer_support, packaging, product_quality |
| claude-haiku-5-5 | r0131 | regular | negative: customer_support | negative: customer_support, product_quality |
| claude-haiku-5-5 | r0133 | regular | negative: refund_process | negative: app_usability, refund_process |
| claude-haiku-5-5 | r0137 | regular | negative: customer_support | negative: customer_support, product_quality |
| claude-haiku-5-5 | r0141 | regular | negative: customer_support, refund_process | negative: refund_process |
| claude-haiku-5-5 | r0148 | regular | positive: packaging | positive: no topics |
| claude-haiku-5-5 | r0151 | regular | negative: packaging, product_quality | negative: delivery_delay, packaging, product_quality |
| claude-haiku-5-5 | r0152 | regular | positive: packaging | positive: no topics |
| claude-haiku-5-5 | r0166 | regular | negative: customer_support, refund_process | negative: refund_process |
| claude-haiku-5-5 | r0201 | regular | negative: app_usability, billing | negative: app_usability |
| claude-haiku-5-5 | r0209 | regular | negative: delivery_delay | negative: app_usability, customer_support, delivery_delay |
| claude-haiku-5-5 | r0217 | regular | negative: customer_support | negative: customer_support, product_quality |
| claude-haiku-5-5 | r0219 | regular | negative: delivery_delay, packaging | negative: delivery_delay, packaging, product_quality |
| claude-haiku-5-5 | r0224 | regular | negative: product_quality | negative: pricing, product_quality |
| claude-haiku-5-5 | r0245 | regular | negative: customer_support, refund_process | negative: refund_process |
| claude-haiku-5-5 | r0255 | regular | neutral: pricing | negative: pricing |
| claude-haiku-5-5 | r0263 | regular | neutral: product_quality | negative: product_quality |
| claude-haiku-5-5 | r0269 | regular | negative: customer_support | negative: customer_support, product_quality |
| claude-haiku-5-5 | r0279 | regular | negative: other | negative: product_quality |
| claude-haiku-5-5 | h003 | sarcasm | negative: packaging, product_quality | negative: packaging |
| claude-haiku-5-5 | h010 | sarcasm | positive: no topics | negative: pricing |
| claude-haiku-5-5 | h011 | sarcasm | negative: product_quality | negative: other |
| claude-haiku-5-5 | h019 | mixed | positive: delivery_delay | neutral: delivery_delay |
| claude-haiku-5-5 | h025 | mixed | positive: no topics | positive: pricing |
| claude-haiku-5-5 | h028 | mixed | neutral: refund_process | positive: refund_process |
| claude-haiku-5-5 | h042 | typos | neutral: no topics | positive: no topics |
| claude-haiku-5-5 | h047 | short | negative: no topics | negative: other |
| claude-sonnet-5-5 | r0016 | regular | negative: delivery_delay | negative: customer_support, delivery_delay |
| claude-sonnet-5-5 | r0030 | regular | positive: packaging | positive: no topics |
| claude-sonnet-5-5 | r0035 | regular | negative: refund_process | negative: app_usability, refund_process |
| claude-sonnet-5-5 | r0038 | regular | positive: packaging | positive: no topics |
| claude-sonnet-5-5 | r0041 | regular | negative: packaging, product_quality | negative: packaging |
| claude-sonnet-5-5 | r0042 | regular | negative: packaging, product_quality | negative: packaging |
| claude-sonnet-5-5 | r0047 | regular | negative: customer_support, refund_process | negative: refund_process |
| claude-sonnet-5-5 | r0061 | regular | negative: packaging, product_quality | negative: packaging |
| claude-sonnet-5-5 | r0080 | regular | neutral: app_usability | positive: app_usability |
| claude-sonnet-5-5 | r0084 | regular | negative: customer_support | negative: customer_support, product_quality |
| claude-sonnet-5-5 | r0089 | regular | negative: refund_process | negative: app_usability, refund_process |
| claude-sonnet-5-5 | r0116 | regular | negative: other | negative: product_quality |
| claude-sonnet-5-5 | r0119 | regular | negative: customer_support, delivery_delay | negative: customer_support, delivery_delay, product_quality |
| claude-sonnet-5-5 | r0125 | regular | negative: customer_support | negative: customer_support, product_quality |
| claude-sonnet-5-5 | r0131 | regular | negative: customer_support | negative: customer_support, product_quality |
| claude-sonnet-5-5 | r0137 | regular | negative: customer_support | negative: customer_support, product_quality |
| claude-sonnet-5-5 | r0149 | regular | negative: refund_process | negative: app_usability, refund_process |
| claude-sonnet-5-5 | r0151 | regular | negative: packaging, product_quality | negative: packaging |
| claude-sonnet-5-5 | r0153 | regular | negative: refund_process | negative: app_usability, refund_process |
| claude-sonnet-5-5 | r0217 | regular | negative: customer_support | negative: customer_support, product_quality |
| claude-sonnet-5-5 | r0219 | regular | negative: delivery_delay, packaging | negative: delivery_delay, packaging, product_quality |
| claude-sonnet-5-5 | r0268 | regular | negative: packaging, product_quality | negative: packaging |
| claude-sonnet-5-5 | r0279 | regular | negative: other | negative: product_quality |
| claude-sonnet-5-5 | h011 | sarcasm | negative: product_quality | negative: other |
| claude-sonnet-5-5 | h021 | mixed | positive: packaging | positive: no topics |
| claude-sonnet-5-5 | h028 | mixed | neutral: refund_process | positive: refund_process |
| claude-sonnet-5-5 | h029 | mixed | neutral: delivery_delay | negative: delivery_delay |
| claude-sonnet-5-5 | h043 | typos | neutral: delivery_delay | positive: delivery_delay |
| claude-sonnet-5-5 | h047 | short | negative: no topics | negative: other |
