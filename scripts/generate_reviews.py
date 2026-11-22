"""Generate a seeded, fictional review dataset with ground-truth labels.

Usage: python scripts/generate_reviews.py [--seed 42] [--out data/sample]

Everything here is fictional. The output is a pure function of the seed (only
`random.Random(seed)` is used: no global random state, no clock), so the committed
sample can be checked byte-for-byte against a fresh run.

Design notes:

* Ground truth is per text: every distinct text maps to exactly one (sentiment, topics)
  label, also for the exact duplicates injected on purpose. Duplicates copy the rating too.
* The star rating is deliberately noisy relative to the sentiment (see RATING_WEIGHTS: a
  few positive reviews have 2-3 stars, a few negative ones 4). Sentiment, not rating, is
  the ground truth; every (sentiment, rating) pair that appears has a non-zero weight.
* The committed sample in data/sample/ must be regenerated whenever the templates, tails or
  any other generator logic change: run `python scripts/generate_reviews.py`. A test fails
  when the committed files differ from a fresh run.
"""

import argparse
import csv
import datetime as dt
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from review_radar.models import Sentiment, Topic

COMPANY = "Quillfeather Home Goods"
DEFAULT_SEED = 42
DEFAULT_COUNT = 300
DEFAULT_START = dt.date(2026, 11, 10)
DEFAULT_DAYS = 9
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "sample"

CSV_HEADER = ("review_id", "review_date", "rating", "text", "source")

SENTIMENT_SHARES = (
    (Sentiment.POSITIVE, 0.45),
    (Sentiment.NEUTRAL, 0.20),
    (Sentiment.NEGATIVE, 0.35),
)
RATING_WEIGHTS: dict[Sentiment, dict[int, int]] = {
    Sentiment.POSITIVE: {5: 55, 4: 33, 3: 8, 2: 4},
    Sentiment.NEUTRAL: {4: 20, 3: 55, 2: 20, 1: 5},
    Sentiment.NEGATIVE: {1: 40, 2: 38, 3: 17, 4: 5},
}
SOURCES = ("web", "app", "email")
SOURCE_WEIGHTS = (55, 30, 15)

TYPO_RATE = 0.05
DUPLICATE_COUNT = 6

PRODUCTS = (
    "linen duvet cover",
    "oak wall shelf",
    "ceramic planter",
    "wool throw blanket",
    "stoneware mug set",
    "bamboo drawer organizer",
    "glass water carafe",
    "cotton bath towels",
    "brass floor lamp",
    "woven storage basket",
)


# Generic trailing clauses (no complaint topics) that lengthen and diversify the texts.
TAILS: dict[Sentiment, tuple[str, ...]] = {
    Sentiment.POSITIVE: (
        "it has made our weekend routine a little more pleasant",
        "my whole family noticed the difference straight away",
        "I have already told two colleagues about this shop",
        "it fits our small flat much better than I had imagined",
        "the details are so careful that you can tell someone cared",
        "I keep finding new ways to use it around the house",
        "it looks even better in the evening light",
        "guests keep complimenting it whenever they come over",
        "it was a relief to find something that simply works",
        "I would happily buy from this shop again without a second thought",
        "after a month of daily use I still smile when I see it",
        "it was exactly the finishing touch our room needed",
        "I wish I had found this shop years ago",
        "the colours are warm and true to the pictures",
        "it feels like something that will last for many years",
        "I am already planning what to order for the spare room",
    ),
    Sentiment.NEUTRAL: (
        "it is neither better nor worse than I expected",
        "I can see it working for someone with different tastes",
        "I will wait and see how it holds up over the winter",
        "my family has mixed feelings about it",
        "it does the job and I have no strong opinion either way",
        "I am keeping it for now while I think it over",
        "it is fine for a first order and I may try something else next time",
        "some details are good and others are only average",
        "I would not call it a bargain, nor a mistake",
        "it sits somewhere in the middle of my expectations",
        "I am not sure yet whether I would recommend it",
        "my partner likes it more than I do",
        "it is hard to say anything more than that it is acceptable",
        "I read other opinions before buying and they were about right",
        "the experience was ordinary from start to finish",
        "I may update this note after a few more weeks of use",
    ),
    Sentiment.NEGATIVE: (
        "I expected much better from a shop like this",
        "it ruined what was supposed to be a pleasant weekend",
        "my family was just as unimpressed as I was",
        "I honestly cannot recommend it to anyone",
        "this was my first order here and probably my last",
        "I have rarely been so disappointed by an online order",
        "I wanted to love it, but it was not to be",
        "it left a bad impression of the whole shop",
        "I am sharing this so other buyers know what to expect",
        "I feel the whole thing was handled carelessly",
        "after all that effort I am simply tired of the whole thing",
        "my colleagues had warned me and I should have listened",
        "it was not at all what the pictures led me to believe",
        "I am left wondering if anyone checks these things",
        "I really hoped for a better experience this time",
        "it made me regret clicking the order button",
    ),
}
TAIL_RATE = 0.95


@dataclass(frozen=True)
class Template:
    sentiment: Sentiment
    topics: tuple[Topic, ...]
    text: str


@dataclass(frozen=True)
class GeneratedReview:
    review_id: str
    review_date: dt.date
    rating: int
    text: str
    source: str
    sentiment: Sentiment
    topics: tuple[Topic, ...]


def _t(sentiment: Sentiment, topics: Sequence[Topic], text: str) -> Template:
    return Template(sentiment, tuple(topics), text)


P, N, G = Sentiment.POSITIVE, Sentiment.NEUTRAL, Sentiment.NEGATIVE
DELIVERY, QUALITY, SUPPORT = Topic.DELIVERY_DELAY, Topic.PRODUCT_QUALITY, Topic.CUSTOMER_SUPPORT
PRICING, BILLING, APP = Topic.PRICING, Topic.BILLING, Topic.APP_USABILITY
PACKAGING, REFUND = Topic.PACKAGING, Topic.REFUND_PROCESS

TEMPLATES: tuple[Template, ...] = (
    # Positive: no complaint topics (a couple of mild exceptions at the end of the block).
    _t(
        P,
        (),
        "I ordered the {product} last week and it is even nicer in person than in the photos. "
        "The finish is clean and the colour matches my living room perfectly. "
        "Very happy with this purchase.",
    ),
    _t(
        P,
        (),
        "Lovely {product}. It arrived early, was well wrapped, and went straight into use. "
        "My partner already asked where we bought it, so I will be ordering a second one soon.",
    ),
    _t(
        P,
        (),
        "Great quality for the money. The {product} feels solid and the stitching and edges "
        "are neat. I have used it every day since it came and it still looks brand new.",
    ),
    _t(
        P,
        (),
        "Bought the {product} as a gift for my sister, who runs a small café, and she loved "
        "it. It looks great next to the counter. Thank you, Quillfeather team!",
    ),
    _t(
        P,
        (),
        "Support was wonderful. I had a question about sizes before ordering the {product} "
        "and got a clear, friendly answer within an hour. The product itself is excellent too.",
    ),
    _t(
        P,
        (),
        "Five stars from me. The {product} is sturdy, good looking and easy to clean. "
        "Ordering through the app took under two minutes and checkout was painless \U0001f60d",
    ),
    _t(
        P,
        (),
        "I was a bit naïve to expect much at this price, but the {product} is really "
        "well made. It has already survived two moves and a very curious cat.",
    ),
    _t(
        P,
        (),
        "Third order from this shop and, as usual, no complaints. The {product} is exactly "
        "as described and the whole process from checkout to doorstep was smooth.",
    ),
    _t(
        P,
        (),
        "Really pleased. The {product} brightened up our hallway and the quality is far "
        "better than similar items I have seen elsewhere. Would recommend to friends.",
    ),
    _t(
        P,
        (),
        "Fast delivery and the {product} was packed with care, nothing scratched or bent. "
        "It looks elegant, it feels durable, and I am glad I chose it over the cheaper one.",
    ),
    _t(
        P,
        (),
        "The app made it easy to track my order and the {product} turned up on the day "
        "promised. Quality is superb and the price was fair. Will shop here again.",
    ),
    _t(
        P,
        (),
        "Cosy, well made and good value. I use the {product} every evening and it has "
        "become my favourite thing in the flat. Thanks for a great product!",
    ),
    _t(
        P,
        (PACKAGING,),
        "Love the {product}, it is beautiful and sturdy. The box was a bit bigger than "
        "needed, but that is a tiny point. Overall I am very satisfied with the order.",
    ),
    _t(
        P,
        (DELIVERY,),
        "The {product} is gorgeous and works well. It took one day longer than the estimate, "
        "yet the quality easily made up for the wait, so four stars from me.",
    ),
    # Neutral: mixed or lukewarm, a few with minor topics.
    _t(
        N,
        (),
        "The {product} is okay. It does what it should and looks decent, but nothing about it "
        "stands out. I might buy again if there is a sale, otherwise I would look around.",
    ),
    _t(
        N,
        (),
        "Average experience overall. The {product} arrived on time and matches the description. "
        "It is fine for the price, though I expected a little more character.",
    ),
    _t(
        N,
        (PRICING,),
        "The {product} is decent quality, but I think it is slightly overpriced compared with "
        'similar items. Not a bad buy, just not the "great deal" the banner promised.',
    ),
    _t(
        N,
        (PACKAGING,),
        "Product is fine, the {product} looks as pictured. The packaging was a lot of plastic "
        "for such a small item, which felt wasteful. Otherwise no real complaints.",
    ),
    _t(
        N,
        (DELIVERY,),
        "The {product} itself is good. Delivery took about a week longer than I hoped, though "
        "the courier was polite. Three stars because the product is fine but the wait was long.",
    ),
    _t(
        N,
        (APP,),
        "I like the {product}, but the app was a bit clumsy when I tried to pick a colour and "
        "I had to restart it twice. The order went through in the end and all is well.",
    ),
    _t(
        N,
        (SUPPORT,),
        "Reply from support took two days, but the person was kind and solved my question about "
        "the {product}. Product is okay. I would have liked a faster answer.",
    ),
    _t(
        N,
        (),
        "It is a perfectly acceptable {product}. Not amazing, not terrible. "
        "It has been a month and it still looks the same, so I guess that is good enough.",
    ),
    _t(
        N,
        (QUALITY,),
        "The {product} looks lovely, but one corner felt slightly rough and the finish is not "
        "as even as in the photos. Usable though, so I am keeping it.",
    ),
    # Negative: every topic appears, alone or combined.
    _t(
        G,
        (DELIVERY,),
        "My {product} arrived nine days after the promised date and nobody told me about the "
        "delay. I had guests coming and had to improvise. The tracking page never changed.",
    ),
    _t(
        G,
        (DELIVERY,),
        "Terrible delivery experience. The courier skipped my address twice and the {product} "
        'finally showed up two weeks late. The shop said "2 days" at checkout.',
    ),
    _t(
        G,
        (QUALITY,),
        "The {product} fell apart within a month. The seams are coming loose and the finish "
        "is already peeling. For something sold as premium this is really disappointing.",
    ),
    _t(
        G,
        (QUALITY,),
        "Poor quality. The {product} looked nice online, but in real life it feels flimsy and "
        "has a strong chemical smell that has not gone away after a week of airing.",
    ),
    _t(
        G,
        (SUPPORT,),
        "I wrote three times about a damaged {product} and got one copy-paste answer that did "
        "not address my problem. Nobody picked up the phone either. Awful customer service.",
    ),
    _t(
        G,
        (SUPPORT, QUALITY),
        "The {product} had a crack on arrival and support kept asking for the same photos "
        "again and again. By the fourth email I gave up. Bad product, worse service.",
    ),
    _t(
        G,
        (PRICING,),
        "Way too expensive. The {product} cost nearly double what a very similar one costs "
        "at other shops, and the price went up again a week after I bought it. Not worth it.",
    ),
    _t(
        G,
        (PRICING, QUALITY),
        "Overpriced and underwhelming. I paid a premium for the {product} and received "
        "something that looks cheap, with visible glue marks. Definitely not worth the money.",
    ),
    _t(
        G,
        (BILLING,),
        "I was charged twice for my {product} order and the second payment is still pending on "
        "my card. The invoice also shows the wrong address. Please fix your billing system.",
    ),
    _t(
        G,
        (BILLING, SUPPORT),
        "My card was charged before the {product} even shipped and the invoice never arrived "
        "by email. When I asked about it, I was passed between two agents who each blamed "
        "the other.",
    ),
    _t(
        G,
        (APP,),
        "The app keeps crashing at checkout. It took me four attempts to order the {product} "
        "and the cart emptied itself every time I switched screens. Very frustrating \U0001f621",
    ),
    _t(
        G,
        (APP, BILLING),
        "Tried to buy the {product} in the app, but the payment screen froze and my order "
        "went through twice. The interface is confusing and the search barely works.",
    ),
    _t(
        G,
        (PACKAGING, QUALITY),
        "The box was crushed and the {product} inside had a dent. There was barely any padding, "
        "just one sheet of paper. How can you ship breakable items like this?",
    ),
    _t(
        G,
        (PACKAGING,),
        "Packaging was awful. The {product} arrived loose in an oversized torn box with "
        "tape everywhere. It was a struggle to open and it looked like it had been returned "
        "before.",
    ),
    _t(
        G,
        (REFUND,),
        "I sent the {product} back three weeks ago and I am still waiting for my refund. "
        "The return form was confusing and the status page says nothing. Very slow process.",
    ),
    _t(
        G,
        (REFUND, SUPPORT),
        "Refund process is a maze. I was told to fill in a form, then to email support, then to "
        "fill in the form again. Nobody could tell me where my money for the {product} is.",
    ),
    _t(
        G,
        (DELIVERY, PACKAGING),
        "Late and damaged. The {product} came a week after the date I was given, in a box "
        "that looked like it had been dropped from a truck. I expected much better.",
    ),
    _t(
        G,
        (QUALITY, REFUND),
        "The {product} was faulty out of the box. Returning it was a hassle and the refund "
        "arrived only after I complained twice. I will not order from here again.",
    ),
    _t(
        G,
        (Topic.OTHER,),
        "Honestly I just did not like it. The {product} was not what I imagined and the "
        "colour looks completely different in my room. Maybe my mistake, but still disappointed.",
    ),
    _t(
        G,
        (DELIVERY, SUPPORT),
        "Waited ten days for the {product} and when I asked where it was, support just "
        "repeated that it was on its way. Zero useful information and zero apology.",
    ),
)

_BY_SENTIMENT: dict[Sentiment, tuple[Template, ...]] = {
    s: tuple(t for t in TEMPLATES if t.sentiment is s) for s in Sentiment
}


def _typo(text: str, rng: random.Random) -> str:
    """Swap two adjacent inner letters of one ASCII word (no-op if none qualifies)."""
    words = text.split(" ")
    candidates = [i for i, w in enumerate(words) if len(w) >= 5 and w.isascii() and w.isalpha()]
    if not candidates:
        return text
    idx = rng.choice(candidates)
    word = words[idx]
    pos = rng.randrange(1, len(word) - 2)
    if word[pos] == word[pos + 1]:
        return text
    words[idx] = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2 :]
    return " ".join(words)


def _add_tail(text: str, sentiment: Sentiment, rng: random.Random) -> str:
    """Append a generic clause to the last sentence (keeps the sentence count unchanged)."""
    tail = rng.choice(TAILS[sentiment])  # drawn always so the stream stays aligned
    if rng.random() >= TAIL_RATE or not text.endswith("."):
        return text
    return f"{text[:-1]}; {tail}."


def _sentiment_plan(n: int, rng: random.Random) -> list[Sentiment]:
    """Exact 45/20/35 mix (up to rounding), shuffled."""
    counts = [round(n * share) for _, share in SENTIMENT_SHARES[:2]]
    counts.append(n - sum(counts))
    plan = [s for (s, _), c in zip(SENTIMENT_SHARES, counts, strict=True) for _ in range(c)]
    rng.shuffle(plan)
    return plan


def generate(
    seed: int = DEFAULT_SEED,
    n: int = DEFAULT_COUNT,
    start: dt.date = DEFAULT_START,
    days: int = DEFAULT_DAYS,
) -> list[GeneratedReview]:
    """Return `n` deterministic fictional reviews spread over `days` days from `start`."""
    rng = random.Random(seed)  # noqa: S311 - not security related
    plan = _sentiment_plan(n, rng)
    dates = sorted(start + dt.timedelta(days=rng.randrange(days)) for _ in range(n))

    rows: list[GeneratedReview] = []
    for date, sentiment in zip(dates, plan, strict=True):
        template = rng.choice(_BY_SENTIMENT[sentiment])
        text = template.text.format(product=rng.choice(PRODUCTS))
        text = _add_tail(text, sentiment, rng)
        if rng.random() < TYPO_RATE:
            text = _typo(text, rng)
        weights = RATING_WEIGHTS[sentiment]
        rating = rng.choices(list(weights), weights=list(weights.values()))[0]
        source = rng.choices(SOURCES, weights=SOURCE_WEIGHTS)[0]
        rows.append(GeneratedReview("", date, rating, text, source, sentiment, template.topics))

    _inject_duplicates(rows, rng)

    return [replace(row, review_id=f"r{i:04d}") for i, row in enumerate(rows, start=1)]


def _inject_duplicates(rows: list[GeneratedReview], rng: random.Random) -> None:
    """Overwrite a few rows with the text, labels and rating of a row on another date.

    Only the destination's date, id and source are kept, so a duplicated text always has
    one ground-truth label and a rating that is allowed for that sentiment.
    """
    if len({row.review_date for row in rows}) < 2:
        return
    for _ in range(min(DUPLICATE_COUNT, len(rows) // 2)):
        src = rng.randrange(len(rows))
        others = [i for i, row in enumerate(rows) if row.review_date != rows[src].review_date]
        dst = rng.choice(others)
        source_row = rows[src]
        rows[dst] = replace(
            rows[dst],
            rating=source_row.rating,
            text=source_row.text,
            sentiment=source_row.sentiment,
            topics=source_row.topics,
        )


def write_outputs(reviews: Sequence[GeneratedReview], seed: int, out_dir: Path) -> None:
    """Write reviews.csv and ground_truth.json (byte-identical on every platform)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "reviews.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter=",", lineterminator="\n")
        writer.writerow(CSV_HEADER)
        for r in reviews:
            writer.writerow([r.review_id, r.review_date.isoformat(), r.rating, r.text, r.source])

    truth = {
        "seed": seed,
        "company": COMPANY,
        "reviews": {
            r.review_id: {"sentiment": r.sentiment.value, "topics": [t.value for t in r.topics]}
            for r in reviews
        },
    }
    payload = json.dumps(truth, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with (out_dir / "ground_truth.json").open("w", encoding="utf-8", newline="") as fh:
        fh.write(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate fictional reviews with ground truth.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    reviews = generate(seed=args.seed)
    write_outputs(reviews, args.seed, args.out)
    print(f"Wrote {len(reviews)} reviews to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
