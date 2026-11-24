# Project passport: interview questions

Ten questions about Review Radar, at the level of a technical interview. The answers are written
by the author, without hints. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and
[docs/PLAN.md](docs/PLAN.md) hold the facts to check them against.

## 1. The cache and its key

What does the tool cache, and what is the cache key made of? What happens to the cached answers
when the review text, the model or the prompt changes?

**My answer:**



---

## 2. Idempotent import

Importing the same CSV twice adds no new reviews. How is a review recognized on a second import?
Which cases does this key handle badly?

**My answer:**



---

## 3. Fallback answers and the cache

In demo mode some answers come from simple rules instead of canned answers. In live mode some
reviews fail. What is written to the cache in each case, and why?

**My answer:**



---

## 4. XSS in the dashboard

The dashboard is built from data that started as untrusted CSV input. How could a script
injection reach the page, and what stops it?

**My answer:**



---

## 5. Rate limits

What happens when the Anthropic API answers with 429 or a 5xx error? How does the tool avoid both
losing work and running up the cost?

**My answer:**



---

## 6. Why Haiku

Why is Claude Haiku 5.5 the default model and not Sonnet 5.5? What evidence would make you switch?

**My answer:**



---

## 7. At most once for the summary

Why is the Telegram summary sent at most once rather than at least once? What happens when the
request times out?

**My answer:**



---

## 8. GDPR

Suppose the reviews were real and came from customers in the EU. What personal data would the
system store and send, and to whom? What would you change to comply with GDPR?

**My answer:**



---

## 9. CI and pinning by SHA

What does CI check on every push? Why are the GitHub Actions pinned to a commit SHA and not to a
tag? What does CI not protect against?

**My answer:**



---

## 10. Limitations

What are the main limitations of the project? Which one would you fix first, and how?

**My answer:**


