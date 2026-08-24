# InsightOps AI

### From customer feedback to business decisions.

> **InsightOps AI is a decision-intelligence platform that transforms unstructured customer feedback into evidence-backed insights, priorities, and recommended actions.**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/Models-Hugging%20Face-FFD21E?style=for-the-badge\&logo=huggingface\&logoColor=black)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-In%20Development-orange?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)]()

---

## The Problem

Customer feedback is everywhere.

Reviews.
Support tickets.
Surveys.
App feedback.
Product comments.

But raw feedback does not directly tell a business **what to do next**.

A conventional sentiment dashboard might tell you:

> **42% of reviews are negative.**

That is useful—but incomplete.

InsightOps AI is being designed to answer the questions behind the number:

> **Why are customers unhappy?**
> **Which issues are driving the problem?**
> **How severe are those issues?**
> **What evidence supports the conclusion?**
> **Which problems deserve attention first?**
> **What action should the business consider?**

---

# The InsightOps Approach

```text
                 CUSTOMER FEEDBACK
                         │
                         ▼
              ┌─────────────────────┐
              │  VALIDATE & CLEAN   │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │     NORMALIZE       │
              └──────────┬──────────┘
                         ▼
        ┌─────────────────────────────────┐
        │         AI ANALYSIS LAYER       │
        │                                 │
        │  Sentiment  •  Emotion          │
        │  Aspects    •  Embeddings       │
        │  Themes     •  Relationships    │
        └────────────────┬────────────────┘
                         ▼
              ┌─────────────────────┐
              │  INTELLIGENCE LAYER │
              │                     │
              │ Evidence            │
              │ Impact              │
              │ Priority            │
              │ Trends              │
              │ Recommendations     │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ DECISION DASHBOARD  │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ BUSINESS ACTION     │
              └─────────────────────┘
```

### The product principle

**Feedback → Understanding → Evidence → Priority → Action**

That is the core idea behind InsightOps AI.

---

# What Makes InsightOps AI Different?

InsightOps AI is **not designed as another sentiment-analysis dashboard**.

The differentiator is the workflow built around the analysis.

Instead of stopping at:

```text
Review → Sentiment
```

the system is designed to progress toward:

```text
Review
  ↓
What was said?
  ↓
What aspect is involved?
  ↓
What emotion is expressed?
  ↓
What themes are emerging?
  ↓
How important is the issue?
  ↓
What evidence supports it?
  ↓
What should the business investigate or prioritize?
```

### Example

A traditional system might surface:

> **Delivery — Negative**

InsightOps AI aims to produce a richer decision signal:

> **Delivery delays are emerging as a high-priority customer pain point.**
>
> Negative feedback is concentrated around delayed orders, with repeated frustration-related emotional signals across the affected theme.
>
> **Evidence:** representative customer reviews
> **Signal:** negative sentiment + repeated aspect mentions
> **Priority:** high
> **Suggested direction:** investigate delivery SLA breaches and identify the affected operational segments.

The goal is not to generate impressive-sounding AI text.

The goal is to connect:

**claim → evidence → priority → action**

---

# Core Intelligence

InsightOps AI is being built around several analytical layers.

| Layer                 | Purpose                                                                           |
| --------------------- | --------------------------------------------------------------------------------- |
| **Validation**        | Ensure uploaded feedback is usable and structurally valid                         |
| **Cleaning**          | Prepare customer text while preserving meaning                                    |
| **Normalization**     | Standardize text without destroying semantic signals                              |
| **Sentiment**         | Determine positive, neutral, or negative customer sentiment                       |
| **Emotion**           | Detect emotional signals such as anger, sadness, joy, fear, surprise, and disgust |
| **Aspect Extraction** | Identify the product/service dimension being discussed                            |
| **Embeddings**        | Represent feedback in semantic vector space                                       |
| **Theme Discovery**   | Group semantically related feedback                                               |
| **Intelligence**      | Connect signals into business-level findings                                      |
| **Prioritization**    | Determine which issues deserve attention                                          |
| **Recommendations**   | Translate findings into potential business actions                                |
| **Evidence Explorer** | Trace insights back to actual customer feedback                                   |
| **Reporting**         | Produce decision-oriented outputs for stakeholders                                |

---

# AI Analysis Stack

### Sentiment Analysis

The current sentiment layer uses:

**`cardiffnlp/twitter-roberta-base-sentiment-latest`**

The service supports:

* Positive / Neutral / Negative classification
* Confidence estimation
* Batch inference
* CPU/GPU execution
* Maximum sequence-length handling
* Invalid/empty-row skipping
* DataFrame-preserving analysis

---

### Emotion Analysis

The current emotion layer uses:

**`j-hartmann/emotion-english-distilroberta-base`**

Supported emotional signals:

```text
Joy
Sadness
Anger
Fear
Surprise
Disgust
```

Emotion is treated as a complementary signal rather than a replacement for sentiment.

---

### Aspect Extraction

The current aspect layer uses:

* KeyBERT
* Sentence Transformers
* `all-MiniLM-L6-v2`

The objective is to move from:

> "This review is negative."

toward:

> "This review is negative **about delivery**."

That additional semantic dimension becomes important when building themes, priorities, and business insights.

---

# From Classification to Decision Intelligence

The most important future layer of InsightOps AI is the **Intelligence Layer**.

This is intentionally separated from the underlying ML models.

```text
MODEL OUTPUTS
     │
     ├── Sentiment
     ├── Emotion
     ├── Aspect
     ├── Semantic similarity
     └── Theme
            │
            ▼
     INTELLIGENCE LAYER
            │
            ├── Evidence
            ├── Severity
            ├── Impact
            ├── Confidence
            ├── Trend
            ├── Priority
            └── Recommended action
            │
            ▼
      BUSINESS INSIGHT
```

This separation is deliberate.

The ML models answer:

> **What does the data indicate?**

The intelligence layer asks:

> **What does that indication mean for a decision-maker?**

---

# Evidence-First Design

InsightOps AI is designed around an important principle:

> **An insight should be traceable to evidence.**

Instead of producing an unsupported statement such as:

> "Customers are increasingly frustrated with delivery."

the system should be able to connect that conclusion to:

* the underlying theme
* relevant aspects
* sentiment distribution
* emotional signals
* review frequency
* representative customer feedback
* confidence
* available trend information

This creates a more auditable path from AI output to business interpretation.

---

# Decision-Oriented Dashboard

The final interface is designed to behave more like a **business intelligence product** than a collection of ML charts.

Planned experience:

```text
┌─────────────────────────────────────────────┐
│              INSIGHTOPS AI                  │
│        Customer Intelligence Platform       │
├─────────────────────────────────────────────┤
│                                             │
│  CUSTOMER HEALTH                            │
│  ─────────────────────────────────────────  │
│                                             │
│  Overall Sentiment     Priority Issues      │
│       62%                   04              │
│                                             │
│  ─────────────────────────────────────────  │
│                                             │
│  🔴 Delivery delays                         │
│     HIGH PRIORITY                            │
│     1,284 related reviews                    │
│                                             │
│     Why it matters                          │
│     Evidence                                │
│     Emotional signals                       │
│     Recommended action                      │
│                                             │
│  ─────────────────────────────────────────  │
│                                             │
│  Emerging Themes      Customer Signals      │
│                                             │
└─────────────────────────────────────────────┘
```

Planned sections include:

* Executive Overview
* Customer Health
* Sentiment
* Emotional Signals
* Customer Pain Points
* Themes
* Priority Issues
* Evidence Explorer
* Recommendations
* Reports

The interface should communicate **what matters**, not simply display everything that can be calculated.

---

# Architecture

```text
InsightOps_AI/
│
├── app.py
├── config.py
├── PROJECT_SPEC.md
├── README.md
├── requirements.txt
│
├── dashboard/
│   ├── aspects.py
│   ├── components.py
│   ├── emotions.py
│   ├── overview.py
│   ├── recommendations.py
│   ├── sentiment.py
│   └── themes.py
│
├── preprocessing/
│   ├── cleaner.py
│   ├── normalizer.py
│   └── validator.py
│
├── schemas/
│   └── models.py
│
├── services/
│   ├── analysis/
│   │   ├── aspects.py
│   │   ├── clustering.py
│   │   ├── embeddings.py
│   │   ├── emotion.py
│   │   └── sentiment.py
│   │
│   ├── intelligence/
│   │
│   └── pipeline/
│       └── orchestrator.py
│
├── reports/
│   ├── exporter.py
│   └── pdf_report.py
│
├── utils/
│   ├── cache.py
│   ├── constants.py
│   ├── helpers.py
│   └── logger.py
│
└── visualization/
    ├── cards.py
    ├── charts.py
    └── tables.py
```

The architecture deliberately separates:

**data preparation → AI analysis → intelligence → presentation → reporting**

This keeps model logic independent from the dashboard and leaves room for the intelligence layer to evolve without rewriting the analysis services.

---

# Data Philosophy

InsightOps AI follows an **audit-friendly data principle**:

> **Never destroy the customer's original feedback.**

Instead of overwriting the source data, analytical attributes are appended.

Conceptually:

```text
RAW INPUT
   │
   ├── source_text
   ├── rating
   ├── source
   └── ingested_at
          │
          ▼
DERIVED SIGNALS
   │
   ├── clean_text
   ├── normalized_text
   ├── sentiment
   ├── sentiment_confidence
   ├── sentiment_score
   ├── emotion
   ├── emotion_confidence
   ├── aspect
   ├── aspect_sentiment
   ├── embedding
   ├── cluster_id
   └── cluster_name
```

This allows downstream insights to remain connected to their original evidence.

---

# Current Development Status

> **InsightOps AI is actively under development.**

### Foundation — Complete

* [x] Project architecture
* [x] Configuration layer
* [x] Environment protection
* [x] Git repository
* [x] GitHub repository
* [x] Data schemas
* [x] Input validation
* [x] Text cleaning
* [x] Text normalization
* [x] Sentiment analysis service
* [x] Emotion analysis service
* [x] Aspect extraction service

### AI Pipeline — In Progress

* [ ] Semantic embeddings
* [ ] Theme / cluster discovery
* [ ] End-to-end pipeline orchestration
* [ ] Intelligence layer
* [ ] Priority scoring
* [ ] Evidence-backed insights
* [ ] Recommendations

### Product Experience — Planned

* [ ] Dataset upload experience
* [ ] Executive dashboard
* [ ] Customer health view
* [ ] Priority issue explorer
* [ ] Evidence explorer
* [ ] Recommendations interface
* [ ] Exportable reports

### Production Readiness — Planned

* [ ] Automated tests
* [ ] Edge-case handling
* [ ] Performance optimization
* [ ] Model caching
* [ ] Deployment
* [ ] Documentation
* [ ] Product screenshots
* [ ] Demo

---

# Tech Stack

| Category                | Technology                 |
| ----------------------- | -------------------------- |
| Language                | Python                     |
| Application UI          | Streamlit                  |
| Data Processing         | Pandas                     |
| NLP                     | Hugging Face Transformers  |
| Sentiment Model         | CardiffNLP RoBERTa         |
| Emotion Model           | DistilRoBERTa Emotion      |
| Aspect Extraction       | KeyBERT                    |
| Embeddings              | Sentence Transformers      |
| Semantic Representation | `all-MiniLM-L6-v2`         |
| Validation / Schemas    | Python Dataclasses         |
| Visualization           | Python visualization stack |
| Version Control         | Git + GitHub               |

---

# Performance Principles

InsightOps AI is being designed with practical deployment constraints in mind.

### Batch inference

Models process feedback in batches rather than invoking inference independently for every row.

### Model reuse

Models should be loaded once and reused rather than repeatedly initialized during a session.

### CPU / GPU awareness

The analysis services support execution on available hardware.

### Caching

Expensive model-loading and repeated computations can be cached where appropriate.

### Controlled memory usage

The pipeline is designed to avoid unnecessary duplication of large datasets and embeddings.

The goal is not premature optimization.

The goal is **predictable, reliable performance**.

---

# Why These Models?

### RoBERTa for sentiment

A transformer-based sentiment model provides richer contextual understanding than simple keyword or lexicon-based sentiment rules.

### DistilRoBERTa for emotion

Emotion provides a second analytical dimension.

Two reviews can both be negative while communicating very different signals:

```text
Negative + Anger
Negative + Sadness
Negative + Fear
```

Those distinctions can matter when interpreting customer experience.

### KeyBERT for aspects

Aspect extraction helps connect sentiment to the **thing being discussed**.

### Sentence embeddings

Embeddings enable semantic similarity and theme discovery beyond exact keyword matching.

Together, these layers create a richer representation of customer feedback than sentiment classification alone.

---

# Product Roadmap

```text
PHASE 01
FOUNDATION
    ↓
PHASE 02
DATA INGESTION
    ↓
PHASE 03
PREPROCESSING
    ↓
PHASE 04
AI ANALYSIS
    ↓
PHASE 05
INTELLIGENCE ENGINE
    ↓
PHASE 06
DECISION DASHBOARD
    ↓
PHASE 07
REPORTING
    ↓
PHASE 08
TESTING & POLISH
    ↓
PHASE 09
DEPLOYMENT
```

The priority is not to maximize the number of AI features.

The priority is to build the smallest complete workflow that genuinely helps a user move from:

> **"Here is my customer feedback."**

to:

> **"Here is what matters, why it matters, and what I should investigate next."**

---

# Example End-to-End Vision

Imagine uploading 10,000 customer reviews.

InsightOps AI should eventually transform them into something closer to:

```text
10,000 REVIEWS
      │
      ▼
CUSTOMER SIGNALS
      │
      ├── 58% Positive
      ├── 17% Neutral
      └── 25% Negative
      │
      ▼
TOP THEMES
      │
      ├── Delivery
      ├── Product Quality
      ├── Customer Support
      └── Pricing
      │
      ▼
PRIORITY ISSUES
      │
      ├── 🔴 Delivery delays
      ├── 🟠 Support response time
      └── 🟡 Product durability
      │
      ▼
EVIDENCE
      │
      ├── Representative reviews
      ├── Sentiment
      ├── Emotion
      ├── Aspect
      └── Theme frequency
      │
      ▼
ACTION
      │
      └── What should the business investigate?
```

That is the destination.

---

# Project Philosophy

### 01 — Don't stop at sentiment.

Sentiment is a signal, not the final answer.

### 02 — Preserve evidence.

Business conclusions should remain connected to customer feedback.

### 03 — Separate models from decisions.

ML predictions provide signals. The intelligence layer turns signals into decisions.

### 04 — Make uncertainty visible.

Confidence should inform interpretation rather than being hidden.

### 05 — Design for the decision-maker.

A business user should not need to understand transformers, embeddings, or clustering algorithms to use the product.

### 06 — Build only what earns its place.

Every feature should contribute to understanding, prioritization, or action.

---

# Project Status

**InsightOps AI is currently in active development.**

The analytical foundation is being built first, followed by semantic theme discovery, the intelligence engine, and the decision-oriented dashboard.

The project is intentionally being developed as a **product**, not simply as a machine-learning demonstration.

---

## Built for the question beyond the chart.

**What does the feedback mean — and what should we do about it?**

---

### Author

**Naveena Sri S**

B.Tech — Computer Science and Engineering

---

> *InsightOps AI — turning customer voice into operational intelligence.*
