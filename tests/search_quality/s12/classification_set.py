"""Hand-labelled queries the embedding router is measured on.

No production classification logs exist -- `SmartRetriever classified query as:` was
never persisted anywhere on disk -- so the router could not be scored against what the
FAST_LLM actually decided in past runs. This set is the substitute: 40 queries labelled
by hand, 8 per category, written to include the cases the router is most likely to get
wrong rather than the ones it is certain to get right.

Deliberately hard entries, by category:

- ``news_current`` is about RECENCY, not topic. "nvidia earnings this quarter" is here
  while the same company's technical questions are not -- what puts it in this category
  is the time-bound clause, not the subject.
- ``comprehensive`` is about the SHAPE of the question. Its entries are multi-clause
  surveys whose individual clauses each belong to some other category -- that is the
  trap: the encoder can match a clause instead of the shape.
- ``code_technical`` entries include ones whose vocabulary is ordinary English
  ("why is my site slow"), because a query does not have to look like code to be one.
- ``academic`` includes queries with no paper-vocabulary at all ("does intermittent
  fasting actually work"), where only the expectation of evidence marks the category.

A label here is the routing bundle a human would pick, which is not always the only
defensible answer; the router is scored on accuracy over the whole set, never on any
single entry. Entries where two bundles are genuinely both fine are exactly the ones
the margin should send to the LLM, and
`test_the_encoder_never_routes_a_labelled_query_to_the_wrong_bundle` counts a decline as
neither right nor wrong.
"""

# (query, expected_category)
LABELLED_QUERIES = [
    # -- general_web: everyday facts, how-to, definitions -----------------------------
    ("how do I get a passport renewed in the US", "general_web"),
    ("what time zone is Denver in", "general_web"),
    ("how to remove red wine from a white shirt", "general_web"),
    ("what is the difference between baking soda and baking powder", "general_web"),
    ("how tall is the Eiffel Tower", "general_web"),
    ("what does a notary public actually do", "general_web"),
    ("how long does it take to hard boil an egg", "general_web"),
    ("explain compound interest in simple terms", "general_web"),

    # -- code_technical: software, APIs, debugging ------------------------------------
    ("typescript type error with generic constraints", "code_technical"),
    ("postgres index not being used by the query planner", "code_technical"),
    ("how to mock a module in vitest", "code_technical"),
    ("docker container exits immediately with code 0", "code_technical"),
    ("kubernetes pod stuck in CrashLoopBackOff", "code_technical"),
    ("difference between git reset soft and hard", "code_technical"),
    # ordinary English, still a software question
    ("why is my site slow on first load", "code_technical"),
    ("what does this segfault in my C program mean", "code_technical"),

    # -- academic: papers, studies, evidence ------------------------------------------
    ("randomized controlled trials on mindfulness for anxiety", "academic"),
    ("original paper introducing batch normalization", "academic"),
    ("systematic review of gut microbiome and depression", "academic"),
    ("what does the literature say about spaced repetition", "academic"),
    ("replication crisis in social psychology", "academic"),
    ("citations for transformer scaling laws", "academic"),
    # no paper-vocabulary; only the expectation of evidence marks it
    ("does intermittent fasting actually work", "academic"),
    ("is there real evidence that creatine improves cognition", "academic"),

    # -- news_current: recency-bound ---------------------------------------------------
    ("nvidia earnings this quarter", "news_current"),
    ("latest EU AI act enforcement news", "news_current"),
    ("who is currently leading the presidential race", "news_current"),
    ("what happened in the market today", "news_current"),
    ("recent layoffs in the tech industry", "news_current"),
    ("current bitcoin price", "news_current"),
    ("breaking news about the hurricane", "news_current"),
    ("what did the Fed announce this week", "news_current"),

    # -- comprehensive: multi-clause surveys -------------------------------------------
    ("compare the economic, environmental and geopolitical effects of LNG exports",
     "comprehensive"),
    ("full landscape of observability vendors: pricing, features, adoption and lock-in",
     "comprehensive"),
    ("everything about lithium: mining, supply chain, recycling and market outlook",
     "comprehensive"),
    ("analyze remote work's impact on productivity, commercial real estate and cities",
     "comprehensive"),
    ("state of nuclear fusion: physics, funding, companies and realistic timelines",
     "comprehensive"),
    ("survey of authentication approaches with security, UX and compliance tradeoffs",
     "comprehensive"),
    ("the history, current technology and future market of solid-state batteries",
     "comprehensive"),
    ("assess GLP-1 drugs across efficacy, side effects, cost and health system impact",
     "comprehensive"),
]

CATEGORIES = sorted({c for _, c in LABELLED_QUERIES})
