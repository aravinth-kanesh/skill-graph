# SkillGraph

**Extract skills from your CV, map their relationships as an interactive graph, and find the shortest path to your target job role.**

<!-- Replace with a real screenshot once deployed -->
<!-- ![SkillGraph demo](assets/demo.png) -->

---

## What it does

SkillGraph takes a CV as input, runs it through a three-strategy NLP pipeline to extract technical skills, then builds a weighted graph where nodes are skills and edges represent semantic or domain relationships. Given a target job role, it computes which skills you're missing and ranks them by proximity to skills you already have -- so you learn in the most efficient order.

---

## Quick start

```bash
git clone https://github.com/<your-username>/skillgraph.git
cd skillgraph
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

Visit `http://localhost:8501`. Use "Load sample CV" to try it without uploading anything.

---

## Features

- **Skill extraction**: Three complementary strategies (substring, regex, spaCy NER) with per-skill confidence scores
- **Relationship graph**: Edges built from sentence-transformer semantic similarity plus 25+ hardcoded domain rules (Python-Flask, Docker-Kubernetes, etc.)
- **Job role matching**: Compare extracted skills against 18+ predefined roles; see matched and missing skills grouped by category
- **Learning recommendations**: Missing skills ranked by shortest-path distance to your current skills in the graph
- **Graph metrics**: Network density, average degree, PageRank-based centrality
- **Export**: Download results as JSON or CSV

---

## Architecture

```
skill-graph/
├── app.py                # Streamlit application
├── skill_extractor.py    # Multi-strategy skill extraction
├── graph_builder.py      # Graph construction, metrics, recommendations
├── visualise.py          # Plotly network graph and confidence chart
├── job_roles.json        # 18+ job role definitions
├── config.yaml           # Configuration
└── test_suite.py         # pytest test suite
```

### Tech stack

| Layer | Library |
|-------|---------|
| NLP | spaCy, sentence-transformers |
| Graph | NetworkX |
| Visualisation | Plotly |
| Frontend | Streamlit |
| Testing | pytest |

### Algorithm complexity

| Component | Time | Space |
|-----------|------|-------|
| Skill extraction | O(n*m) | O(n) |
| Semantic similarity | O(n^2) | O(n^2) |
| Graph building | O(n^2 + k) | O(n + e) |
| Recommendations | O(n*log n) | O(n) |

n = skills, m = vocabulary size, e = edges, k = domain relationships

---

## How it works

### Skill extraction

Three strategies run in sequence; each assigns a confidence score:

1. **Substring matching** (0.95): Direct lookup of known skills in the input text. Fast and precise for explicit mentions.
2. **Regex patterns** (0.85): Catches common abbreviations like `ML` (Machine Learning), `NLP`, `RESTful`.
3. **spaCy NER** (0.80): Named Entity Recognition over PRODUCT and ORG entities, cross-referenced against the skill database to reduce false positives.

### Graph construction

1. Encode all extracted skills with `all-MiniLM-L6-v2`
2. Compute cosine similarity; add an edge between any two skills above the threshold (default 0.3)
3. Overlay domain knowledge edges at higher weights (e.g. Python-Flask: 0.95)
4. Compute PageRank to identify the most central skill

### Recommendations algorithm

For each missing skill, score it by proximity to your current skills:

```
score = sum(1 / (shortest_path_distance + 1)) for each current skill
```

Skills closer to what you already know score higher. Skills with no graph path score 0.

---

## Testing

```bash
pytest -v --cov=.
```

The test suite covers unit tests for each module, integration tests for the full pipeline, edge cases (empty input, unicode, special characters), and a performance test asserting extraction completes in under 5 seconds.

---

## Configuration

Edit `config.yaml` to change the similarity threshold, transformer model, graph layout parameters, or logging level.

---

## Roadmap

- PDF and DOCX CV parsing
- Skill proficiency levels (Beginner / Intermediate / Expert) with weighted graph influence
- GitHub Actions CI badge
- Shareable results via URL params
