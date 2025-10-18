# SkillGraph 🧠

**AI-powered skill mapping and job matching platform**

A sophisticated tool that analyses your technical skills from CVs/resumes, visualises skill relationships as an interactive network graph, and recommends learning paths to match target job roles.

---

## Features

### 🎯 Core Functionality

- **Intelligent Skill Extraction**: Multi-strategy NLP-based extraction using spaCy, regex patterns and transformer embeddings
- **Skill Relationship Graph**: Semantic similarity + domain knowledge to build meaningful skill connections
- **Interactive Visualisation**: Network graph with community detection and category-based colouring
- **Job Role Matching**: Compare your skills against 18+ job roles (Backend, Frontend, Data Science, Cloud, etc.)
- **Learning Recommendations**: Algorithmic suggestions for optimal skill acquisition order
- **Confidence Scoring**: Extraction confidence metrics for each detected skill

### 📊 Analytics & Insights

- Network density and connectivity metrics
- PageRank-based skill importance
- Category-based skill breakdown
- Skill gap analysis with missing skills highlighted
- Export results (JSON, CSV)

### 🎨 User Experience

- Multiple input methods (text, file upload)
- Configurable similarity thresholds
- Real-time processing with timing metrics
- Beautiful Plotly visualisations
- Mobile-responsive interface

---

## Architecture

### Tech Stack

- **Backend**: Python 3.8+
- **NLP**: spaCy, sentence-transformers, Transformers
- **Graph**: NetworkX
- **Visualisation**: Plotly
- **Frontend**: Streamlit
- **Testing**: pytest

### Project Structure

```
skill-graph/
│
├── app.py                # Main Streamlit application
├── skill_extractor.py    # Multi-strategy skill extraction
├── graph_builder.py      # Graph construction & metrics
├── visualise.py          # Plotly visualisation
├── job_roles.json        # Job requirement definitions
└── config.yaml           # Configuration settings
```

### Algorithm Complexity

| Component | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Skill Extraction | O(n*m) | O(n) |
| Semantic Similarity | O(n²) | O(n²) |
| Graph Building | O(n² + k) | O(n + e) |
| Recommendations | O(n*log(n)) | O(n) |

Where n = number of skills, m = vocabulary size, e = edges, k = relationships

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Setup

1. **Clone and navigate**
```bash
git clone https://github.com/yourusername/skillgraph.git
cd skillgraph
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

4. **Run application**
```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

---

## Usage

### Basic Workflow

1. **Input Your Profile**
   - Paste CV text or upload a `.txt` file
   - Include specific tools, frameworks, and technologies

2. **Select Target Role**
   - Choose from 18+ predefined job roles
   - Or compare without a specific role

3. **Generate Graph**
   - Click "Generate Skill Graph"
   - View extracted skills and confidence scores

4. **Analyse Results**
   - Explore skill relationships in network graph
   - See matched vs. missing skills for target role
   - Get personalised learning recommendations

5. **Export**
   - Download results as JSON or CSV

### Configuration

Edit `config.yaml` to customise:
- Similarity threshold (0.0-1.0)
- Model name
- Graph layout parameters
- Logging level

---

## Skill Extraction Strategy

The system uses **three complementary strategies**:

### 1. Substring Matching (High Confidence: 0.95)
Direct detection of known skills in text
- Fast and accurate for explicit mentions
- Handles case-insensitive matching

### 2. Pattern Matching (Medium Confidence: 0.85)
Regex patterns for common acronyms:
- `ML` → Machine Learning
- `NLP` → Natural Language Processing
- `API` → REST APIs

### 3. NER with spaCy (Variable Confidence: 0.80)
Named Entity Recognition for implicit mentions
- Detects product names and organisations
- Captures context-based skills

**Result**: Confidence-scored list sorted by extraction confidence

---

## Graph Construction

### Step 1: Semantic Similarity
- Encode skills with `sentence-transformers`
- Compute cosine similarity matrix
- Connect skills above threshold (default: 0.3)

### Step 2: Domain Knowledge
- Add 25+ hardcoded skill relationships
- Examples: Python → Flask, JavaScript → React
- Higher weight than semantic similarity

### Step 3: Metrics Computation
- Calculate PageRank for skill importance
- Compute network density
- Analyse connectivity patterns

---

## Recommendations Algorithm

Given target role and current skills:

1. **Identify missing skills** for target role
2. **For each missing skill**, calculate proximity score:
   ```
   score = Σ(1 / (shortest_path_distance + 1)) for all current skills
   ```
3. **Sort by score** (highest = easiest to learn)
4. **Return top recommendations**

**Intuition**: Skills closest to your existing skillset are easier to acquire.

---

## Testing

Run the test suite:

```bash
pytest -v --cov=.
```

Individual module tests:
```bash
python skill_extractor.py      # Run embedded tests
python graph_builder.py         # Run embedded tests
python visualise.py             # Run embedded tests
```

---

## Performance Considerations

### Optimisation Strategies
1. **Caching**: Embed model loaded once at startup
2. **Lazy Loading**: Models only loaded when needed
3. **Graph Algorithms**: NetworkX uses optimised C implementations
4. **Vectorisation**: NumPy for similarity computations

### Scalability Limits
- Tested up to 500 skills
- Graph computation: <1s for typical CV
- Recommendation algorithm: O(n*log(n)) ensures fast suggestions

---

## Design Decisions

### Why NetworkX?
- Lightweight, Pythonic graph library
- Rich algorithm suite (PageRank, community detection, etc.)
- Well-maintained and battle-tested

### Why sentence-transformers?
- Pre-trained on semantic similarity tasks
- Fast inference (~10ms per embedding)
- Outperforms fastText/Word2Vec on skill relationships

### Why spaCy + Regex + Substring?
- **Defense in depth**: Multiple strategies reduce false negatives
- **Confidence scoring**: Different strategies give different confidence levels
- **Robustness**: Handles varied CV formats

### Why Streamlit?
- Rapid development for data apps
- Built-in caching for performance
- Beautiful default styling
- Easy deployment

---

## Future Enhancements

### Planned Features
- [ ] **LLM-based extraction**: GPT for context-aware skill detection
- [ ] **Temporal analysis**: Track skill trends over time
- [ ] **Market data integration**: Scrape job postings for demand analysis
- [ ] **Skill proficiency levels**: Infer from CV context (junior/senior)
- [ ] **Custom skill databases**: User-defined skill categories
- [ ] **Collaborative filtering**: Recommend skills based on similar profiles

### Research Directions
- Fine-tune BERT for skill NER
- Implement skill hierarchy (dependencies)
- Build knowledge graph of skill relationships
- Apply graph neural networks for recommendations