import logging
from typing import List, Dict, Tuple
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Load pre-trained sentence transformer
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# Hardcoded skill relationships (domain knowledge)
SKILL_RELATIONSHIPS = [
    ("Python", "Flask", 0.95),
    ("Python", "Django", 0.95),
    ("Python", "FastAPI", 0.95),
    ("Flask", "REST APIs", 0.90),
    ("Django", "REST APIs", 0.90),
    ("FastAPI", "REST APIs", 0.90),
    ("Python", "NumPy", 0.85),
    ("Python", "Pandas", 0.85),
    ("NumPy", "Pandas", 0.88),
    ("Pandas", "Machine Learning", 0.80),
    ("Scikit-learn", "Machine Learning", 0.95),
    ("TensorFlow", "Machine Learning", 0.95),
    ("PyTorch", "Machine Learning", 0.95),
    ("Machine Learning", "Python", 0.85),
    ("JavaScript", "React", 0.92),
    ("JavaScript", "Node.js", 0.90),
    ("Node.js", "Express", 0.95),
    ("Python", "SQL", 0.85),
    ("Python", "Docker", 0.80),
    ("Docker", "Kubernetes", 0.92),
    ("Docker", "CI/CD", 0.88),
    ("Git", "CI/CD", 0.85),
    ("AWS", "Docker", 0.80),
    ("AWS", "DevOps", 0.95),
    ("Kubernetes", "DevOps", 0.95),
]

def build_skill_graph(skills: List[str], similarity_threshold: float = 0.3) -> nx.Graph:
    """
    Build a skill relationship graph using semantic similarity + domain knowledge.

    Time Complexity: O(n^2) where n = number of skills
    Space Complexity: O(n^2) for similarity matrix

    Args:
        skills: List of extracted skills
        similarity_threshold: Minimum similarity score for edge creation

    Returns:
        NetworkX graph with skill nodes and relationships
    """
    if not skills:
        return nx.Graph()

    G = nx.Graph()

    # Add all nodes
    for skill in skills:
        G.add_node(skill)

    # Strategy 1: Semantic similarity using embeddings
    try:
        if model:
            embeddings = model.encode(skills)
            similarity_matrix = cosine_similarity(embeddings)

            for i, skill_a in enumerate(skills):
                for j in range(i + 1, len(skills)):
                    score = float(similarity_matrix[i][j])
                    if score >= similarity_threshold:
                        G.add_edge(skill_a, skills[j],
                                   weight=score,
                                   source="semantic")
        else:
            logger.warning("Model not loaded; skipping semantic similarity")
    except Exception as e:
        logger.error(f"Error in semantic similarity: {e}")

    # Strategy 2: Domain-specific relationships
    try:
        for skill_a, skill_b, weight in SKILL_RELATIONSHIPS:
            if skill_a in skills and skill_b in skills:
                # Add or update edge (prefer domain knowledge)
                if G.has_edge(skill_a, skill_b):
                    # Keep highest weight
                    current = G[skill_a][skill_b]['weight']
                    G[skill_a][skill_b]['weight'] = max(current, weight)
                else:
                    G.add_edge(skill_a, skill_b, weight=weight, source="domain")
    except Exception as e:
        logger.error(f"Error in domain relationships: {e}")

    return G

def compute_graph_metrics(G: nx.Graph) -> Dict[str, float]:
    """
    Compute statistics about the skill graph.

    Returns:
        Dictionary with graph metrics
    """
    if G.number_of_nodes() == 0:
        return {}

    metrics = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
    }

    # PageRank (which skills are most central/important)
    try:
        pagerank = nx.pagerank(G)
        metrics["top_skill"] = max(pagerank, key=pagerank.get)
        metrics["avg_pagerank"] = sum(pagerank.values()) / len(pagerank)
    except Exception as e:
        logger.error(f"Error computing PageRank: {e}")

    # Average degree
    degrees = [G.degree(n) for n in G.nodes()]
    metrics["avg_degree"] = sum(degrees) / len(degrees) if degrees else 0

    return metrics

def get_skill_recommendations(G: nx.Graph, skills: set, job_skills: set) -> List[Tuple[str, float]]:
    """
    Recommend skills to learn based on graph proximity to target skills.

    Args:
        G: Skill graph
        skills: Current skills (set)
        job_skills: Target job skills (set)

    Returns:
        List of (recommended_skill, score) tuples, sorted by score
    """
    missing = job_skills - skills
    recommendations = {}

    for missing_skill in missing:
        if missing_skill not in G.nodes():
            # Skill has no graph connections, lowest priority
            recommendations[missing_skill] = 0.0
            continue

        # Score based on proximity to existing skills
        score = 0
        try:
            for current_skill in skills:
                if current_skill in G.nodes():
                    # Shortest path distance
                    try:
                        distance = nx.shortest_path_length(G, current_skill, missing_skill)
                        score += 1.0 / (distance + 1)
                    except nx.NetworkXNoPath:
                        pass
        except Exception as e:
            logger.error(f"Error in recommendations: {e}")
            score = 0.5

        recommendations[missing_skill] = score

    return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)

# Unit tests
def test_build_skill_graph():
    """Test graph building."""
    skills = ["Python", "Flask", "Django", "JavaScript", "React"]
    G = build_skill_graph(skills)

    assert G.number_of_nodes() == 5, "All skills should be nodes"
    assert G.number_of_edges() > 0, "Should have some edges"

    # Check specific relationships
    assert G.has_edge("Python", "Flask"), "Python-Flask should be connected"
    assert G.has_edge("JavaScript", "React"), "JavaScript-React should be connected"

    logger.info("Graph building tests passed!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_build_skill_graph()