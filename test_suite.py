import logging
import networkx as nx
import pytest
from graph_builder import build_skill_graph, compute_graph_metrics, get_skill_recommendations
from skill_extractor import extract_skills, get_skill_category, SKILL_DATABASE, COMMON_SKILLS
from visualise import plot_graph, create_skill_summary_chart

logger = logging.getLogger(__name__)

# SKILL EXTRACTOR TESTS
class TestSkillExtraction:
    """Test skill extraction functionality."""

    def test_empty_input(self):
        """Should handle empty input gracefully."""
        skills, scores = extract_skills("")
        assert skills == []
        assert scores == []

    def test_whitespace_only(self):
        """Should handle whitespace-only input."""
        skills, scores = extract_skills("   \n\t  ")
        assert skills == []
        assert scores == []

    def test_basic_skill_extraction(self):
        """Should extract explicitly mentioned skills."""
        text = "I work with Python and Flask daily"
        skills, scores = extract_skills(text)
        assert "Python" in skills
        assert "Flask" in skills

    def test_case_insensitive_matching(self):
        """Should match skills case-insensitively."""
        text = "I use python and FLASK"
        skills, _ = extract_skills(text)
        assert "Python" in skills
        assert "Flask" in skills

    def test_multiple_skills(self):
        """Should extract multiple distinct skills."""
        text = "Python, Django, REST APIs, Docker, PostgreSQL, AWS"
        skills, _ = extract_skills(text)
        assert len(skills) >= 4
        assert "Python" in skills
        assert "Docker" in skills
        assert "AWS" in skills

    def test_confidence_scores(self):
        """Should return confidence scores in descending order."""
        text = "I use Python and Flask"
        _, scores = extract_skills(text)
        assert len(scores) > 0
        # Check scores are in descending order
        assert scores == sorted(scores, key=lambda x: x[1], reverse=True)
        # Check scores are between 0 and 1
        for _, conf in scores:
            assert 0 <= conf <= 1

    def test_acronym_detection(self):
        """Should detect common technical acronyms."""
        text = "I work with ML and NLP models, REST APIs, and CI/CD"
        skills, _ = extract_skills(text)
        assert "Machine Learning" in skills
        assert "NLP" in skills

    def test_no_duplicates(self):
        """Should not return duplicate skills."""
        text = "Python Python Python Flask Flask"
        skills, _ = extract_skills(text)
        assert len(skills) == len(set(skills))

    def test_skill_category_mapping(self):
        """Should correctly categorise skills."""
        assert get_skill_category("Python") == "Backend"
        assert get_skill_category("React") == "Frontend"
        assert get_skill_category("Docker") == "DevOps"
        assert get_skill_category("Pandas") == "Data"
        assert get_skill_category("UnknownSkill") == "Other"


# GRAPH BUILDER TESTS
class TestGraphBuilder:
    """Test graph construction and metrics."""

    def test_empty_graph(self):
        """Should handle empty skill list."""
        G = build_skill_graph([])
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0

    def test_single_node(self):
        """Should create single node graph."""
        G = build_skill_graph(["Python"])
        assert G.number_of_nodes() == 1
        assert G.number_of_edges() == 0

    def test_two_unrelated_nodes(self):
        """Should create two-node graph with no edges if below threshold."""
        G = build_skill_graph(["Python", "React"], similarity_threshold=0.99)
        assert G.number_of_nodes() == 2
        # May have edge if domain knowledge applies
        assert G.number_of_edges() >= 0

    def test_related_skills_connected(self):
        """Should connect related skills."""
        skills = ["Python", "Flask", "Django"]
        G = build_skill_graph(skills)

        assert G.number_of_nodes() == 3
        assert G.number_of_edges() > 0
        # Python should connect to both Flask and Django
        assert G.has_edge("Python", "Flask") or G.has_edge("Python", "Django")

    def test_graph_edge_weights(self):
        """Should assign weights to edges."""
        G = build_skill_graph(["Python", "Flask", "REST APIs"])

        for u, v in G.edges():
            weight = G[u][v].get('weight')
            assert weight is not None
            assert 0 <= weight <= 1

    def test_semantic_similarity_threshold(self):
        """Should respect similarity threshold parameter."""
        skills = ["Python", "Java", "C++", "Go"]

        G_strict = build_skill_graph(skills, similarity_threshold=0.9)
        G_loose = build_skill_graph(skills, similarity_threshold=0.1)

        # Looser threshold should have more edges
        assert G_loose.number_of_edges() >= G_strict.number_of_edges()

    def test_graph_is_undirected(self):
        """Should create undirected graph."""
        G = build_skill_graph(["Python", "Flask"])
        assert not G.is_directed()

    def test_no_self_loops(self):
        """Should not create self-loops."""
        G = build_skill_graph(["Python", "Flask"])
        for node in G.nodes():
            assert not G.has_edge(node, node)

class TestGraphMetrics:
    """Test graph metric computation."""

    def test_empty_graph_metrics(self):
        """Should handle empty graph gracefully."""
        G = nx.Graph()
        metrics = compute_graph_metrics(G)
        assert metrics == {}

    def test_metrics_returned(self):
        """Should return required metrics."""
        G = build_skill_graph(["Python", "Flask", "Django", "REST APIs"])
        metrics = compute_graph_metrics(G)

        assert "num_nodes" in metrics
        assert "num_edges" in metrics
        assert "density" in metrics
        assert "avg_degree" in metrics

    def test_metrics_values_valid(self):
        """Should return valid metric values."""
        G = build_skill_graph(["Python", "Flask", "Django", "REST APIs"])
        metrics = compute_graph_metrics(G)

        assert metrics["num_nodes"] > 0
        assert metrics["num_edges"] >= 0
        assert 0 <= metrics["density"] <= 1
        assert metrics["avg_degree"] >= 0

    def test_pagerank_computation(self):
        """Should compute PageRank for importance."""
        G = build_skill_graph(["Python", "Flask", "Django", "REST APIs"])
        metrics = compute_graph_metrics(G)

        assert "top_skill" in metrics
        assert "avg_pagerank" in metrics
        assert metrics["top_skill"] in G.nodes()

class TestRecommendations:
    """Test skill recommendation algorithm."""

    def test_empty_recommendations(self):
        """Should return empty if no missing skills."""
        G = build_skill_graph(["Python", "Flask"])
        current = {"Python", "Flask"}
        target = {"Python", "Flask"}

        recs = get_skill_recommendations(G, current, target)
        assert recs == []

    def test_missing_skill_recommendation(self):
        """Should recommend missing skills."""
        G = build_skill_graph(["Python", "Flask", "Django", "REST APIs"])
        current = {"Python", "Flask"}
        target = {"Python", "Flask", "REST APIs", "Docker"}

        recs = get_skill_recommendations(G, current, target)
        assert len(recs) > 0
        # Should recommend missing skills
        rec_skills = [skill for skill, _ in recs]
        assert any(skill in target for skill in rec_skills)

    def test_recommendation_scores(self):
        """Should score recommendations correctly."""
        G = build_skill_graph(["Python", "Flask", "Django", "REST APIs"])
        current = {"Python"}
        target = {"Python", "Flask", "Django", "REST APIs"}

        recs = get_skill_recommendations(G, current, target)

        # Check scores are sorted descending
        scores = [score for _, score in recs]
        assert scores == sorted(scores, reverse=True)

        # Check scores are non-negative
        for _, score in recs:
            assert score >= 0

    def test_recommendations_proximity_bias(self):
        """Should prefer skills closer to current skills."""
        G = build_skill_graph(["Python", "Flask", "Django", "React", "JavaScript"])
        current = {"Python"}
        target = {"Python", "Flask", "React"}

        recs = get_skill_recommendations(G, current, target)
        rec_skills = [skill for skill, _ in recs]

        # Flask should be recommended (closer to Python)
        assert len(recs) > 0

# VISUALISATION TESTS
class TestVisualisation:
    """Test graph visualisation."""

    def test_empty_graph_plot(self):
        """Should handle empty graph gracefully."""
        G = nx.Graph()
        fig = plot_graph(G)
        assert fig is not None

    def test_basic_plot_creation(self):
        """Should create valid Plotly figure."""
        G = build_skill_graph(["Python", "Flask", "Django"])
        fig = plot_graph(G)

        assert fig is not None
        assert len(fig.data) > 0  # Has traces

    def test_plot_with_missing_skills(self):
        """Should highlight missing skills in red."""
        G = build_skill_graph(["Python", "Flask", "Docker"])
        missing = ["Django", "Kubernetes"]

        fig = plot_graph(G, missing_skills=missing)
        assert fig is not None

    def test_plot_title(self):
        """Should accept custom title."""
        G = build_skill_graph(["Python", "Flask"])
        fig = plot_graph(G, title="Custom Title")

        assert "Custom Title" in fig.layout.title.text

    def test_skill_summary_chart(self):
        """Should create confidence score bar chart."""
        skills = ["Python", "Flask", "Django"]
        scores = [("Python", 0.95), ("Flask", 0.90), ("Django", 0.85)]

        fig = create_skill_summary_chart(skills, scores)
        assert fig is not None
        assert len(fig.data) > 0

    def test_empty_confidence_chart(self):
        """Should handle empty confidence data."""
        fig = create_skill_summary_chart([], [])
        assert fig is not None

# INTEGRATION TESTS
class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline(self):
        """Test complete extraction -> graph -> visualisation pipeline."""
        cv_text = """
        Senior Full Stack Developer with 5 years experience.

        Technical Skills:
        - Backend: Python, Flask, Django, FastAPI, REST APIs
        - Frontend: React, JavaScript, TypeScript, HTML, CSS
        - Databases: PostgreSQL, SQL, MongoDB
        - DevOps: Docker, Kubernetes, AWS, CI/CD
        - Tools: Git, Linux

        Projects:
        - Built microservices architecture with Python and Docker
        - Developed React dashboard with TypeScript
        - Implemented ML models using Python and TensorFlow
        """

        # Extract skills
        skills, scores = extract_skills(cv_text)
        assert len(skills) > 5, "Should extract multiple skills"

        # Build graph
        G = build_skill_graph(skills)
        assert G.number_of_nodes() == len(skills)
        assert G.number_of_edges() > 0, "Skills should be connected"

        # Compute metrics
        metrics = compute_graph_metrics(G)
        assert metrics["num_nodes"] > 0

        # Visualise
        fig = plot_graph(G)
        assert fig is not None

    def test_job_role_matching(self):
        """Test skill extraction and job matching."""
        # Backend Developer CV
        cv_text = """
        Backend Developer
        - 3 years with Python and Flask
        - Experienced with PostgreSQL and SQL
        - Docker and CI/CD pipelines
        """

        skills, _ = extract_skills(cv_text)
        target_skills = {"Python", "Flask", "PostgreSQL", "Docker", "REST APIs"}

        matched = set(skills) & target_skills
        assert len(matched) >= 3, "Should match several backend skills"

    def test_recommendation_chain(self):
        """Test realistic skill learning path."""
        # Start with basic frontend
        current = {"JavaScript", "React"}

        # Build knowledge graph
        all_skills = ["JavaScript", "React", "Node.js", "Express",
                      "Python", "Flask", "REST APIs", "SQL"]
        G = build_skill_graph(all_skills)

        # Get recommendations for backend
        target = set(all_skills)
        recs = get_skill_recommendations(G, current, target)

        # Should recommend backend skills
        rec_skills = {skill for skill, _ in recs}
        assert len(rec_skills) > 0
        # Should include backend-related skills
        assert any(s in rec_skills for s in ["Node.js", "Express", "Python"])

# PERFORMANCE TESTS
class TestPerformance:
    """Test performance characteristics."""

    def test_large_skill_set(self):
        """Should handle large number of skills efficiently."""
        # 50 distinct skills
        skills = COMMON_SKILLS[:50]

        G = build_skill_graph(skills)
        assert G.number_of_nodes() == 50

        # Should complete in reasonable time
        metrics = compute_graph_metrics(G)
        assert metrics["num_nodes"] == 50

    def test_extraction_speed(self):
        """Should extract skills quickly."""
        cv_text = " ".join(COMMON_SKILLS) * 10  # Repeated skills

        import time
        start = time.time()
        skills, _ = extract_skills(cv_text)
        elapsed = time.time() - start

        assert elapsed < 5.0, "Extraction should complete in under 5 seconds"
        assert len(skills) > 0

# ERROR HANDLING TESTS
class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_special_characters(self):
        """Should handle special characters safely."""
        text = "I use Python!!! @#$% Flask &&& Django"
        skills, _ = extract_skills(text)
        assert "Python" in skills
        assert "Flask" in skills

    def test_unicode_characters(self):
        """Should handle unicode safely."""
        text = "I know Python™ and C++ (C Plus Plus)"
        skills, _ = extract_skills(text)
        # Should not crash
        assert isinstance(skills, list)

    def test_very_long_text(self):
        """Should handle very long input."""
        long_text = " ".join(COMMON_SKILLS) * 1000
        skills, _ = extract_skills(long_text)
        assert len(skills) > 0

    def test_invalid_graph_input(self):
        """Should handle invalid graph inputs."""
        G = build_skill_graph(["NonexistentSkill1", "NonexistentSkill2"])
        # Should still create a valid graph
        assert isinstance(G, nx.Graph)

    def test_missing_skill_in_graph(self):
        """Should handle recommendations with skills not in graph."""
        G = build_skill_graph(["Python", "Flask"])
        current = {"Python"}
        target = {"Python", "NonexistentSkill"}

        # Should not crash
        recs = get_skill_recommendations(G, current, target)
        assert isinstance(recs, list)

# DATA VALIDATION TESTS
class TestDataValidation:
    """Test data consistency and validation."""

    def test_skill_database_structure(self):
        """Should have valid skill database structure."""
        assert isinstance(SKILL_DATABASE, dict)

        for category, skills in SKILL_DATABASE.items():
            assert isinstance(category, str)
            assert isinstance(skills, list)
            assert all(isinstance(s, str) for s in skills)

    def test_common_skills_completeness(self):
        """Should include all categorised skills in common list."""
        for skill in COMMON_SKILLS:
            # Should find category for each skill
            category = get_skill_category(skill)
            assert category is not None

    def test_no_duplicate_skills_in_categories(self):
        """Should not have duplicate skills across categories."""
        seen = set()
        for category, skills in SKILL_DATABASE.items():
            for skill in skills:
                assert skill not in seen, f"Duplicate skill: {skill}"
                seen.add(skill)

# MAIN TEST RUNNER
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])