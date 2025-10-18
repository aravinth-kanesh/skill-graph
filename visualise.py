import networkx as nx
import plotly.graph_objects as go
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

CATEGORY_COLORS = {
    "Backend": "#3B82F6",  # Blue
    "Frontend": "#10B981",  # Green
    "Data": "#F59E0B",  # Amber
    "DevOps": "#8B5CF6",  # Purple
    "Databases": "#EC4899",  # Pink
    "Other": "#6B7280"  # Gray
}

CATEGORY_MAP = {
    "Python": "Backend",
    "Flask": "Backend",
    "Django": "Backend",
    "FastAPI": "Backend",
    "Spring": "Backend",
    "Java": "Backend",
    "Node.js": "Backend",
    "Express": "Backend",
    "Go": "Backend",
    "Rust": "Backend",
    "C++": "Backend",
    "C#": "Backend",
    ".NET": "Backend",
    "NumPy": "Data",
    "Pandas": "Data",
    "Scikit-learn": "Data",
    "Machine Learning": "Data",
    "TensorFlow": "Data",
    "PyTorch": "Data",
    "Statistics": "Data",
    "R": "Data",
    "JavaScript": "Frontend",
    "React": "Frontend",
    "Vue": "Frontend",
    "Angular": "Frontend",
    "TypeScript": "Frontend",
    "HTML": "Frontend",
    "CSS": "Frontend",
    "Tailwind": "Frontend",
    "Next.js": "Frontend",
    "Svelte": "Frontend",
    "Docker": "DevOps",
    "Kubernetes": "DevOps",
    "AWS": "DevOps",
    "GCP": "DevOps",
    "Azure": "DevOps",
    "Jenkins": "DevOps",
    "CI/CD": "DevOps",
    "Terraform": "DevOps",
    "Linux": "DevOps",
    "Git": "DevOps",
    "SQL": "Databases",
    "PostgreSQL": "Databases",
    "MySQL": "Databases",
    "MongoDB": "Databases",
    "Redis": "Databases",
    "Cassandra": "Databases",
    "Elasticsearch": "Databases",
    "DynamoDB": "Databases",
    "REST APIs": "Other",
    "GraphQL": "Other",
    "Microservices": "Other",
    "SOLID": "Other",
    "Design Patterns": "Other",
    "Agile": "Other",
    "System Design": "Other",
}

def plot_graph(G: nx.Graph,
               missing_skills: Optional[List[str]] = None,
               title: str = "Skill Relationship Graph") -> go.Figure:
    """
    Create an interactive Plotly visualisation of the skill graph.

    Args:
        G: NetworkX graph
        missing_skills: Skills required but not present
        title: Graph title

    Returns:
        Plotly figure object
    """
    if missing_skills is None:
        missing_skills = []

    if G.number_of_nodes() == 0:
        logger.warning("Empty graph provided")
        return go.Figure()

    try:
        # Use spring layout with better parameters
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42, scale=10)
    except Exception as e:
        logger.error(f"Error in layout: {e}")
        pos = nx.spring_layout(G, seed=42)

    # Build edge trace
    edge_x = []
    edge_y = []
    edge_weights = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        weight = G[edge[0]][edge[1]].get('weight', 0.5)
        edge_weights.extend([weight, weight, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='rgba(125, 125, 125, 0.5)'),
        hoverinfo='none',
        mode='lines',
        name='Relationships'
    )

    # Build node trace
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []
    node_names = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_names.append(node)

        # Build hover text
        neighbors = list(G.neighbors(node))
        degree = G.degree(node)
        hover_text = f"<b>{node}</b><br>"
        hover_text += f"Connections: {degree}<br>"
        if neighbors:
            hover_text += f"Connected to: {', '.join(neighbors[:5])}"
            if len(neighbors) > 5:
                hover_text += f", +{len(neighbors) - 5} more"
        node_text.append(hover_text)

        # Color: missing skills = red, else by category
        if node in missing_skills:
            node_colors.append("red")
        else:
            category = CATEGORY_MAP.get(node, "Other")
            node_colors.append(CATEGORY_COLORS.get(category, "gray"))

        # Size proportional to degree (connectivity)
        node_sizes.append(15 + 8 * G.degree(node))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_names,
        hovertext=node_text,
        textposition="top center",
        textfont=dict(size=10, color='black'),
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white'),
            opacity=0.9
        ),
        name='Skills'
    )

    # Create legend for categories
    category_traces = []
    for category, color in CATEGORY_COLORS.items():
        category_trace = go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=category
        )
        category_traces.append(category_trace)

    if missing_skills:
        missing_trace = go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Missing Skills'
        )
        category_traces.append(missing_trace)

    all_traces = [edge_trace, node_trace] + category_traces

    fig = go.Figure(data=all_traces,
                    layout=go.Layout(
                        title=dict(
                            text=title,
                            font=dict(size=20, color='#1F2937')
                        ),
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(b=50, l=20, r=20, t=80),
                        plot_bgcolor='rgba(240, 240, 240, 0.5)',
                        paper_bgcolor='white',
                        xaxis=dict(showgrid=False, zeroline=False, visible=False),
                        yaxis=dict(showgrid=False, zeroline=False, visible=False),
                        font=dict(family="Arial, sans-serif", size=12),
                        height=600
                    ))

    return fig

def create_skill_summary_chart(skills: List[str],
                               confidence_scores: List[tuple]) -> go.Figure:
    """
    Create a bar chart showing skill confidence scores.

    Args:
        skills: List of skills
        confidence_scores: List of (skill, confidence) tuples

    Returns:
        Plotly bar chart
    """
    if not confidence_scores:
        return go.Figure()

    skill_names = [s[0] for s in confidence_scores]
    scores = [s[1] for s in confidence_scores]
    colors = [CATEGORY_COLORS.get(CATEGORY_MAP.get(s, "Other"), "gray")
              for s in skill_names]

    fig = go.Figure(data=[
        go.Bar(
            x=skill_names,
            y=scores,
            marker=dict(color=colors),
            text=[f'{s:.0%}' for s in scores],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.0%}<extra></extra>'
        )
    ])

    fig.update_layout(
        title="Skill Extraction Confidence",
        xaxis_title="Skills",
        yaxis_title="Confidence Score",
        yaxis=dict(range=[0, 1]),
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(240, 240, 240, 0.5)'
    )

    return fig


# Unit tests
def test_plot_graph():
    """Test visualisation."""
    G = nx.Graph()
    G.add_edges_from([("Python", "Flask"), ("Flask", "REST APIs")])

    fig = plot_graph(G, missing_skills=["Django"])
    assert fig is not None, "Figure should be created"
    assert "Python" in str(fig.data), "Skills should be in figure"

    logger.info("Visualisation tests passed!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_plot_graph()