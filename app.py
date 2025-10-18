import json
import logging
import time
import streamlit as st
from graph_builder import build_skill_graph, compute_graph_metrics, get_skill_recommendations
from skill_extractor import extract_skills, get_skill_category
from visualise import plot_graph, create_skill_summary_chart

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="SkillGraph",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "SkillGraph: AI-powered skill mapping and job matching"
    }
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .success-box {
            background-color: #d1e7dd;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #198754;
        }
        .warning-box {
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 SkillGraph – AI-Powered Skill Mapping")
st.markdown("*Match your skills to job requirements and discover learning paths*")

# Load job roles
try:
    with open("job_roles.json") as f:
        job_roles = json.load(f)
except Exception as e:
    logger.error(f"Failed to load job_roles.json: {e}")
    job_roles = {}

# === SIDEBAR ===
with st.sidebar:
    st.header("⚙️ Configuration")

    similarity_threshold = st.slider(
        "Skill Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Lower = more diverse connections, Higher = tighter clustering"
    )

    include_recommendations = st.checkbox(
        "Show Learning Recommendations",
        value=True,
        help="Suggest skills to learn based on your target role"
    )

    st.divider()
    st.markdown("**About**")
    st.markdown(
        """
        SkillGraph helps you:
        - Visualise skill relationships
        - Compare against job requirements
        - Find learning paths to target roles

        Built with **NetworkX**, **Transformers**, and **Streamlit**
        """
    )

# === MAIN CONTENT ===
# Input section
st.header("📋 Input Your Profile")

input_method = st.radio(
    "Choose input method:",
    ["Text Input", "Upload File"],
    horizontal=True
)

text_input = ""

if input_method == "Text Input":
    text_input = st.text_area(
        "Paste your CV, LinkedIn profile, or resume:",
        height=200,
        placeholder="Include your skills, experiences, tools you've used, etc..."
    )
else:
    uploaded_file = st.file_uploader(
        "Upload a text file (.txt)",
        type=["txt"]
    )
    if uploaded_file:
        text_input = uploaded_file.read().decode("utf-8")
        st.success(f"Loaded: {uploaded_file.name}")

# Job role selection
st.header("🎯 Target Job Role")

col1, col2 = st.columns([2, 1])
with col1:
    selected_role = st.selectbox(
        "Compare to Job Role:",
        ["None"] + list(job_roles.keys()),
        help="Choose a role to see required vs. acquired skills"
    )
with col2:
    st.write("")  # Spacing
    if selected_role != "None":
        st.metric("Required Skills", len(job_roles[selected_role]))

# === MAIN PROCESSING ===
if st.button("🚀 Generate Skill Graph", use_container_width=True):
    if not text_input.strip():
        st.warning("⚠️ Please enter some text to extract skills.")
    else:
        with st.spinner("Analysing skills..."):
            start_time = time.time()

            # Extract skills
            try:
                skills, confidence_scores = extract_skills(text_input)
            except Exception as e:
                st.error(f"Error extracting skills: {e}")
                logger.error(f"Skill extraction failed: {e}")
                st.stop()

            if not skills:
                st.warning("⚠️ No skills found in the text. Try adding technical terms.")
            else:
                # Build graph
                try:
                    G = build_skill_graph(skills, similarity_threshold)
                except Exception as e:
                    st.error(f"Error building graph: {e}")
                    logger.error(f"Graph building failed: {e}")
                    st.stop()

                execution_time = time.time() - start_time

                # === DISPLAY RESULTS ===
                # Success message
                st.markdown(f"""
                    <div class="success-box">
                    ✅ <b>Extracted {len(skills)} skills</b> • Processing time: {execution_time:.2f}s
                    </div>
                """, unsafe_allow_html=True)

                # Skill confidence chart
                st.subheader("📊 Skill Extraction Confidence")
                fig_confidence = create_skill_summary_chart(skills, confidence_scores)
                st.plotly_chart(fig_confidence, use_container_width=True)

                # Metrics
                st.subheader("📈 Graph Metrics")
                metrics = compute_graph_metrics(G)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Skills", metrics.get("num_nodes", 0))
                with col2:
                    st.metric("Connections", metrics.get("num_edges", 0))
                with col3:
                    st.metric("Network Density", f"{metrics.get('density', 0):.2f}")
                with col4:
                    st.metric("Avg Degree", f"{metrics.get('avg_degree', 0):.1f}")

                # Main graph visualisation
                st.subheader("🔗 Skill Relationship Network")

                missing_skills = []
                if selected_role != "None":
                    role_skills = set(job_roles[selected_role])
                    missing_skills = list(role_skills - set(skills))

                fig = plot_graph(G, missing_skills,
                                 title=f"Your Skills Network {f'→ {selected_role}' if selected_role != 'None' else ''}")
                st.plotly_chart(fig, use_container_width=True)

                # === JOB ROLE COMPARISON ===
                if selected_role != "None":
                    st.subheader("🎯 Job Role Analysis")

                    role_skills = set(job_roles[selected_role])
                    current_skills = set(skills)
                    matched_skills = role_skills & current_skills
                    missing = role_skills - current_skills

                    # Comparison metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        match_pct = (len(matched_skills) / len(role_skills) * 100) if role_skills else 0
                        st.metric("Skills Match", f"{match_pct:.0f}%")
                    with col2:
                        st.metric("Matched", len(matched_skills))
                    with col3:
                        st.metric("Missing", len(missing))

                    # Matched skills
                    with st.expander("✅ Matched Skills", expanded=True):
                        matched_by_category = {}
                        for skill in matched_skills:
                            cat = get_skill_category(skill)
                            if cat not in matched_by_category:
                                matched_by_category[cat] = []
                            matched_by_category[cat].append(skill)

                        for category in sorted(matched_by_category.keys()):
                            st.markdown(f"**{category}**: {', '.join(sorted(matched_by_category[category]))}")

                    # Missing skills
                    if missing:
                        with st.expander("⚠️ Missing Skills", expanded=True):
                            missing_by_category = {}
                            for skill in missing:
                                cat = get_skill_category(skill)
                                if cat not in missing_by_category:
                                    missing_by_category[cat] = []
                                missing_by_category[cat].append(skill)

                            for category in sorted(missing_by_category.keys()):
                                st.markdown(
                                    f"**{category}**: {', '.join(sorted(missing_by_category[category]))}",
                                    help="Focus on these to match the role"
                                )

                    # === RECOMMENDATIONS ===
                    if include_recommendations and missing:
                        st.subheader("💡 Learning Path Recommendations")

                        try:
                            recommendations = get_skill_recommendations(G, current_skills, role_skills)

                            if recommendations:
                                st.markdown("**Recommended order to learn (based on skill proximity):**")

                                for idx, (skill, score) in enumerate(recommendations[:5], 1):
                                    category = get_skill_category(skill)
                                    col1, col2, col3 = st.columns([1, 3, 1])
                                    with col1:
                                        st.metric(f"#{idx}", "", label_visibility="collapsed")
                                    with col2:
                                        st.write(f"**{skill}** ({category})")
                                    with col3:
                                        st.metric("Priority", f"{score:.1%}", label_visibility="collapsed")
                        except Exception as e:
                            logger.error(f"Error generating recommendations: {e}")
                            st.warning("Could not generate recommendations")

                # === SKILL BREAKDOWN BY CATEGORY ===
                st.subheader("📂 Skills by Category")

                skills_by_category = {}
                for skill in skills:
                    cat = get_skill_category(skill)
                    if cat not in skills_by_category:
                        skills_by_category[cat] = []
                    skills_by_category[cat].append(skill)

                cols = st.columns(min(3, len(skills_by_category)))
                for col_idx, (category, cat_skills) in enumerate(sorted(skills_by_category.items())):
                    with cols[col_idx % len(cols)]:
                        st.markdown(f"**{category}**")
                        for skill in sorted(cat_skills):
                            st.write(f"• {skill}")

                # === EXPORT OPTIONS ===
                st.subheader("📥 Export Results")

                col1, col2 = st.columns(2)

                with col1:
                    # Export as JSON
                    export_data = {
                        "skills": skills,
                        "total_skills": len(skills),
                        "graph_metrics": metrics,
                        "skills_by_category": skills_by_category,
                    }

                    if selected_role != "None":
                        export_data["job_role"] = selected_role
                        export_data["matched_skills"] = list(matched_skills)
                        export_data["missing_skills"] = list(missing)
                        export_data["match_percentage"] = match_pct

                    json_str = json.dumps(export_data, indent=2)
                    st.download_button(
                        label="📄 Download as JSON",
                        data=json_str,
                        file_name="skillgraph_results.json",
                        mime="application/json"
                    )

                with col2:
                    # Export as CSV
                    csv_data = "Skill,Category,Confidence\n"
                    for skill, conf in confidence_scores:
                        cat = get_skill_category(skill)
                        csv_data += f"{skill},{cat},{conf:.2f}\n"

                    st.download_button(
                        label="📊 Download as CSV",
                        data=csv_data,
                        file_name="skillgraph_results.csv",
                        mime="text/csv"
                    )

                # === FOOTER ===
                st.divider()
                st.markdown(
                    """
                    **💡 Tips for best results:**
                    - Include specific tools, frameworks, and technologies you've used
                    - Mention projects and their technical stacks
                    - Be explicit about soft skills (e.g., "System Design", "Agile")
                    - The model learns from context, so longer, more detailed inputs work better

                    **🔧 Technical Details:**
                    - Uses `sentence-transformers` for semantic similarity
                    - Graph built with `NetworkX` and optimised layout
                    - Interactive visualisation via `Plotly`
                    """
                )