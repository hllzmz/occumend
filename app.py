import os
import io
import base64
import pathlib

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, render_template

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

import openai
import chromadb
from chromadb.utils import embedding_functions

from dotenv import load_dotenv


# File paths
ABILITIES_FILE_PATH = (
    pathlib.Path(__file__).parent.resolve() / "data" / "abilities.xlsx"
)
INTERESTS_FILE_PATH = (
    pathlib.Path(__file__).parent.resolve() / "data" / "interests.xlsx"
)
KNOWLEDGE_FILE_PATH = (
    pathlib.Path(__file__).parent.resolve() / "data" / "knowledge.xlsx"
)
OCCUPATIONS_FILE_PATH = (
    pathlib.Path(__file__).parent.resolve() / "data" / "occupations.xlsx"
)
SKILLS_FILE_PATH = pathlib.Path(__file__).parent.resolve() / "data" / "skills.xlsx"
DB_PATH = pathlib.Path(__file__).parent.resolve() / "chroma_db"


# API KEYS
load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")

# CHAT LLM MODEL
LLM_CHAT_MODEL = "openai/gpt-oss-20b:free"


# DATA LOADING AND PREPARATION FUNCTIONS
def load_data():
    """Loads and merges interest and occupation data."""
    try:
        df_jobs = pd.read_excel(
            OCCUPATIONS_FILE_PATH, usecols=["O*NET-SOC Code", "Title"]
        ).set_index("O*NET-SOC Code")
        df_interests = pd.read_excel(INTERESTS_FILE_PATH)
        df_interests = df_interests[df_interests["Scale ID"] == "OI"]
        df_riasec_profiles = df_interests.pivot_table(
            index="O*NET-SOC Code", columns="Element ID", values="Data Value"
        )
        riasec_mapping = {
            "1.B.1.a": "R_score",
            "1.B.1.b": "I_score",
            "1.B.1.c": "A_score",
            "1.B.1.d": "S_score",
            "1.B.1.e": "E_score",
            "1.B.1.f": "C_score",
        }
        df_riasec_profiles.rename(columns=riasec_mapping, inplace=True)
        return df_jobs.join(df_riasec_profiles, how="inner")
    except FileNotFoundError as e:
        print(
            f"ERROR: Required data file not found: '{e.filename}'. Please make sure the files are in the correct directory."
        )
        return None


def clustering(df, k):
    """Applies clustering to job profiles and assigns cluster names."""
    features = ["R_score", "I_score", "A_score", "S_score", "E_score", "C_score"]
    job_scores = df[features].fillna(0)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(job_scores)
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=features)

    def get_cluster_name(center):
        top_three = center.nlargest(3).index
        return f"{top_three[0][0]}-{top_three[1][0]}-{top_three[2][0]}"

    cluster_centers["cluster_name"] = cluster_centers.apply(get_cluster_name, axis=1)
    df["cluster_name"] = df["cluster"].map(cluster_centers["cluster_name"].to_dict())
    return df


def get_top_elements_for_jobs(filename, scale_id_filter="IM"):
    """Gets the top 5 most important elements (knowledge, skills, abilities) for each job from the given file."""
    try:
        df = pd.read_excel(filename)
        df_important = df[df["Scale ID"] == scale_id_filter]
        top_elements = (
            df_important.sort_values("Data Value", ascending=False)
            .groupby("O*NET-SOC Code")
            .head(5)
        )
        return (
            top_elements.groupby("O*NET-SOC Code")["Element Name"].apply(list).to_dict()
        )
    except FileNotFoundError:
        print(
            f"Warning: Competency file not found: '{filename}'. This section will be skipped."
        )
        return {}


# CHART GENERATION FUNCTIONS
def create_radar_chart_image(user_profile_values):
    """Creates a radar chart for the user profile and returns it as a Base64 string."""
    labels = np.array(
        [
            "Realistic",
            "Investigative",
            "Artistic",
            "Social",
            "Enterprising",
            "Conventional",
        ]
    )
    stats = np.concatenate((user_profile_values, [user_profile_values[0]]))

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels, fontsize=10)

    ax.plot(angles, stats, color="teal", linewidth=2)
    ax.fill(angles, stats, color="teal", alpha=0.25)

    ax.set_ylim(0, 5)
    ax.grid(True)
    fig.tight_layout(pad=1.5)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"


def create_bar_chart_image(recommendations):
    """Creates a horizontal bar chart for job similarities and returns it as a Base64 string."""
    labels = [rec["Title"] for rec in recommendations]
    values = [(rec["similarity"] * 100) for rec in recommendations]

    labels.reverse()
    values.reverse()

    fig, ax = plt.subplots(figsize=(5, 5)) 
    ax.barh(labels, values, color="teal")

    ax.set_xlabel("Similarity (%)")
    ax.set_xlim(0, 100)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"


# INITIALIZE APP AND LOAD DATA
df_job_profiles = load_data()
if df_job_profiles is not None:
    df_clustered_jobs = clustering(df_job_profiles, k=8)
    knowledge_map = get_top_elements_for_jobs(KNOWLEDGE_FILE_PATH)
    skills_map = get_top_elements_for_jobs(SKILLS_FILE_PATH)
    abilities_map = get_top_elements_for_jobs(ABILITIES_FILE_PATH)
else:
    df_clustered_jobs = None
    print("Critical error: Server could not be started due to a data loading failure.")

app = Flask(__name__)

# Initialize LLM and Vector DB clients
try:
    if not OPEN_ROUTER_API_KEY:
        print(
            "Warning: OPENROUTER_API_KEY environment variable not set. Chat functionality will not work."
        )
        llm_client = None
    else:
        llm_client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPEN_ROUTER_API_KEY,
        )

    # ChromaDB connection
    chroma_client = chromadb.PersistentClient(path=DB_PATH)

    # Get the embedding function
    model_name = "all-MiniLM-L6-v2"
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )

    # Get the collection
    collection_name = "onet_data"
    onet_collection = chroma_client.get_collection(
        name=collection_name, embedding_function=sentence_transformer_ef
    )

except Exception as e:
    print(f"Error initializing RAG components: {e}")
    llm_client = None
    onet_collection = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    if df_clustered_jobs is None:
        return (
            jsonify({"error": "Server could not load data. Please check server logs."}),
            500,
        )

    user_answers = request.json
    user_profile = {
        "R_score": np.mean(user_answers.get("R", [0])),
        "I_score": np.mean(user_answers.get("I", [0])),
        "A_score": np.mean(user_answers.get("A", [0])),
        "S_score": np.mean(user_answers.get("S", [0])),
        "E_score": np.mean(user_answers.get("E", [0])),
        "C_score": np.mean(user_answers.get("C", [0])),
    }

    features = ["R_score", "I_score", "A_score", "S_score", "E_score", "C_score"]
    job_scores = df_clustered_jobs[features].fillna(0)
    user_vector = pd.DataFrame([user_profile])[features].values
    similarity_scores = cosine_similarity(user_vector, job_scores.values)
    df_clustered_jobs["similarity"] = similarity_scores[0]

    top_job_recommendations = df_clustered_jobs.sort_values(by="similarity", ascending=False).head(15)

    recommendations = []
    for index, row in top_job_recommendations.iterrows():
        job_code = index
        rec_data = {
            "Title": row["Title"],
            "cluster_name": row["cluster_name"],
            "similarity": row["similarity"],
            "knowledge": knowledge_map.get(job_code, []),
            "skills": skills_map.get(job_code, []),
            "abilities": abilities_map.get(job_code, []),
        }
        recommendations.append(rec_data)

    radar_chart_img = create_radar_chart_image(list(user_profile.values()))
    bar_chart_img = create_bar_chart_image(recommendations)

    return jsonify(
        {
            "recommendations": recommendations,
            "chart_images": {"radar": radar_chart_img, "bar": bar_chart_img},
        }
    )


@app.route("/chat", methods=["POST"])
def chat():
    if not llm_client or not onet_collection:
        return (
            jsonify({"error": "Chat functionality is not configured on the server."}),
            500,
        )

    data = request.json
    user_question = data.get("question")
    user_profile_str = data.get("profile_summary")

    if not user_question:
        return jsonify({"error": "No question provided."}), 400

    # Query the vector DB to find relevant documents
    try:
        retrieved_results = onet_collection.query(
            query_texts=[user_question],
            n_results=5,
        )
        retrieved_docs = "\n\n---\n\n".join(retrieved_results["documents"][0])
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return (
            jsonify({"error": "Failed to retrieve information from knowledge base."}),
            500,
        )

    # Construct the prompt for the LLM
    system_prompt = (
        "You are 'Career Compass AI', an expert and empathetic career strategist. "
        "Your primary goal is to help the user understand their RIASEC profile and explore potential career paths in a thoughtful and empowering way. "
        "You are not a simple Q&A bot; you are a guide."
        "\n\n"
        "### Core Directives:\n"
        "1.  **Persona & Tone**: Be professional, encouraging, and insightful. Use a positive tone that builds the user's confidence. Address the user directly and respectfully."
        "2.  **Grounding is Critical**: Base ALL your answers strictly on the user's profile summary and the O*NET job documents provided in the context. Explicitly reference the user's RIASEC scores (e.g., 'Your high score in Enterprising suggests...') and the provided job data. DO NOT invent information or provide details about jobs not included in the context."
        "3.  **Synthesize, Don't Just List**: Do not just repeat the information given to you. Your value lies in connecting the dots. Explain *why* a certain job fits (or doesn't fit) the user's profile by linking specific job tasks or work environments to their RIASEC interests."
        "4.  **Structure and Formatting**: Structure your answers for maximum clarity. Use simple HTML tags: `<h3>` for main sections, `<strong>` for emphasis, and `<ul>` with `<li>` for lists. Keep paragraphs concise."
        "5.  **Maintain Dialogue**: Always end your response with a thoughtful, open-ended question to encourage further exploration and keep the conversation going. For example, 'Which of these aspects sounds most appealing to you?' or 'Would you like to dive deeper into the daily tasks of a Landscape Architect?'"
        "\n\n"
        "### Boundaries:\n"
        "- You are NOT a life coach or a therapist. Avoid giving psychological advice."
        "- You do NOT guarantee job placement or salary outcomes."
        "- You do NOT provide information outside of the career context (e.g., financial advice, personal opinions)."
    
    )

    human_prompt = (
        f"USER PROFILE: {user_profile_str}\n\n"
        f"O*NET JOB DOCUMENTS:\n"
        f"---------------------\n"
        f"{retrieved_docs}\n"
        f"---------------------\n\n"
        f"Based on all the above, answer my question: '{user_question}'"
    )

    # Call the LLM to get a response
    try:
        response = llm_client.chat.completions.create(
            model=LLM_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt},
            ],
            temperature=0.7,
            max_tokens=2000,
        )
        answer = response.choices[0].message.content
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return (
            jsonify({"error": "An error occurred while generating the response."}),
            500,
        )


if __name__ == "__main__":
    if df_clustered_jobs is not None:
        app.run(debug=True)
    else:
        print("Application cannot start due to a data loading error.")
