import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import torch
import os
import json
from typing import List, Dict, Tuple
import io
from transformers import AutoTokenizer, DistilBertForSequenceClassification, Trainer, AutoModelForSequenceClassification, TrainingArguments, AutoConfig

# --- THIRD-PARTY PARSING IMPORTS (MANDATORY FOR DOCX/PDF) ---
# NOTE: These libraries must be installed separately (pip install python-docx pdfplumber)
try:
    import docx
except ImportError:
    st.warning("`python-docx` not installed. DOCX files cannot be processed.")
    docx = None # Fallback to prevent runtime errors if import fails

try:
    import pdfplumber
except ImportError:
    st.warning("`pdfplumber` not installed. PDF files cannot be processed.")
    pdfplumber = None # Fallback to prevent runtime errors if import fails
# -----------------------------------------------------------

# --- PLOTLY IMPORTS ADDED HERE ---
import plotly.express as px
import plotly.graph_objects as go
# ---------------------------------

# Import Cohere, with a robust check as it's an external library
try:
    import cohere
    # FIX: Use the primary Client import which is stable for V2
    from cohere import Client
    COHERE_AVAILABLE = True
except ImportError:
    # This might happen if the environment is missing the cohere package.
    COHERE_AVAILABLE = False
    class MockCohereClient: # Define a mock class to prevent runtime errors if Cohere is missing
        def __init__(self, *args, **kwargs): pass
        def chat(self, *args, **kwargs): raise ImportError("Cohere library not found.")
    Client = MockCohereClient # Use the fixed Client name


# --- Streamlit Config ---
st.set_page_config(
    layout="wide",
    page_title="Human Value Analysis Dashboard",
    page_icon="🤖",
    initial_sidebar_state="collapsed"
)

# --- Constants & Paths ---
LABEL_COLS = ['Self-direction', 'Stimulation', 'Hedonism', 'Achievement', 'Power',
              'Security', 'Conformity', 'Tradition', 'Benevolence', 'Universalism']

# CRITICAL FIX: Point to the deployment-ready folder
SAVED_MODEL_PATH = "xdwake/HVD_distilbert"
METRICS_FILE_PATH = "./all_experiment_metrics.csv"
HYPERPARAMS_FILE_PATH = "./hyperparameters.json"

COHERE_MODEL_NAME = "command-r-plus-08-2024"

# Mock data for Model Info page if local files are missing
MOCK_METRICS = {
    'F1_Macro': 0.885,
    'F1_Micro': 0.901,
    'Exact_Match_Accuracy': 0.912,
    'Hyperparameters': {
        "base_model": "DistilBERT",
        "learning_rate": "2e-5",
        "epochs": 4,
        "batch_size": 16,
        "weight_decay": 0.01,
        "optimizer": "AdamW"
    }
}

# --- Plotly Radar Chart Function ---

def create_radar_chart(df: pd.DataFrame):
    """Generates an interactive Plotly Radar Chart for value comparison."""
    
    # Calculate the max value across both datasets to set a clean radial range
    max_count = max(df['Current Project Count'].max(), df['Target Benchmark Count'].max())
    
    # Create the Plotly figure
    fig = go.Figure()

    # Add the current project data trace
    fig.add_trace(go.Scatterpolar(
        r=df['Current Project Count'],
        theta=df['Value'],
        fill='toself',
        name='Current Project Profile',
        marker_color='rgb(59, 130, 246)', # Tailwind Blue-500
        opacity=0.8
    ))

    # Add the target benchmark data trace
    fig.add_trace(go.Scatterpolar(
        r=df['Target Benchmark Count'],
        theta=df['Value'],
        fill='toself',
        name='Target Benchmark Profile',
        marker_color='rgb(236, 72, 153)', # Tailwind Pink-500
        opacity=0.4 # Less opaque than current profile
    ))
    
    # Update layout for a professional look
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_count * 1.1], # Set max range slightly higher
                tickfont=dict(size=10, color='#94a3b8'),
                gridcolor="#334155"
            ),
            # Position the angular axis labels outside for clarity
            angularaxis=dict(
                linecolor='#475569',
                tickfont=dict(size=12, family="Inter", color='#f1f5f9')
            )
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(color='#f1f5f9')
        ),
        title_text='Schwartz Value Profile Comparison',
        title_x=0.5,
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
    )
    
    return fig

# --- Cohere Helper Functions ---

def create_value_category_list(row, label_cols):
    """Converts prediction columns (1/0) into a comma-separated string of predicted labels."""
    tags = [label for label in label_cols if row[f'{label}_Pred'] == 1]
    return ", ".join(tags) if tags else "None"

# We remove the cache decorator for Cohere functions since the client relies on an API key
# which often changes and should not be cached.
def batch_process_requirements(co, df, batch_size=50):
    """Processes requirements in batches using Cohere for broad analysis."""
    explanations = []
    total_batches = (len(df) + batch_size - 1) // batch_size
    progress_bar = st.progress(0, text="Starting Cohere batch analysis...")

    for i in range(0, len(df), batch_size):
        batch_number = i // batch_size + 1
        # Starts at 80% after local analysis (0-80), Cohere uses 80-100
        progress_percentage = int((batch_number / total_batches) * 15) + 80 

        progress_bar.progress(progress_percentage, text=f"Processing batch {batch_number}/{total_batches} with Cohere...")

        batch = df.iloc[i : i + batch_size]

        # Combine multiple requirements into a single prompt
        batch_text = "\n".join(
            [f"Requirement: {row['Requirement']} | Values: {row['Human Value Category']}"
             for _, row in batch.iterrows()]
        )

        prompt = f"""Here are several software requirements for a company, along with the identified human value categories they reflect.
{batch_text}
Based on this give a broad analysis about this batch of requirements. The analysis must be a single, continuous paragraph without any headings, titles, or lists, to be used later in a synthesis step."""

        try:
            # FIX: Replace 'messages' with 'message' for the Cohere V2 client chat method
            response = co.chat(
                model=COHERE_MODEL_NAME,
                message=prompt
            )
            # FIX: Ensure we use the correct response property (already done, but keeping for safety)
            generated_text = response.text 
            explanations.append(generated_text)
            
        except Exception as e:
            st.error(f"Error processing Cohere batch {batch_number}: {e}")
            explanations.append(f"Batch analysis failed due to error: {e}")

    progress_bar.progress(95, text="Combining batch summaries...")
    combined_explanation = "\n\n".join(explanations)
    return combined_explanation

def final_analysis_with_cohere(co, combined_explanation, value_counts_series):
    """
    Generate the final analysis by incorporating both batch summaries and value occurrences.
    """
    # Format value counts into readable text
    value_counts_text = "\n".join([f"{value}: {int(count)} occurrences" for value, count in value_counts_series.items()])
    
    # --- UPDATED PROMPT: Changed "Core Values and Mission" to "Strategic Alignment and Potential Conflict" ---
    final_prompt = f"""You are an expert in Schwartzz Human Value studies.
Here is a summary of multiple software requirements and their associated human values:

{combined_explanation}

Additionally, here is an overview of how frequently each human value appears in the dataset:

{value_counts_text}

---
**CRITICAL INSTRUCTION REGARDING DATA CONFIDENCE:**
Due to limitations in the training data for the model that generated these counts, the predictions for **Stimulation** (0 occurrences) and **Tradition** (0 occurrences) are considered highly unreliable.

You **MUST NOT** analyze the absence of **Stimulation** or **Tradition** as a meaningful finding or implication in the 'Least Represented Values' section. Instead, treat these two values as *masked* or *unreliable*.
---

Based on this, provide a broad analysis of the company's core values and how they align with its mission.
I'd like you to provide an insight addressing the most represented values and the least and how that describes the company.
Divide your analysis in 3 main themes:
1. "Most Represented Values"
2. "**Strategic Alignment and Potential Conflict**"
3. "Least Represented Values"

**EXECUTIVE REPORT FORMATTING MANDATES:**
* Start the report with a **'Executive Summary'** section (3-4 sentences long) that highlights the project’s main value focus (from Theme 1) and the largest identified risk/gap (from Theme 3).
* For the "Least Represented Values" topic, focus only on the lowest-counted values among the **eight reliable human values** (i.e., ignore Stimulation and Tradition). Discuss why these reliable, but low-count, areas might be underrepresented or overlooked in the requirements document.
* Ensure the report uses a **professional and analytical tone**. Use **Markdown headings and bullet points** liberally to ensure the analysis is clear and easily scannable.
"""


    try:
        # Call Cohere API
        response = co.chat(
            model=COHERE_MODEL_NAME,
            message=final_prompt
        )

        final_analysis = response.text
        return final_analysis

    except Exception as e:
        st.error(f"Error generating final analysis: {e}")
        return f"Final analysis generation failed: {e}"
    
# --- REPLACEMENT PARSER FUNCTION (Supports TXT, CSV, DOCX, PDF) ---
import re
from typing import List
# Note: You must ensure 'docx', 'pdfplumber', and 'st' (Streamlit) are imported 
# at the top of your UI.py script for the file handling to work.

# --- MODIFIED separate_requirements FUNCTION (More Robust Regex) ---
import re
from typing import List

def separate_requirements(raw_text: str) -> List[str]:
    """
    Extracts individual requirements blocks from text.
    Supports formats like:
    3.1.2.1 REQ-API-001: The system shall...
    3.1.2.2 REQ-API-002: ...
    """

    if not raw_text:
        return []

    pattern = re.compile(
        r"(?:^\s*\d+(?:\.\d+)*\s+)?"           # optional numeric section like 3.1.2.1
        r"((?:[A-Z]{2,5}-){1,4}\d{1,4}:\s*)"   # requirement ID (e.g., REQ-API-001:)
        r"(.*?)"                               # requirement text (non-greedy)
        r"(?=(?:^\s*\d+(?:\.\d+)*\s+)?(?:[A-Z]{2,5}-){1,4}\d{1,4}:|\Z)",  # next req or end
        re.DOTALL | re.MULTILINE
    )

    matches = pattern.findall(raw_text)
    requirements = [(req_id + text).strip() for req_id, text in matches if len(text.strip()) > 0]

    return requirements



def parse_srs_document(uploaded_file):
    """
    Parses uploaded file (TXT, CSV, DOCX, or PDF) and extracts individual 
    requirements using the separate_requirements function.
    """
    if uploaded_file is None:
        # This catch is vital if the uploaded_file variable might be None
        return []

    file_extension = uploaded_file.name.split('.')[-1].lower()
    raw_text = ""

    try:
        if file_extension in ('txt', 'csv'):
            # Use getvalue() for reliable reading of Streamlit's UploadedFile object
            raw_text = uploaded_file.getvalue().decode("utf-8")
            
        elif file_extension == 'docx':
            if docx is None:
                st.error("DOCX parsing failed: 'python-docx' library is not installed.")
                return []
            
            # docx.Document reads the file object directly
            document = docx.Document(uploaded_file)
            full_text = [p.text.strip() for p in document.paragraphs if p.text.strip()]
            raw_text = "\n\n".join(full_text)

        elif file_extension == 'pdf':
            if pdfplumber is None:
                st.error("PDF parsing failed: 'pdfplumber' library is not installed.")
                return []
                
            # pdfplumber.open reads the file object directly
            with pdfplumber.open(uploaded_file) as pdf:
                full_text = []
                for page in pdf.pages:
                    # layout=False can sometimes improve text extraction reliability
                    text = page.extract_text(layout=False) 
                    if text:
                        full_text.append(text)
                # Combine all text content
                raw_text = "\n\n".join(full_text)
                
        else:
            st.error(f"File format **.{file_extension}** is not supported.")
            return []

    except Exception as e:
        # Catch unexpected errors during file processing
        st.error(f"An unexpected error occurred while reading the file: {type(e).__name__}: {e}")
        return []

    if raw_text:
        # Pass the extracted text to the requirement separation function
        return separate_requirements(raw_text)
    
    # Return an empty list if no text was extracted
    return []


# --- MODERNIZED CSS ---
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; background-color: #0f172a; color: #f8fafc !important; }
.stApp { background: radial-gradient(circle at top left, #1e293b, #0f172a); color: #f1f5f9 !important; }
h1, h2, h3 { font-weight: 700; letter-spacing: -0.5px; color: #f1f5f9 !important; }
h1 { color: #60a5fa !important; text-shadow: 0 1px 3px rgba(0,0,0,0.4); margin-bottom: 0.2em; }
h2 { border-left: 4px solid #3b82f6; padding-left: 10px; color: #e2e8f0 !important; }
div[data-testid="stButton"] > button { background: linear-gradient(135deg, #2563eb, #1e40af); color: #f8fafc !important; font-weight: 600; border-radius: 10px; padding: 0.6rem 1.4rem; border: none; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4); transition: 0.2s ease-in-out; }
div[data-testid="stButton"] > button:hover { background: linear-gradient(135deg, #1e3a8a, #1d4ed8); transform: translateY(-2px); }
textarea, input, select { background-color: #1e293b !important; color: #f1f5f9 !important; border-radius: 8px !important; border: 1px solid #334155 !important; }
textarea:focus, input:focus, select:focus { border-color: #3b82f6 !important; box-shadow: 0 0 0 1px #3b82f6 !important; }
[data-testid="stFileUploaderDropzone"] { background-color: #1e293b !important; color: #e2e8f0 !important; border: 2px dashed #475569 !important; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: #3b82f6 !important; }
[role="progressbar"] { background-color: #334155 !important; border-radius: 9999px; }
[role="progressbar"] > div { background-color: #3b82f6 !important; }
[data-testid="stMetricValue"] { font-size: 2.2rem !important; font-weight: 800; color: #22c55e !important; }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
.stPlotlyChart, .stVegaLiteChart, .stPyplot { background-color: #1e293b !important; border-radius: 12px; padding: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
</style>
""", unsafe_allow_html=True)

# Set global Matplotlib style for consistency
plt.style.use('dark_background')
plt.rcParams.update({
    "figure.facecolor": "#1e293b",
    "axes.facecolor": "#1e293b",
    "axes.edgecolor": "#f1f5f9",
    "xtick.color": "#f1f5f9",
    "ytick.color": "#f1f5f9",
    "grid.color": "#334155",
    "text.color": "#f1f5f9",
    "font.size": 10
})

# --- Model Loading Function ---
@st.cache_resource
def load_model(model_path=SAVED_MODEL_PATH):
    # Ensure all necessary files are present
    if not os.path.exists(model_path):
        st.error(f"Model files not found at: {model_path}. Please ensure the directory exists and contains `config.json` and `pytorch_model.bin`.")
        return None, None
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Use AutoModelForSequenceClassification for robust loading regardless of base model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

# --- Prediction Function ---
def predict_values(requirements_list, tokenizer, model):
    # Tokenize
    encodings = tokenizer(requirements_list, padding=True, truncation=True, return_tensors="pt")
    # Inference
    with torch.no_grad():
        outputs = model(**encodings)
        # Apply sigmoid to logits to get probabilities (0 to 1)
        probs = torch.sigmoid(outputs.logits).numpy() 
    
    # Binary predictions using 0.5 threshold
    preds = (probs >= 0.5).astype(int)
    return preds, probs

# --- Utility Functions ---
def calculate_co_occurrence(df, label_cols):
    """Calculates the co-occurrence matrix between all pairs of labels."""
    df_labels = df[label_cols]
    co_occurrence_matrix = df_labels.T.dot(df_labels)
    # Set diagonal to zero for visualization purposes (self-co-occurrence is always max)
    np.fill_diagonal(co_occurrence_matrix.values, 0)
    return co_occurrence_matrix

def get_latest_test_metrics(file_path):
    """Safely loads the latest test metrics or returns mock data."""
    try:
        # Check if the metrics file exists at the user-specified path
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Find the last row tagged 'Final_Test'
            latest_test_row = df[df['Metric_Set'] == 'Final_Test'].iloc[-1].to_dict()
            # Ensure Hyperparameters field is parsed if necessary
            if 'Hyperparameters' in latest_test_row and isinstance(latest_test_row['Hyperparameters'], str):
                latest_test_row['Hyperparameters'] = json.loads(latest_test_row['Hyperparameters'])
            return latest_test_row
        else:
            # File not found: Use mock data and issue a warning
            st.warning(f"Metrics file not found at: {file_path}. Using mock data for demonstration.")
            return MOCK_METRICS
    except Exception as e:
        st.error(f"Error reading metrics from file: {e}. Using mock data.")
        return MOCK_METRICS

# --- Page: Analyze ---
def analyze_page():
    st.title("Live Requirement Analysis")
    st.write("Upload or paste requirements to analyze their relationship to the 10 Basic Human Values (Schwartz's Theory).")
    
    # --- COHERE API Key Input (Prioritizes Streamlit Secrets) ---
    
    # 1. Check Streamlit Secrets first (ideal for deployment)
    cohere_key_from_secrets = st.secrets.get("cohere_api_key")
    cohere_api_key = cohere_key_from_secrets
    
    if COHERE_AVAILABLE:
        st.subheader("Cohere Configuration")
        
        if cohere_key_from_secrets:
            st.info("✅ Cohere API Key loaded securely from **`st.secrets['cohere_api_key']`**.")
        else:
            # 2. Fallback to user input if not found in secrets (good for local testing)
            user_input_key = st.text_input(
                "Enter Cohere API Key (Required for Advanced Analysis)",
                type="password",
                help="Your Cohere API key is needed to run the detailed generative analysis. For security, please use Streamlit secrets for deployment."
            )
            cohere_api_key = user_input_key
            
        st.markdown("---")
        
        # Initialize Cohere Client if key is provided
        cohere_client = None
        if cohere_api_key:
            try:
                # Initialize Client with the retrieved key
                cohere_client = Client(cohere_api_key)
            except Exception as e:
                st.error(f"Failed to initialize Cohere client. Please check the key's validity: {e}")
                cohere_client = None
                
    else:
        st.warning("Cohere library not found. Advanced Analysis is disabled.")
        cohere_client = None
        
    # --- Requirement Input ---
    col1, col2 = st.columns([1, 1])
    with col1:
        # Updated accepted file types
        uploaded_file = st.file_uploader("Upload File (.txt, .csv, .docx, or .pdf)", type=['txt', 'csv', 'docx', 'pdf'])
    with col2:
        st.markdown("<div style='text-align:center;color:#94a3b8;margin:8px 0;'>— OR —</div>", unsafe_allow_html=True)
        text_input = st.text_area("Paste requirements here (one per line)...", height=120, placeholder="e.g.,\nREQ-001: The system shall be responsive.\nNFR-002: User data must be encrypted.")
    st.markdown("---")

    if st.button("Analyze Document", type="primary", use_container_width=True):
        
        status_placeholder = st.empty()
        status_placeholder.progress(10, text="Parsing requirements...")
        
        requirements_list = []
        if uploaded_file:
            # Use the newly updated robust parser
            requirements_list = parse_srs_document(uploaded_file)
        elif text_input.strip():
            # For pasted text, use simple line splitting
            requirements_list = [req.strip() for req in text_input.split('\n') if req.strip()]

        if not requirements_list:
            st.warning("Please upload a file or paste text to analyze.")
            status_placeholder.empty()
            return

        st.success(f"Found {len(requirements_list)} valid requirements.")

        # --- Model Loading ---
        status_placeholder.progress(40, text="Loading local prediction model (This happens once)...")
        tokenizer, model = load_model()
        if not model:
            status_placeholder.empty()
            return
            
        # --- Prediction ---
        status_placeholder.progress(60, text="Running local predictions...")
        preds, probs = predict_values(requirements_list, tokenizer, model)

        status_placeholder.progress(80, text="Preparing results...")

        # Create results DataFrame
        df_results = pd.DataFrame({
            'Requirement': requirements_list,
            **{f'{label}_Pred': preds[:, i] for i, label in enumerate(LABEL_COLS)},
            **{f'{label}_Prob': [f'{p:.3f}' for p in probs[:, i]] for i, label in enumerate(LABEL_COLS)},
        })
        
        # --- CALCULATE VALUE COUNTS ---
        # Calculate value counts using the binary predictions (needed for Cohere prompt)
        value_counts_series = df_results[[f'{label}_Pred' for label in LABEL_COLS]].sum()
        value_counts_series.index = [c.replace('_Pred', '') for c in value_counts_series.index] # Clean up index for prompt

        # Create a new column with the predicted value names for the Cohere batch processing
        df_results['Human Value Category'] = df_results.apply(
            lambda row: create_value_category_list(row, LABEL_COLS), axis=1
        )
        
        # --- Display Results ---
        st.header("Results Summary")
        
        col_metrics_1, col_metrics_2, col_metrics_3 = st.columns(3)
        total_reqs = len(requirements_list)
        total_tags = preds.sum()
        
        with col_metrics_1:
            st.metric("Total Requirements", total_reqs, delta_color="off")
        with col_metrics_2:
            st.metric("Total Value Tags", total_tags, delta_color="off")
        with col_metrics_3:
            st.metric("Avg. Tags per Req", f"{total_tags / total_reqs:.2f}", delta_color="off")


        # --- Visualization Layout (Refined) ---
        st.markdown("---")
        st.subheader("Visual Analysis of Value Distribution")
        
        chart_col_1, chart_col_2 = st.columns([1, 1.2]) # Give heatmap slightly more width

        # --- 1. Value Distribution (Bar Chart) ---
        with chart_col_1:
            st.markdown("#### 1. Value Frequency Distribution")
            
            fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
            sns.barplot(x=value_counts_series.values, y=value_counts_series.index, palette='viridis', ax=ax_bar)
            ax_bar.set_xlabel("Number of Requirements Tagged", fontsize=10)
            ax_bar.set_ylabel("Human Value", fontsize=10)
            ax_bar.set_title("Overall Value Taggings", fontsize=12)
            st.pyplot(fig_bar)
        
        # --- 2. Co-Occurrence Heatmap ---
        with chart_col_2:
            st.markdown("#### 2. Value Co-Occurrence Heatmap")
            st.write("Identifies requirements tagged with multiple values, often indicating a strong thematic connection or potential conflict.")
            
            # Create a clean DataFrame of only binary predictions for the matrix calculation
            df_binary = pd.DataFrame(preds, columns=LABEL_COLS)
            co_occurrence_matrix = calculate_co_occurrence(df_binary, LABEL_COLS)

            fig_heat, ax_heat = plt.subplots(figsize=(10, 8))
            sns.heatmap(co_occurrence_matrix, annot=True, fmt='d', cmap='YlGnBu', cbar_kws={'label': 'Co-occurrence Count'}, ax=ax_heat, linewidths=.5, linecolor="#334155")
            ax_heat.set_title("Co-occurrence Matrix of Human Values", fontsize=12)
            st.pyplot(fig_heat)

        st.markdown("---")
        
        # --- COHERE ANALYSIS INTEGRATION ---
        st.header("4. Generative Strategic Analysis (via Cohere)")
        
        if cohere_client:
            
            try:
                # Prepare DataFrame for Cohere: only need Requirement and the combined values string
                df_for_cohere = df_results[['Requirement', 'Human Value Category']]
                
                # Step 1: Get the combined explanation for all batches
                combined_broad_analysis = batch_process_requirements(
                    cohere_client, 
                    df_for_cohere, 
                    batch_size=25 # Using 25 as per the user's original call, but default is 50
                )
                
                status_placeholder.progress(98, text="Generating final strategic analysis...")

                # Step 2: Generate the final analysis
                final_analysis = final_analysis_with_cohere(
                    cohere_client, 
                    combined_broad_analysis, 
                    value_counts_series
                )
                
                st.success("Generative Analysis Complete! (Powered by Cohere)")
                
                with st.expander("View Full Strategic Analysis", expanded=True):
                    st.markdown(final_analysis)
                    
            except Exception as e:
                st.error(f"An error occurred during Cohere processing. Check the log for details. Error: {e}")
                
        else:
            st.info("The generative analysis requires the Cohere library and a valid API Key. Please ensure both are available.")

        # --- Raw Data and Download ---
        st.markdown("---")
        st.subheader("5. Detailed Prediction Data")
        
        # Display the interactive table with predictions and probabilities
        st.dataframe(df_results.drop(columns=['Human Value Category']), use_container_width=True)
        
        # Download button
        csv = df_results.drop(columns=['Human Value Category']).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='value_predictions.csv',
            mime='text/csv',
            key='download_csv'
        )

        status_placeholder.empty()
        st.success("Analysis Complete ✅")

# --- Page: Dashboard ---
def dashboard_page():
    st.title("Human Values Analysis Dashboard")
    st.markdown("This section is typically used for long-term analysis, comparing different projects, or visualizing the evolution of metrics over time. **Data persistence is required for this page.**")
    st.warning("Since this app is session-based, the Dashboard currently serves as a placeholder for future long-term data analysis (e.g., comparing current analysis to a past baseline).")


def model_page():
    st.title("Model Training Details")
    st.info("This section contains crucial metadata for validating the model's reliability and ensuring reproducibility.")
    
    col_path, col_model = st.columns(2)
    with col_path:
        st.markdown(f"**Deployment Path (Source):** `{SAVED_MODEL_PATH}`")
    with col_model:
        st.markdown("**Base Model:** DistilBERT Base Uncased")
    
    # Load metrics safely (using your existing function)
    latest_metrics = get_latest_test_metrics(METRICS_FILE_PATH)

    st.markdown("---")
    st.subheader("Performance Metrics (Final Test Set)")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    
    f1_macro = latest_metrics.get('F1_Macro', 0)
    f1_micro = latest_metrics.get('F1_Micro', 0)
    acc = latest_metrics.get('Exact_Match_Accuracy', 0)

    with col_m1:
        st.metric("Macro F1", f"{f1_macro:.4f}")
    with col_m2:
        st.metric("Micro F1", f"{f1_micro:.4f}")
    with col_m3:
        st.metric("Exact Match Accuracy", f"{acc:.4f}")

    st.markdown("---")
    st.subheader("Training Hyperparameters")

    try:
        # Load hyperparameters from the JSON file using the dedicated path
        with open(HYPERPARAMS_FILE_PATH, 'r') as f:
            loaded_params = json.load(f)

        # Map JSON keys (snake_case) to Display Names (Title Case)
        hyperparams = {
            "Learning Rate": loaded_params.get("learning_rate", "N/A"),
            "Epochs": loaded_params.get("epochs", "N/A"),
            "Train Batch Size": loaded_params.get("train_batch_size", "N/A"),
            "Eval Batch Size": loaded_params.get("eval_batch_size", "N/A"),
            "Warmup Ratio": loaded_params.get("warmup_ratio", "N/A"),
            "Weight Decay": loaded_params.get("weight_decay", "N/A"),
            "Optimizer": loaded_params.get("optimizer", "N/A"),
            "Max Seq Length": loaded_params.get("max_seq_length", "N/A"),
            "Hidden Size": loaded_params.get("hidden_size", "N/A"),
            "Hidden Layers": loaded_params.get("hidden_layers", "N/A"),
            "Dropout": loaded_params.get("dropout", "N/A"),
            "Attention Heads": loaded_params.get("attention_heads", "N/A")
        }

        hps_df = pd.DataFrame(list(hyperparams.items()), columns=['Parameter', 'Value']).set_index('Parameter')
        st.table(hps_df)

    except Exception as e:
        st.warning(f"⚠️ Could not load hyperparameters from JSON file at {HYPERPARAMS_FILE_PATH}: {e}")



# --- Main App Execution ---
tab1, tab2 = st.tabs(["📄 Analyze Requirements", "⚙️ Model Details"])
with tab1:
    analyze_page()

with tab2:
    model_page()

