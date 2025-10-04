# 1) Install dependencies (run this cell first — may take a minute)
# In Colab this runs shell pip installs. If you run locally, adapt accordingly.
# !pip install --quiet openai networkx pyvis sentence-transformers spacy
# !python -m spacy download en_core_web_sm

print('Packages installed. Restart the runtime if necessary and then continue.')

# 2) Imports and helpers
import os
import json
from getpass import getpass
from collections import Counter, defaultdict
from networkx.readwrite import json_graph

import networkx as nx
from pyvis.network import Network
from IPython.display import IFrame, display, HTML

import spacy
nlp = spacy.load('en_core_web_sm')

# Optional: OpenAI (only if you provide an API key)
try:
    import openai
except Exception:
    openai = None

# Embeddings model (sentence-transformers)
from sentence_transformers import SentenceTransformer, util
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

print('Imports ready.')

# !pip install PyPDF2

# Option B: upload a PDF file and extract text
from google.colab import files
import PyPDF2

# Upload the PDF
uploaded = files.upload()
filename = list(uploaded.keys())[0]

# Read and extract text from the PDF
text = ""
with open(filename, 'rb') as f:
    reader = PyPDF2.PdfReader(f)
    for page in reader.pages:
        text += page.extract_text() + "\n"

# Replace course_notes with extracted text
course_notes = text

print('Length of notes (chars):', len(course_notes))
print('\n--- Preview ---\n', course_notes[:1000])

# Preprocessing helpers

def split_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

sentences = split_sentences(course_notes)
print('Found', len(sentences), 'sentences')
for i,s in enumerate(sentences[:10]):
    print(i+1, '-', s)

def summarize_chunk(text):
    """Summarize one chunk."""
    prompt = f"Summarize the following text clearly and concisely:\n\n{text}"
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful summarizer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=400
    )
    return resp.choices[0].message.content.strip()


def extract_concepts_openai(text, max_concepts=15):
    """Extract key concepts from a given text (chunk or summary)."""
    prompt = (
        f"Extract up to {max_concepts} of the most important unique concepts from the following notes. "
        "Return ONLY JSON in this format:\n"
        '{"concepts": [{"name": "...", "description": "..."}]}\n\n'
        f"Notes:\n{text}"
    )

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You extract key concepts from study material."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=3000
    )

    content = resp.choices[0].message.content.strip()
    try:
        return json.loads(content).get("concepts", [])
    except Exception:
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            try:
                return json.loads(m.group(0)).get("concepts", [])
            except Exception:
                pass
        return []


def chunk_generator(text, chunk_size=4000, overlap=200):
    """Yield chunks one by one without keeping everything in memory."""
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        yield text[start:end]
        if end >= len(text):
            break
        start = end - overlap


def summarize_and_extract_concepts(text, max_concepts=30):
    """Stream summarize each chunk and extract concepts incrementally."""
    os.makedirs("summaries", exist_ok=True)
    all_concepts = []

    # 1️⃣ Summarize and extract per chunk
    for i, chunk in enumerate(chunk_generator(text), 1):
        print(f"⏳ Processing chunk {i}...")

        summary = summarize_chunk(chunk)
        with open(f"summaries/summary_{i}.txt", "w", encoding="utf-8") as f:
            f.write(summary)

        # Extract local concepts from this chunk’s summary
        chunk_concepts = extract_concepts_openai(summary, max_concepts=10)
        all_concepts.extend(chunk_concepts)

        # free memory immediately
        del chunk
        del summary
        del chunk_concepts

    # 2️⃣ Deduplicate and refine
    unique = {}
    for c in all_concepts:
        name = c.get("name", "").strip().lower()
        if name and name not in unique:
            unique[name] = c

    print("📘 Merging concepts...")
    combined_text = "\n".join(f"- {c['name']}: {c['description']}" for c in unique.values())

    # Extract final top 20–30 concepts from merged summaries
    final_concepts = extract_concepts_openai(combined_text, max_concepts=max_concepts)

    # Clean up disk
    for f in os.listdir("summaries"):
        os.remove(os.path.join("summaries", f))
    os.rmdir("summaries")

    return final_concepts

import spacy
from collections import Counter
from spacy.lang.en.stop_words import STOP_WORDS


# --- Configuration ---
use_openai = True

# If you set use_openai=True you'll be prompted to paste your OpenAI API key (kept only in-memory for the session)
from getpass import getpass
MODEL_NAME = 'gpt-4'  # change if you'd like (e.g., gpt-4o, gpt-4-turbo)

# --- Imports ---
import json
import re
from collections import Counter
from openai import OpenAI

client = OpenAI(api_key='sk-proj-......')

# def extract_concepts_openai(text, max_concepts=30):
#     """Ask the LLM to return a strict JSON with a list of concepts and short descriptions."""
#     prompt = (
#         "Extract the most important concepts from the following course notes. "
#         "Return ONLY valid JSON in the following format:\n"
#         '{"concepts": [{"name": "...", "description": "..."}, ...]}\n'
#         "Do not add extra commentary. Keep descriptions short (1–2 sentences).\n\n"
#         f"Notes:\n{text[:4000]}"
#     )

#     resp = client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant that extracts concepts from textbook/course notes."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.0,
#         max_tokens=2000
#     )

#     content = resp.choices[0].message.content.strip()

#     # Try to parse JSON
#     try:
#         j = json.loads(content)
#         return j.get("concepts", [])
#     except Exception:
#         # Try to recover JSON-like substring
#         m = re.search(r"\{.*\}", content, re.S)
#         if m:
#             try:
#                 j = json.loads(m.group(0))
#                 return j.get("concepts", [])
#             except Exception:
#                 pass
#         raise ValueError("Could not parse JSON from model output:\n" + content)


def extract_concepts_spacy(text, top_k=30):
    doc = nlp(text)

    candidates = []
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip().lower()

        # Filter short or uninformative chunks
        if len(chunk_text) <= 2:
            continue

        # Filter chunks that are just stopwords or pronouns
        if any(tok.text.lower() in STOP_WORDS or tok.pos_ in {"PRON", "DET"} for tok in chunk):
            continue

        # Optional: lemmatize
        lemmatized = " ".join([tok.lemma_ for tok in chunk if not tok.is_stop and tok.pos_ != "DET"])
        if lemmatized:
            candidates.append(lemmatized)

    counts = Counter(candidates)
    most = [c for c, _ in counts.most_common(top_k)]
    return [{"name": c, "description": ""} for c in most]

# --- Run extraction ---
if use_openai:
    print("Extracting concepts with OpenAI...")
    try:
        concepts = summarize_and_extract_concepts(text)
    except Exception as e:
        print("OpenAI extraction failed:", e)
        print("Falling back to spaCy extraction...")
        concepts = extract_concepts_spacy(course_notes)
else:
    print("Using spaCy fallback extraction")
    concepts = extract_concepts_spacy(course_notes)


print(f"\nExtracted {len(concepts)} concepts (sample):")
for c in concepts:
    print(f"- {c['name']}: {c['description']}")


# Build NetworkX graph
G = nx.Graph()

# Add nodes
for c in concepts:
    name = c['name'] if isinstance(c, dict) else str(c)
    desc = c.get('description','') if isinstance(c, dict) else ''
    G.add_node(name, description=desc)

# Edge creation by co-occurrence in sentences
for sent in sentences:
    present = []
    s_lower = sent.lower()
    for node in G.nodes():
        if node in s_lower:
            present.append(node)
    # if none matched by exact substring, try fuzzy containment by tokens
    if len(present) < 2:
        tokens = [t.lemma_.lower() for t in nlp(sent)]
        for node in list(G.nodes()):
            node_tokens = [t.lemma_.lower() for t in nlp(node)]
            if any(tok in tokens for tok in node_tokens):
                if node not in present:
                    present.append(node)

    for i in range(len(present)):
        for j in range(i+1, len(present)):
            u, v = present[i], present[j]
            if G.has_edge(u,v):
                G[u][v]['weight'] += 1
            else:
                G.add_edge(u, v, weight=1)

print('Graph built: ', G.number_of_nodes(), 'nodes,', G.number_of_edges(), 'edges')

def build_graph_from_concepts(concepts, sentences):
    # Build NetworkX graph
  G = nx.Graph()

  # Add nodes
  for c in concepts:
      name = c['name'] if isinstance(c, dict) else str(c)
      desc = c.get('description','') if isinstance(c, dict) else ''
      G.add_node(name, description=desc)

  # Edge creation by co-occurrence in sentences
  for sent in sentences:
      present = []
      s_lower = sent.lower()
      for node in G.nodes():
          if node in s_lower:
              present.append(node)
      # if none matched by exact substring, try fuzzy containment by tokens
      if len(present) < 2:
          tokens = [t.lemma_.lower() for t in nlp(sent)]
          for node in list(G.nodes()):
              node_tokens = [t.lemma_.lower() for t in nlp(node)]
              if any(tok in tokens for tok in node_tokens):
                  if node not in present:
                      present.append(node)

      for i in range(len(present)):
          for j in range(i+1, len(present)):
              u, v = present[i], present[j]
              if G.has_edge(u,v):
                  G[u][v]['weight'] += 1
              else:
                  G.add_edge(u, v, weight=1)


  return G;

  print('Graph built: ', G.number_of_nodes(), 'nodes,', G.number_of_edges(), 'edges')



def embedding_graph(G, embed_model):
  node_texts = []
  for n, d in G.nodes(data=True):
    desc = d.get('description','')
    node_texts.append(f"{n}. {desc}" if desc else n)

  corpus_embeddings = embed_model.encode(node_texts, convert_to_tensor=True)
  print('Computed', len(node_texts), 'node embeddings')




def upload_pdf_and_process_to_graph():
    # 1️⃣ Upload PDF
    uploaded = files.upload()
    if len(uploaded) == 0:
        raise ValueError("No file uploaded!")

    filename = list(uploaded.keys())[0]

    # 2️⃣ Extract text from PDF
    text = ""
    with open(filename, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"

    print(f"Uploaded PDF: {filename}")
    print("Length of extracted text (chars):", len(text))
    print("\n--- Preview ---\n", text[:1000])

    # 3️⃣ Process text to build graph JSON
    # Assumes these functions are already defined:
    # split_sentences, summarize_and_extract_concepts, build_graph_from_concepts, embedding_graph
    sentences = split_sentences(text)
    concepts = summarize_and_extract_concepts(text)
    G = build_graph_from_concepts(concepts, sentences)
    embedding_graph(G, embed_model)

    return G;



# Visualize with pyvis
net = Network(height='600px', width='100%', notebook=True, bgcolor='#ffffff')

for n, d in G.nodes(data=True):
    title = d.get('description','') or n
    net.add_node(n, label=n, title=title)

for u, v, d in G.edges(data=True):
    net.add_edge(u, v, value=d.get('weight',1), title=f"weight={d.get('weight',1)}")

net.repulsion(node_distance=200, central_gravity=0.33)
net.show('concept_graph.html')

# Display inline
IFrame('concept_graph.html', width='100%', height=600)

# Prepare node texts and compute embeddings
node_texts = []
for n, d in G.nodes(data=True):
    desc = d.get('description','')
    node_texts.append(f"{n}. {desc}" if desc else n)

corpus_embeddings = embed_model.encode(node_texts, convert_to_tensor=True)
print('Computed', len(node_texts), 'node embeddings')

# show first few nodes
for i,txt in enumerate(node_texts[:10]):
    print(i+1, '-', txt)



def get_top_nodes_for_question(question, top_k=5):
    q_emb = embed_model.encode(question, convert_to_tensor=True)
    hits = util.semantic_search(q_emb, corpus_embeddings, top_k=top_k)[0]
    results = []
    for hit in hits:
        idx = hit['corpus_id']
        score = hit['score']
        results.append((node_texts[idx], score))
    return results

def build_graph_context(top_nodes):
    # top_nodes: list of node_text strings
    context_lines = []
    for t,_ in top_nodes:
        # find node name (split before first dot if present)
        name = t.split('.')[0]
        desc = t
        context_lines.append(f"- {desc}")
    # Add simple nearby edges information
    context_lines.append('\nNearby edges among these nodes:')
    for i in range(len(top_nodes)):
        name_i = top_nodes[i][0].split('.')[0]
        for j in range(i+1, len(top_nodes)):
            name_j = top_nodes[j][0].split('.')[0]
            if G.has_edge(name_i, name_j):
                context_lines.append(f"- {name_i} <-> {name_j} (weight={G[name_i][name_j]['weight']})")
    return '\n'.join(context_lines)


def answer_question_with_graph(question, top_k=5, model_name=MODEL_NAME, temperature=0):
    top = get_top_nodes_for_question(question, top_k=top_k)
    context = build_graph_context(top)

    prompt = (
        'You are an educational tutor. Use ONLY the provided context (which contains key concepts and nearby relations extracted from course notes) to answer the student question. '
        'If the context does not contain enough information, say you do not have sufficient information rather than inventing facts.\n\n'
        f'Context:\n{context}\n\nQuestion: {question}\n\nAnswer:'
    )

    # --- Updated OpenAI v1.x API call ---
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a concise educational tutor who answers using given context."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=400
    )

    # new response format
    answer = resp.choices[0].message.content.strip()
    return answer

# Demo usage (try your own question)
q = 'what are the challenges of agentic systems'
print('Top nodes for question:')
print(get_top_nodes_for_question(q, top_k=5))
print('\nCalling the QA agent...')
ans = answer_question_with_graph(q, top_k=5)
print('\nAnswer (or context returned):\n', ans)

# Save graph to JSON
from networkx.readwrite import json_graph
jl = json_graph.node_link_data(G)
with open('concept_graph.json','w') as f:
    json.dump(jl, f, indent=2)
print('Saved concept_graph.json to working directory.')

# Provide a quick download link in Colab
from google.colab import files
# files.download('concept_graph.json')  # uncomment to download interactively

# !pip install jinja2

# ===============================
# 🌐 Educational Tutor Web App (Gradio)
# ===============================
import gradio as gr
import matplotlib.pyplot as plt
import networkx as nx
from sentence_transformers import util
from google.colab import files
import PyPDF2
import tempfile
from pyvis.network import Network
import os

# Global graph
G = nx.Graph()

# ---------- Functions ----------

def upload_pdf_and_process_to_graph(pdf_file):
    """Process uploaded PDF and build the concept graph."""
    global G
    # Read PDF
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text() + "\n"

    print("Length of extracted text (chars):", len(text))

    # Process text into graph
    sentences = split_sentences(text)
    concepts = summarize_and_extract_concepts(text)
    G = build_graph_from_concepts(concepts, sentences)
    embedding_graph(G, embed_model)

    # Return visualization
    return visualize_graph_file()


# def visualize_graph_html():
#     """Draw and return the educational concept graph as interactive HTML."""
#     global G
#     net = Network(height='600px', width='100%', notebook=False, bgcolor='#ffffff', directed=False)

#     # Add nodes
#     for n, d in G.nodes(data=True):
#         title = d.get('description', '') or n
#         net.add_node(n, label=n, title=title)

#     # Add edges
#     for u, v, d in G.edges(data=True):
#         net.add_edge(u, v, value=d.get('weight', 1), title=f"weight={d.get('weight', 1)}")

#     net.repulsion(node_distance=200, central_gravity=0.33)

#     # Save HTML to temp file
#     tmp_dir = tempfile.gettempdir()
#     html_path = os.path.join(tmp_dir, "concept_graph.html")
#     net.show(html_path, notebook=False)

#     # Read HTML content as string and return
#     with open(html_path, "r", encoding="utf-8") as f:
#         html_content = f.read()

#     return html_content


def visualize_graph_file():
    """Draw the educational concept graph and return the file path."""
    global G
    net = Network(height='600px', width='100%', notebook=False, bgcolor='#ffffff', directed=False)

    for n, d in G.nodes(data=True):
        title = d.get('description', '') or n
        net.add_node(n, label=n, title=title)

    for u, v, d in G.edges(data=True):
        net.add_edge(u, v, value=d.get('weight', 1), title=f"weight={d.get('weight', 1)}")

    net.repulsion(node_distance=200, central_gravity=0.33)

    # Save HTML to temp file
    tmp_dir = tempfile.gettempdir()
    html_path = os.path.join(tmp_dir, "concept_graph.html")
    net.show(html_path, notebook=False)

    return html_path  # Return path for gr.File



def get_answer(question):
    """Return both graph visualization and text answer for the question."""
    if G.number_of_nodes() == 0:
        return None, "Please upload a PDF first to build the concept graph."

    # Get top nodes & answer
    top_nodes = get_top_nodes_for_question(question, top_k=10)
    img_path = visualize_graph_file()
    answer = answer_question_with_graph(question, top_k=10)
    return img_path, answer

# ---------- Gradio Interface ----------

with gr.Blocks(title="Educational Tutor Graph") as demo:
    gr.Markdown("## 🧠 Educational Tutor Graph Demo")
    gr.Markdown("Upload a PDF to build the concept graph, then ask questions based on the concepts.")

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Upload PDF", file_types=['.pdf'])
            upload_button = gr.Button("Process PDF", variant="primary")
        with gr.Column(scale=2):
            graph_output = gr.File(label="Download/Open Knowledge Graph")

    with gr.Row():
        with gr.Column(scale=1):
            question = gr.Textbox(label="Enter your question", placeholder="e.g., What is Multi-Agent Pattern?")
            ask_button = gr.Button("Ask Tutor", variant="primary")
        with gr.Column(scale=2):
            answer_output = gr.Textbox(label="Answer", lines=4)

    # Connect buttons
    upload_button.click(fn=upload_pdf_and_process_to_graph, inputs=pdf_input, outputs=graph_output)
    ask_button.click(fn=get_answer, inputs=question, outputs=[graph_output, answer_output])

demo.launch(debug=True)

