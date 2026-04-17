import streamlit as st
import sys
import os

# Add src to python path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from retrieval import VectorStore, chunk_documents
from llm_client import GroqClient
from explain import ExplainabilityModule
import retrieval

# Page config
st.set_page_config(page_title="GenAI Explainability Demo", layout="wide")

st.title("Explainable RAG Demo")
st.markdown("Ask a question and see the answer along with the retrieval evidence and explainability graph.")

# Initialize components
@st.cache_resource
def init_resources():
    st.write("Initializing resources...")
    # Load chunks and build index
    chunks = chunk_documents(str(retrieval.DATA_DIR / "raw_cleaned.json"))
    store = VectorStore()
    store.build_index(chunks)
    
    llm = GroqClient()
    explainer = ExplainabilityModule()
    
    return store, llm, explainer

try:
    store, llm, explainer = init_resources()
    st.success("Resources loaded successfully!")
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()

# User Input
query = st.text_input("Enter your query:", "What is big data?")

if st.button("Get Answer"):
    if not query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            try:
                # 1. Retrieve
                results = store.search(query, top_k=5)
                
                # Remove duplicates for context
                seen = set()
                clean_results = []
                for r in results:
                    if r["chunk_id"] not in seen:
                        seen.add(r["chunk_id"])
                        clean_results.append(r)
                
                # 2. Generate Answer
                answer = llm.generate_answer(query, clean_results)
                
                # 3. Explain
                report = explainer.generate_explainability_report(
                    query=query,
                    answer=answer,
                    retrieved_chunks=clean_results
                )
                
                # Display Results
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Answer")
                    st.write(answer)
                    
                    st.subheader("Retrieved Evidence")
                    for item in report["evidence_nodes"]:
                        with st.expander(f"Chunk {item['chunk_id']} (Score: {item['relevance_score']})"):
                            st.write(item['excerpt'])

                with col2:
                    st.subheader("Explainability Graph")
                    fig = explainer.visualize_evidence()
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.write("No graph to display.")
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")
