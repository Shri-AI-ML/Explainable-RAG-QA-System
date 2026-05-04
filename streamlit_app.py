import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from src.retrieval.retrieval import run_retrieval
from src.llm_client import GroqClient

# ---------------- UI ----------------
st.set_page_config(page_title="RAG Demo", layout="wide")

st.title("Explainable RAG ")
st.markdown("Ask a question and see retrieved evidence.")

# INIT 
@st.cache_resource
def init_llm():
    return GroqClient()

llm = init_llm()

# INPUT 
query = st.text_input("Enter your query:", "What is artificial intelligence?")

# BUTTON 
if st.button("Get Answer"):
    if not query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Thinking..."):
            try:
                #  STEP 1: Retrieval
                results = run_retrieval(query)
                st.success("retrieval done ")   # DEBUG

                if not results:
                    st.warning("No results found.")
                    st.stop()

                # STEP 2: LLM
                try:
                    answer = llm.generate_answer(query, results)
                    st.success("llm done ")   # DEBUG
                except Exception as e:
                    st.error(f"LLM Error: {e}")
                    st.stop()

                #  DISPLAY          
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("Answer")
                    st.write(answer)

                with col2:
                    st.subheader("Retrieved Evidence")

                    for r in results:
                        with st.expander(f"Chunk {r.get('chunk_id', 'N/A')}"):
                            st.write(r.get("text", ""))

            except Exception as e:
                st.error(f"System Error: {e}")