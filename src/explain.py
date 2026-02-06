import networkx as nx
import matplotlib.pyplot as plt


class ExplainabilityModule:
    def __init__(self):
        self.graph = nx.DiGraph()

    def generate_explainability_report(
        self,
        query,
        answer,
        retrieved_chunks,
        min_score=0.65,
        top_k=5
    ):
        # -------- Filter + rank chunks --------
        filtered_chunks = [
            c for c in retrieved_chunks if c.get("score", 0) >= min_score
        ]

        filtered_chunks = sorted(
            filtered_chunks,
            key=lambda x: x["score"],
            reverse=True
        )[:top_k]

        max_score = max(
            (c["score"] for c in filtered_chunks),
            default=1.0
        )

        evidence_nodes = []
        for chunk in filtered_chunks:
            evidence_nodes.append({
                "chunk_id": chunk["chunk_id"],
                "relevance_score": round(chunk["score"], 3),
                "normalized_score": round(chunk["score"] / max_score, 3),
                "excerpt": chunk["text"][:200]
            })

        report = {
            "query": query,
            "answer": answer,
            "evidence_nodes": evidence_nodes,
            "explainability_method": "Top-K semantic retrieval with score normalization"
        }

        # -------- Build graph --------
        self.graph.clear()

        self.graph.add_node(
            "Question",
            label="User Question",
            color="orange"
        )

        self.graph.add_node(
            "Answer",
            label="LLM Answer",
            color="green"
        )

        self.graph.add_edge("Question", "Answer", label="answered_by")

        for chunk in filtered_chunks:
            cid = chunk["chunk_id"]
            score = round(chunk["score"], 3)

            self.graph.add_node(
                cid,
                label=f"{cid}\nScore: {score}",
                color="lightblue"
            )

            self.graph.add_edge(
                "Question",
                cid,
                label="retrieved_from"
            )

            self.graph.add_edge(
                cid,
                "Answer",
                weight=score,
                label="supports"
            )

        return report

    def visualize_evidence(self):
        if self.graph.number_of_nodes() == 0:
            print("[WARN] No graph data to visualize")
            return

        plt.figure(figsize=(10, 8))

        pos = nx.spring_layout(self.graph, seed=42)

        colors = [
            self.graph.nodes[n].get("color", "lightgrey")
            for n in self.graph.nodes
        ]

        labels = {
            n: self.graph.nodes[n].get("label", n)
            for n in self.graph.nodes
        }

        nx.draw(
            self.graph,
            pos,
            labels=labels,
            node_color=colors,
            node_size=2500,
            font_size=8,
            edge_color="gray"
        )

        edge_labels = nx.get_edge_attributes(self.graph, "label")
        nx.draw_networkx_edge_labels(
            self.graph,
            pos,
            edge_labels=edge_labels,
            font_size=7
        )

        plt.title("RAG Explainability Graph")
        plt.axis("off")
        return plt.gcf()
