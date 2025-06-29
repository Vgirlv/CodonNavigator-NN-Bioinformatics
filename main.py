import streamlit as st 
from Bio import SeqIO, Seq
import numpy as np
import io
import matplotlib.pyplot as plt
import base64
import seaborn as sns

st.set_page_config(page_title=" Open Reading Frames Finder", layout="wide")

st.markdown("""
    <style>
            
        body {
            background-color: #0d1117;
            color: #c9d1d9;
        }
        .stApp {
            background-color: #0d1117;
        }
        .st-bq, .st-bo, .st-bx, .st-bw, .st-bu {
            background-color: #161b22 !important;
        }
        .css-18ni7ap, .css-1d391kg, .css-1v0mbdj {
            color: #c9d1d9 !important;
        }
        .st-bf, .stButton>button {
            background-color: #238636;
            color: white;
            border: none;
        }
        .stButton>button:hover {
            background-color: #2ea043;
            color: white;
        }
        h1, h2, h3, h4 {
            color: #58a6ff;
        }
    </style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "intro"

if st.session_state.page == "intro":
    st.title("🚀 CodonNavigator - Interpretable FFNN Model for ORF Identification from FASTA")
    st.markdown("""
        ### 🌌 About This Project
       Codonavigator is a lightweight, interpretable bioinformatics tool designed to detect Open Reading Frames (ORFs) from FASTA-format DNA sequences using a heuristic neural network approach. Instead of relying on model training, it uses a manually crafted Feedforward Neural Network (FFNN) based on codon logic and napkin math to identify start and stop codons efficiently. The tool highlights functional codons within each ORF, calculates GC content, and provides protein translation for biologically meaningful insights. It also features a weight heatmap for neural logic interpretability using Seaborn and a pie chart to visualize nucleotide composition. Built with Python, Streamlit, NumPy, Biopython, and Matplotlib, Codonavigator offers an intuitive interface for genomic exploration without the complexity of traditional ML models.


        **Architecture** used here is a small **Feedforward Neural Network (FFNN)**:
        - Input layer: 3 units (Start / Stop / Neutral codon vector)
        - Hidden layer: 2 units (manually tuned)
        - Output: 1 binary unit (for ORF detection)

        ---

        ### 📖 User Guide
        1. Upload a valid `.fasta` file. 
        2. Codons will be scanned and analyzed.
        3. ORFs will be detected, highlighted, and translated into protein.
        4. You’ll get a **pie chart** of the nucleotide composition and **GC content**.
                
        ---

        ### 👥 Target Users
        - Curious students  
        - Molecular biologists  
        - Computational biology researchers  
        - Educators and instructors  
        - Aspiring data scientists  
        - Recruiters and portfolio reviewers

        Ready to explore your genes like never before? 🌱🔬
    """)
    if st.button("✨ Let’s Start!"):
        st.session_state.page = "main"
        st.rerun()

elif st.session_state.page == "main":
    st.title("🧬 ORF Detection and Protein Translation via Handcrafted Neural Network")
    st.write("Upload a `.fasta` DNA file and CodoNavigator will find Open Reading Frames in your sequences")

    uploaded_file = st.file_uploader("Upload your FASTA file", type=["fasta", "fa"])

    if uploaded_file:
        fasta_io = io.StringIO(uploaded_file.getvalue().decode("utf-8"))

        codon_map = {
            "ATG": [1, 0, 0],
            "TAA": [0, 1, 0],
            "TAG": [0, 1, 0],
            "TGA": [0, 1, 0],
        }

        def codon_to_vector(codon):
            return np.array(codon_map.get(codon, [0, 0, 1]))

        def forward_nn(codon_vectors):
            W1 = np.array([[2, -1], [-1, 2], [0, 0.5]])
            b1 = np.zeros((1, 2))
            W2 = np.array([[1], [1]])
            b2 = np.array([[0]])
            A = []
            activated = False
            for vec in codon_vectors:
                z1 = vec @ W1 + b1
                a1 = np.maximum(0, z1)
                z2 = a1 @ W2 + b2
                y = 1 if activated else 0
                if vec[0] == 1:
                    activated = True
                    y = 1
                elif vec[1] == 1 and activated:
                    y = 1
                    activated = False
                A.append(y)
            return A

        def find_orfs(seq):
            seq = seq.upper().replace("\n", "")
            codons = [seq[i:i+3] for i in range(0, len(seq)-2, 3)]
            codon_vecs = [codon_to_vector(c) for c in codons]
            output = forward_nn(codon_vecs)
            orfs = []
            inside = False
            start = 0
            for i, val in enumerate(output):
                if val == 1 and not inside:
                    inside = True
                    start = i * 3
                elif val == 0 and inside:
                    end = i * 3
                    orfs.append((start, end))
                    inside = False
            if inside:
                orfs.append((start, len(seq)))
            return orfs

        def highlight_codons(seq):
            colored_seq = ""
            for i in range(0, len(seq) - 2, 3):
                codon = seq[i:i+3]
                if codon == "ATG":
                    colored_seq += f"<span style='color:lime;font-weight:bold'>{codon}</span>"
                elif codon in ["TAA", "TAG", "TGA"]:
                    colored_seq += f"<span style='color:tomato;font-weight:bold'>{codon}</span>"
                else:
                    colored_seq += codon
            return colored_seq

        def gc_content(seq):
            seq = seq.upper()
            g = seq.count('G')
            c = seq.count('C')
            return round(100 * (g + c) / len(seq), 2) if len(seq) > 0 else 0

        def plot_base_pie(seq, orf_label):
            seq = seq.upper()
            counts = {'A': seq.count('A'), 'T': seq.count('T'), 'G': seq.count('G'), 'C': seq.count('C')}
            labels = counts.keys()
            sizes = counts.values()
            colors = ['#FF00FF','#00FFFF','#39FF14','#FFD700']
            explode = (0.05, 0.05, 0.05, 0.05)
            fig, ax = plt.subplots()
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', startangle=140, wedgeprops=dict(width=0.5, edgecolor='w'))
            ax.legend(wedges, labels, title="Bases", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            ax.axis('equal')
            ax.set_title(f'Nucleotide Composition: {orf_label}')
            st.pyplot(fig)

        def plot_weight_heatmap(W1):
            nucleotides = ['Start Codon', 'Stop Codon', 'Neutral Codon', 'Other']
            hidden_units = ['Hidden Unit 1', 'Hidden Unit 2']
            heatmap_data = np.zeros((4, 2))
            heatmap_data[:3, :] = W1
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", center=0,
                        xticklabels=hidden_units, yticklabels=nucleotides)
            ax.set_title("First-Layer Weights: Codon → Hidden Unit")
            ax.set_xlabel("Hidden Units")
            ax.set_ylabel("Codon Types")
            st.pyplot(fig)

        summary_lines = []

        for record in SeqIO.parse(fasta_io, "fasta"):
            seq_id = record.id
            sequence = str(record.seq)
            orfs = find_orfs(sequence)
            st.subheader(f"🔬 {seq_id}")
            if orfs:
                for i, (start, end) in enumerate(orfs):
                    orf_seq = sequence[start:end]
                    gc = gc_content(orf_seq)
                    protein = Seq.Seq(orf_seq).translate(to_stop=True)
                    st.write(f"ORF {i+1}: Positions {start} - {end} ({end - start} bp)")
                    st.write(f"GC Content: {gc}%")
                    st.write(f"Protein Translation: {protein}")
                    highlighted = highlight_codons(orf_seq)
                    st.markdown(highlighted, unsafe_allow_html=True)
                    plot_base_pie(orf_seq, f"{seq_id}_ORF{i+1}")
                    summary_lines.append(f"\nORF {i+1}: {start}-{end} ({end-start} bp)")
                    summary_lines.append(f"GC Content: {gc}%")
                    summary_lines.append(f"Protein: {protein}")
                    summary_lines.append(f"Sequence: {orf_seq}\n")
            else:
                st.write("❌ No ORFs found.")

        st.subheader("🧠 FFNN Weight Visualization")
        W1 = np.array([[2, -1], [-1, 2], [0, 0.5]])
        plot_weight_heatmap(W1)

        if summary_lines:
            st.markdown("---")
            st.subheader("📝 Export Summary")
            summary_text = "\n".join(summary_lines)
            b64 = base64.b64encode(summary_text.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="orf_summary.txt">📄 Download Summary File</a>'
            st.markdown(href, unsafe_allow_html=True)
