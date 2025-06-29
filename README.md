# CodonNavigator-NN-Bioinformatics
Codon_navigatorIdentifies Open Reading Frames (ORFs) in DNA sequences from FASTA files
# Codonavigator 🧬⚡  
**Open Reading Frame (ORF) Detection via Handcrafted Neural Networks**  

*A deterministic, training-free neural net that decodes DNA sequences with pure NumPy logic — no black boxes, just biological intuition.*  

[![Streamlit Demo](https://img.shields.io/badge/🔬-Live_Demo-2EA44F?style=flat&logo=Streamlit)](https://your-streamlit-app-url.herokuapp.com)  
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)  
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)  

---

## 🔍 Overview  
Codonavigator identifies **Open Reading Frames (ORFs)** in DNA sequences using a custom Feedforward Neural Network (FFNN) built from scratch with:  
- **Zero model training** (weights designed via "napkin math")  
- **Full interpretability** (heatmaps reveal decision logic)  
- **Biologist-friendly outputs** (protein translations, GC content, and visualizations)  

---

## 🛠️ Features  
| Feature | Implementation |  
|---------|----------------|  
| **ORF Detection** | 3-2-1 FFNN with hardcoded biological rules |  
| **Start/Stop Highlighting** | Color-coded codon annotations |  
| **Protein Translation** | Biopython-backed standard genetic code |  
| **GC Content Analysis** | Real-time % calculation |  
| **Base Composition** | Interactive Matplotlib pie charts |  
| **Weight Visualization** | Seaborn heatmaps for model transparency |  

---

## 🚀 Quick Start    
   ```bash  
   pip install streamlit numpy biopython matplotlib seaborn
   streamlit run orf_finder.py   
