# ENTAgent 🧠🔎🌐🧑‍⚕️
## AI Agents for Complex Knowledge in Otolaryngology

This repository accompanies the manuscript:

**ENTAgent: AI Agents for Complex Knowledge Otolaryngology**  
Tsz Kin Chan, Dinh-Nguyen, Nguyen, Ngoc-Duy Dinh  

---

## 📬 Contact

- **Prof. Ngoc-Duy Dinh** (Corresponding Author)  
  Email: ngocduydinh@cuhk.edu.hk  
  Address:
  Department of Biomedical Engineering,
  The Chinese University of Hong Kong,
  Shatin, N.T., Hong Kong, China


---

## 🔍 Overview

ENTAgent is a multi-agent large language model (LLM) framework for complex reasoning and decision support in **Otolaryngology (ENT)**.  
It integrates:

- Retrieval-Augmented Generation (RAG)  
- Self-Mixture of Agents (MoA)  
- Verbal Reinforcement Learning (Reflexion)  
- Tool-augmented research (PubMed, ArXiv, Google Search)

ENTAgent provides accurate, explainable, and self-correcting responses for:

- Short-answer clinical questions  
- Essay-type medical examinations  
- Large-scale board-style multiple-choice questions (MCQs)

---

## 📈 Key Features

- Domain-specific medical knowledge base (textbooks, guidelines, journals)  
- Self-aggregation of multiple LLM responses (Mixture-of-Agents)  
- Reflexion-based self-critique and iterative refinement  
- Autonomous literature search and evidence grounding  
- Explainable scoring and reasoning loop  

---

## 📊 Performance

| Task | Baseline LLM | ENTAgent |
|------|--------------|----------|
| Short Q&A Accuracy | 70% | **81.3% (+11.3%)** |
| Essay Length | 153 words | **425 words (2.78×)** |
| MCQ (BoardVitals) | 71% (Human Avg) | **75.5% (Statistically Significant)** |

---

## 🧠 System Architecture

1. **RAG Agent** – retrieves ENT knowledge from vector database (FAISS + Voyage embeddings)  
2. **Self-Mixture of Agents** – generates and aggregates multiple in-model answers  
3. **Evaluation Agent (Reflexion)** – scores, critiques, and plans revisions  
4. **Research Agent** – searches PubMed, ArXiv, and web sources  
5. **Revisor Agent** – refines and corrects answers iteratively  

---

## 📁 Planned Repository Structure

```
ENTAgents/
├── data/               # ENT knowledge base, exam datasets
├── rag/                # Vector database construction
├── agents/
│   ├── supervisor.py
│   ├── evaluation.py
│   ├── research.py
│   ├── revisor.py
│   └── moa.py
├── prompts/            # Reflexion and agent prompts
├── evaluation/         # MCQ scoring and statistics
├── figures/            # Paper figures and diagrams
└── README.md
```

---

## 🧪 Datasets

- Short Questions: RCPSC, Zalzal et al.  
- Essay Questions: Third M.B.B.S. ENT Examination  
- MCQs: BoardVitals (992 questions, 3 difficulty levels)

---

## 🎯 Applications

- ENT board exam preparation  
- Medical student education  
- Clinical decision support  
- Autonomous literature review  
- Rare disease knowledge retrieval  
- Future multimodal diagnosis (CT, MRI, endoscopy)


---

## 🧠 Citation

```bibtex
@article{chan2026entagent,
  title={ENTAgent: AI Agents for Complex Knowledge Otolaryngology},
  author={Chan, Tsz Kin, Nguyen, Dinh-Nguyen and Dinh, Ngoc-Duy},
  journal={Under Review},
  year={2026}
  note={https://www.medrxiv.org/content/10.1101/2025.01.01.25319863v1}
}
```

---

## 💰 Acknowledgements

Supported by the Research Grant Council of Hong Kong  
General Research Fund (Ref No. 14211223)

---

## 🔗 Repository

https://github.com/duydinhlab/ENTAgents




