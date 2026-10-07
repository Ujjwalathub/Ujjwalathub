<h1 align="center">Ujjwal Singh</h1>
<h3 align="center">Machine Learning Engineer · Deep Learning · NLP / LLMs · Applied ML Systems</h3>

<p align="center">
  3rd-year B.Tech CSE (AI/ML) · Lamrin Tech Skills University (2024–2028)<br/>
  I build ML systems end to end: data pipeline → model → explainability → API → deployed UI.
</p>

<p align="center">
  <a href="https://www.linkedin.com/in/ujjwal-singh-994305327/"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/></a>
  <a href="mailto:thesinghujjwal@gmail.com"><img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white"/></a>
  <a href="https://deep-sight-ews.vercel.app"><img src="https://img.shields.io/badge/Live_Demo-DeepSight_EWS-000?style=for-the-badge&logo=vercel&logoColor=white"/></a>
</p>

---

## 👤 About

I work on applied machine learning where the model has to ship: not just a notebook score, but a served endpoint, an explanation a non-ML person can read, and a UI on top.

My work so far spans:

- **Risk and tabular ML**: loan default early warning (FinBERT + LightGBM + SHAP), crime-threat classification, commodity forecasting (ARIMA + XGBoost)
- **Computer vision**: skin-lesion triage (EfficientNet-B0 5-fold ensemble + LightGBM meta-learner)
- **LLM / RAG / agents**: multilingual voice RAG over campus documents, multi-agent travel platform, LangChain ReAct concierge
- **Backend and deployment**: FastAPI, Docker Compose, Celery, Redis, WebSockets, Postgres, Vercel, Hugging Face Spaces

---

## 🎯 What I'm Looking For

ML engineering / data science internships and research collaborations: roles where I own a model from data to deployment.
**Contact:** [thesinghujjwal@gmail.com](mailto:thesinghujjwal@gmail.com)

---

## 🚀 Featured Projects

| # | Project | Domain | Core ML | Links |
|---|---------|--------|---------|-------|
| 1 | **DeepSight EWS** | Credit risk | FinBERT + LightGBM/XGBoost + SHAP | [Repo](https://github.com/Ujjwalathub/DeepSight-EWS) · [Live](https://deep-sight-ews.vercel.app) |
| 2 | **Derm-Referral AI (The Skin Guardian)** | Healthcare CV | EfficientNet-B0 ensemble + LightGBM | [Repo](https://github.com/Ujjwalathub/The-Skin-Guardian) |
| 3 | **KSP Crime Analytics API** | Public safety | Random Forest, Isolation Forest, DBSCAN | [Repo](https://github.com/Ujjwalathub/KSP-Crime-Analytics-API) |
| 4 | **Campus Voice Assistant** | RAG / speech | bge-m3 + reranker + Gemini + Whisper | [Repo](https://github.com/Ujjwalathub/Campus-Voice-Assistant) |
| 5 | **WanderOps** | Multi-agent LLM | Google ADK, RAG, CascadeFlow routing | [Repo](https://github.com/Ujjwalathub/WanderOps) |
| 6 | **AlphaSeries Price Tracker** | Forecasting | ARIMA + XGBoost + news sentiment | [Repo](https://github.com/Ujjwalathub/AlphaSeries--A-Price-Tracker-and-Predictor) |
| 7 | **Match-Mate (FIFA 2026 Companion)** | LLM agent app | LangChain ReAct, Groq Llama 3.3 70B | [Repo](https://github.com/Ujjwalathub/Match-Mate-During-FIFA) |
| 8 | **Logi-Slice Prime** | Logistics AI | Gemini risk analysis, IoT telemetry | [Repo](https://github.com/Ujjwalathub/Logi-Slice-Prime) |

---

### 1. DeepSight EWS: Loan Default Early Warning System
*Built for IDBI Innovation 2026* · [Repo](https://github.com/Ujjwalathub/DeepSight-EWS) · [Live demo](https://deep-sight-ews.vercel.app)

**Problem:** Lenders see default risk late. Structured fields (loan amount, DTI, FICO) miss distress signals buried in borrower narratives, and black-box scores are hard to act on.

**Approach:** dual-branch architecture.
- **Tabular branch:** engineered financial features → LightGBM / XGBoost.
- **Text branch:** FinBERT sentiment/distress features extracted from loan narratives.
- **Fusion and explainability:** both branches feed the default-probability model; SHAP gives global and per-borrower explanations (waterfall plots).

**Delivered:**
- FastAPI endpoints for portfolio summary, high-risk borrower alerts, and individual risk explanations; SHAP plots served as static assets
- React 19 + TanStack dashboard: portfolio health score, prioritized risk alerts with primary drivers, SHAP waterfall per borrower
- Modular ML pipeline: `data_preprocessing` → `feature_engineering` → `finbert_sentiment_analyzer` → `model_training` → `shap_explainer`

**Stack:** Python · FastAPI · scikit-learn · LightGBM · XGBoost · SHAP · FinBERT (Hugging Face) · React 19 · Vite · TailwindCSS v4 · Recharts · Vercel

---

### 2. Derm-Referral AI (The Skin Guardian): Skin Lesion Triage
[Repo](https://github.com/Ujjwalathub/The-Skin-Guardian)

**Problem:** Primary-care clinicians without dermatology training either over-refer (overloading specialists) or under-refer (delaying malignancy diagnosis).

**Approach:** two-stage pipeline.
1. **Vision stage:** EfficientNet-B0 (ImageNet pre-trained, fine-tuned) as a **5-fold out-of-fold ensemble** → CNN malignancy probability.
2. **Meta-learner:** LightGBM over the CNN score + clinical metadata (age, sex, site, lesion size, ABCDE scores) → final risk.
3. **Triage engine:** probability thresholds map to 🔴 RED (urgent biopsy), 🟡 YELLOW (tele-dermatology), 🟢 GREEN (routine monitoring).

**Engineering highlights:**
- Out-of-fold training to prevent leakage between stages
- Severe class imbalance (~1:50) handled with weighted sampling, `pos_weight`, stratified K-fold
- Trained on consumer-GPU limits: AMP, gradient accumulation (effective batch 64), 8-bit AdamW, CUDA→CPU fallback
- FastAPI service with Pydantic validation, health checks, CORS; experiment tracking with Weights & Biases

**Reported results (held-out validation, from repo README):** ROC-AUC 0.934 · sensitivity 0.978 · specificity 0.845. Precision is low by design (high recall prioritized for malignancy screening).

**Stack:** PyTorch · torchvision · LightGBM · Albumentations · FastAPI · Pydantic · pandas · scikit-learn · W&B · Docker

---

### 3. KSP Crime Analytics API
*Karnataka State Police Datathon 2026, Challenge 2* · [Repo](https://github.com/Ujjwalathub/KSP-Crime-Analytics-API)

**What it does:** backend for a crime analytics platform: executive stats, anomaly timeline, geospatial hotspots, criminal-network "kingpins", and threat prediction.

**Models:**
- **Random Forest classifier:** threat level (Low / Medium / High / Critical) from age, base risk score, connections; returns class probabilities
- **Isolation Forest:** unsupervised anomaly detection on crime time series
- **DBSCAN:** geospatial clustering for hotspots
- **Degree centrality:** ranks criminals by association-network position

**Endpoints:** `/api/v1/overview/stats` · `/analytics/timeline` · `/geospatial/hotspots` · `/network/kingpins` · `POST /predict-risk`
**Deployed** via Docker on Hugging Face Spaces · MIT licensed.
**Stack:** FastAPI · pandas · scikit-learn · Uvicorn · Docker

---

### 4. Campus Voice Assistant: Multilingual Voice RAG
[Repo](https://github.com/Ujjwalathub/Campus-Voice-Assistant)

**Problem:** Students and staff can't quickly find answers inside campus circulars, fee structures, and rulebooks, especially in Hindi/Hinglish.

**Pipeline:** PDF upload → **IBM Docling** layout-aware parsing (tables, multi-column) → chunking → **BAAI/bge-m3** multilingual embeddings → **Qdrant** → hybrid retrieval (dense + BM25) → **bge-reranker-v2-m3** cross-encoder → **Gemini 1.5 Flash** answer with source and page citations.

**Features:**
- English, Hindi, and Hinglish via text or voice (Faster-Whisper / Sarvam AI STT)
- Streaming SSE responses; multi-turn conversation memory
- Async ingestion with **Celery + Redis** so uploads never block the API
- Admin-key-protected document management; fully containerized (FastAPI, Postgres, Redis, Qdrant)

**Stack:** FastAPI · Qdrant · PostgreSQL · Celery · Redis · Docling · bge-m3 · Faster-Whisper · Gemini · Docker Compose

---

### 5. WanderOps: Multi-Agent Platform for Boutique Travel Agencies
*APAC Gen AI Academy (Google)* · [Repo](https://github.com/Ujjwalathub/WanderOps)

Three cooperating agents built on **Google Agent Development Kit**:
- **Customer agent:** RAG-grounded itineraries over proprietary travel documents, with persistent user memory (Hindsight + Firestore). Grounding tested with "trick questions" (e.g., accessibility, AC timing, cancellation rules)
- **Strategic agent:** natural-language → SQL over BigQuery public datasets via an MCP server for expansion analysis
- **Operations agent:** n8n workflows for daily operations checks, weather alerts, client notifications

**CascadeFlow routing:** simple queries go to Gemini Flash, complex ones to Pro, with automatic escalation when Flash output fails validation (design target: majority of requests served by Flash to cut cost).
**Stack:** Google ADK · Gemini 1.5 Flash/Pro · FastAPI · BigQuery · Firestore · ChromaDB / Vertex AI Vector Search · n8n · Docker Compose

---

### 6. AlphaSeries: Commodity Price Tracker and Predictor
[Repo](https://github.com/Ujjwalathub/AlphaSeries--A-Price-Tracker-and-Predictor)

Hybrid forecasting: **ARIMA** captures trend/seasonality, **XGBoost** models residuals using market indicators and **news sentiment**. Pipeline ingests market and news data (Alpha Vantage, RapidAPI), computes sentiment, trains the hybrid model, stores forecasts, and serves them through `/api/v1` endpoints (forecasts, metrics, news, market data, sentiment) to a React + TanStack + Tailwind dashboard.
**Stack:** FastAPI · SQLAlchemy (async SQLite) · statsmodels / ARIMA · XGBoost · React · Recharts

---

### 7. Match-Mate: FIFA 2026 Multilingual Matchday Companion
[Repo](https://github.com/Ujjwalathub/Match-Mate-During-FIFA)

- **AI concierge:** LangChain **ReAct agent** with 5 database-backed tools (menu, bathroom wait times, menu search, section info, nearest amenities), powered by Groq (Llama 3.3 70B) or Gemini; answers in 8 languages with seat-section context
- **Real-time layer:** simulated IoT bathroom wait times → Postgres → **Redis Pub/Sub** → **WebSocket** broadcast to the React UI
- **Ordering:** in-seat food orders with pickup codes and async fulfilment; n8n webhook alerts for congestion and order-ready events
- **Auth:** Better Auth microservice with role-based access

**Stack:** FastAPI · LangChain · Groq · PostgreSQL · Redis · WebSockets · React 19 · TypeScript · TanStack · Better Auth · n8n · Docker Compose

---

### 8. Logi-Slice Prime: Intelligent Logistics Backend
[Repo](https://github.com/Ujjwalathub/Logi-Slice-Prime)

Event-driven Node.js/TypeScript backend: live shipment telemetry over **Socket.IO**, an IoT simulator for development, geocoding via RapidAPI, and **Gemini-powered route risk analysis** producing security alerts and route-deviation notifications. Zod validation, Winston logging, Helmet security; Supabase and Docker setup included.
**Stack:** Node.js 20 · TypeScript · Express · Socket.IO · Google Generative AI · Zod · Docker

---

## 🛠️ Technical Skills

| Area | Tools |
|------|-------|
| **Languages** | Python, SQL, C++, Java, TypeScript, JavaScript |
| **ML / DL** | PyTorch, TensorFlow/Keras, scikit-learn, LightGBM, XGBoost, EfficientNet / transfer learning, Albumentations, OpenCV |
| **NLP / LLM** | Hugging Face (FinBERT), LangChain, RAG, embeddings (bge-m3), cross-encoder reranking, Gemini, Groq / Llama, Faster-Whisper, Google ADK |
| **Explainability / Eval** | SHAP, k-fold / out-of-fold validation, imbalance handling (weighted sampling, pos_weight), ROC-AUC / recall-oriented metrics |
| **Forecasting / Anomaly** | ARIMA, Isolation Forest, DBSCAN, sentiment features |
| **Data** | pandas, NumPy, Matplotlib, Plotly, PostgreSQL, SQLite, Qdrant, ChromaDB, Redis, BigQuery |
| **Backend / MLOps** | FastAPI, Pydantic, Celery, WebSockets, Docker / Compose, Weights & Biases, n8n, Git |
| **Frontend** | React, Next.js, TanStack Router/Query, Tailwind CSS, Recharts |
| **Deploy** | Vercel, Hugging Face Spaces, Google Cloud (Cloud Run, Firestore, BigQuery) |

---

## 📜 Certifications (Forage Job Simulations, Aug 2025)

| Program | Company | Focus | Date |
|---------|---------|-------|------|
| Data Science | **BCG X** | Problem framing, EDA & cleaning, feature engineering, modeling & evaluation, insights | Aug 9, 2025 |
| GenAI | **BCG X** | Data extraction, AI-powered financial chatbot | Aug 10, 2025 |
| Data Analytics | **Quantium** | Customer analytics, experimentation & uplift testing, commercial application | Aug 6, 2025 |
| Data Science | **British Airways** | Lounge-eligibility modeling (Heathrow T3), customer buying-behaviour prediction | Aug 3, 2025 |
| GenAI Powered Data Analytics | **Tata** | Risk profiling, delinquency prediction with AI, AI-driven collections strategy | Aug 3, 2025 |

---

## 🏆 Competitions

- **Pan-IIT Convolve 4.0** (participant)
- **IIT Guwahati GenAI** (semi-finalist)
- **IDBI Innovation 2026** (DeepSight EWS)
- **KSP Datathon 2026** (Crime Analytics, Challenge 2)
- **APAC Gen AI Academy, Cohort 3** (WanderOps)

---

## 📊 GitHub Stats

<p align="center">
  <img height="170" src="https://github-readme-stats.shion.dev/api?username=ujjwalathub&theme=dark&hide_border=true&include_all_commits=true&count_private=false"/>
  <img height="170" src="https://github-readme-stats.shion.dev/api/top-langs/?username=ujjwalathub&theme=dark&hide_border=true&include_all_commits=true&count_private=false&layout=compact"/>
</p>

---

<p align="center">
  <b>Open to ML engineering internships.</b> <a href="mailto:thesinghujjwal@gmail.com">thesinghujjwal@gmail.com</a> · <a href="https://www.linkedin.com/in/ujjwal-singh-994305327/">LinkedIn</a>
</p>
