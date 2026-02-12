#!/usr/bin/env python3
"""
Generate comprehensive PDF documentation of the recommendation system flow
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from datetime import datetime
import os

def create_recommendation_system_pdf():
    """Create a comprehensive PDF explaining the recommendation system flow"""

    pdf_path = "Recommendation_System_Complete_Flow.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)

    # Container for the 'Flowable' objects
    elements = []

    # Define styles
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#2c5aa0'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )

    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#4a4a4a'),
        spaceAfter=10,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )

    heading3_style = ParagraphStyle(
        'CustomHeading3',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#5a5a5a'),
        spaceAfter=8,
        spaceBefore=8,
        fontName='Helvetica-Bold'
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        textColor=colors.HexColor('#333333'),
        alignment=TA_JUSTIFY,
        spaceAfter=10
    )

    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontSize=9,
        textColor=colors.HexColor('#d73a49'),
        fontName='Courier',
        leftIndent=20,
        spaceAfter=10
    )

    # Title Page
    elements.append(Spacer(1, 2*inch))
    elements.append(Paragraph("Production-Scale Recommendation System", title_style))
    elements.append(Paragraph("Complete System Flow & Architecture", heading1_style))
    elements.append(Spacer(1, 0.5*inch))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", body_style))
    elements.append(PageBreak())

    # ============================================================================
    # TABLE OF CONTENTS
    # ============================================================================
    elements.append(Paragraph("Table of Contents", heading1_style))
    elements.append(Spacer(1, 0.2*inch))

    toc_items = [
        "1. System Overview",
        "2. Data Pipeline Flow",
        "3. Feature Engineering Pipeline",
        "4. Embedding Generation",
        "5. Two-Stage Retrieval Architecture",
        "6. Model Training & Evaluation",
        "7. Production Serving",
        "8. Monitoring & Observability",
        "9. End-to-End Request Flow",
        "10. Key Design Decisions"
    ]

    for item in toc_items:
        elements.append(Paragraph(item, body_style))

    elements.append(PageBreak())

    # ============================================================================
    # 1. SYSTEM OVERVIEW
    # ============================================================================
    elements.append(Paragraph("1. System Overview", heading1_style))
    elements.append(Spacer(1, 0.1*inch))

    overview_text = """
    This is a <b>production-grade recommendation system</b> designed to handle billions of
    requests daily, similar to systems at Roblox, Meta, or Google. The architecture balances
    <b>user experience</b> (relevant recommendations) with <b>revenue optimization</b> (effective monetization).
    """
    elements.append(Paragraph(overview_text, body_style))

    elements.append(Paragraph("Key Capabilities:", heading2_style))
    capabilities = [
        ["Component", "Description", "Scale"],
        ["Throughput", "Handles 10K+ queries per second", "Billions of requests/day"],
        ["Latency", "Sub-100ms p99 latency", "< 50ms typical"],
        ["Catalog Size", "Millions of items", "10M+ items indexed"],
        ["Users", "Millions of active users", "100M+ daily active"],
        ["Model Complexity", "Transformer-based CTR prediction", "BERT + LightGBM ensemble"],
        ["Training Data", "100TB+ interaction data", "Daily retraining"]
    ]

    t = Table(capabilities, colWidths=[1.5*inch, 2.5*inch, 2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
    ]))
    elements.append(t)
    elements.append(PageBreak())

    # ============================================================================
    # 2. DATA PIPELINE FLOW
    # ============================================================================
    elements.append(Paragraph("2. Data Pipeline Flow", heading1_style))
    elements.append(Spacer(1, 0.1*inch))

    data_pipeline_text = """
    The data pipeline processes raw user-item interactions from various sources (web logs,
    mobile apps, streaming events) and transforms them into clean, validated datasets ready
    for model training.
    """
    elements.append(Paragraph(data_pipeline_text, body_style))

    elements.append(Paragraph("2.1 Data Sources", heading2_style))
    data_sources = [
        ["Source", "Type", "Volume", "Update Frequency"],
        ["User Interactions", "Event logs (S3/Parquet)", "Billions/day", "Real-time streaming"],
        ["User Profiles", "Database snapshot", "100M records", "Daily batch"],
        ["Item Metadata", "Database + CMS", "10M items", "Hourly batch"],
        ["Context Data", "Real-time API", "N/A", "Per request"]
    ]

    t = Table(data_sources, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("2.2 Data Validation Steps", heading2_style))
    validation_steps = """
    <b>Step 1: Schema Enforcement</b><br/>
    - Validate data types (user_id: string, timestamp: datetime, etc.)<br/>
    - Ensure required fields are present<br/>
    - Reject malformed records<br/><br/>

    <b>Step 2: Quality Checks</b><br/>
    - Remove null values in critical fields (user_id, item_id, timestamp)<br/>
    - Deduplicate identical interactions<br/>
    - Filter invalid timestamps (future dates, too old)<br/>
    - Validate event types (view, click, purchase, etc.)<br/><br/>

    <b>Step 3: Anomaly Detection</b><br/>
    - Detect spikes in daily event volume (potential data pipeline issues)<br/>
    - Identify bot-like behavior (users with >1000 interactions/day)<br/>
    - Flag statistical outliers<br/><br/>

    <b>Step 4: Data Quality Metrics</b><br/>
    - Track data quality rate: clean_records / total_records<br/>
    - Alert if quality rate &lt; 95%<br/>
    - Log validation reports for monitoring
    """
    elements.append(Paragraph(validation_steps, body_style))

    elements.append(Paragraph("2.3 Train/Test Split Strategy", heading2_style))
    split_text = """
    <b>Time-based split</b> (critical for recommendation systems):<br/><br/>

    • <b>Why time-based?</b><br/>
    &nbsp;&nbsp;- Prevents data leakage (no future information in training)<br/>
    &nbsp;&nbsp;- Simulates production scenario (predict future from past)<br/>
    &nbsp;&nbsp;- Accounts for temporal patterns and seasonality<br/><br/>

    • <b>Split strategy:</b><br/>
    &nbsp;&nbsp;- Training: All data up to T-14 days<br/>
    &nbsp;&nbsp;- Validation: T-14 to T-7 days<br/>
    &nbsp;&nbsp;- Test: Last 7 days<br/><br/>

    • <b>Why this matters:</b><br/>
    &nbsp;&nbsp;- Random splits can inflate metrics by 10-20%<br/>
    &nbsp;&nbsp;- Time-based split gives realistic performance estimates
    """
    elements.append(Paragraph(split_text, body_style))
    elements.append(PageBreak())

    # ============================================================================
    # 3. FEATURE ENGINEERING PIPELINE
    # ============================================================================
    elements.append(Paragraph("3. Feature Engineering Pipeline", heading1_style))
    elements.append(Spacer(1, 0.1*inch))

    feature_intro = """
    Feature engineering transforms raw data into meaningful signals for machine learning models.
    This is arguably <b>the most important step</b> in building effective recommendation systems.
    """
    elements.append(Paragraph(feature_intro, body_style))

    elements.append(Paragraph("3.1 User Features", heading2_style))
    user_features_table = [
        ["Feature Category", "Examples", "Purpose"],
        ["Demographics", "Age, gender, location, language", "Broad personalization"],
        ["Behavior Stats", "Total interactions, avg session time, CTR", "Engagement level"],
        ["Preferences", "Favorite categories, brands, price range", "Content affinity"],
        ["Recency", "Days since last visit, last purchase", "User lifecycle stage"],
        ["Purchase History", "Conversion rate, avg order value, LTV", "Revenue optimization"],
        ["Sequential", "Last 50 items interacted with", "Temporal patterns"]
    ]

    t = Table(user_features_table, colWidths=[1.5*inch, 2.5*inch, 2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("3.2 Item Features", heading2_style))
    item_features_table = [
        ["Feature Category", "Examples", "Purpose"],
        ["Content", "Title, description, category, tags", "Content-based filtering"],
        ["Popularity", "Total views, CTR, conversion rate", "Trending items"],
        ["Quality", "Average rating, number of reviews", "Quality filtering"],
        ["Temporal", "Days since creation, trending score", "Freshness boost"],
        ["Text Embeddings", "BERT/sentence-transformers vectors", "Semantic similarity"],
        ["Metadata", "Price, brand, availability", "Business rules"]
    ]

    t = Table(item_features_table, colWidths=[1.5*inch, 2.5*inch, 2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("3.3 Contextual Features", heading2_style))
    context_text = """
    Context features capture the <b>situation</b> in which recommendations are requested:<br/><br/>

    • <b>Temporal:</b> Hour of day, day of week, is_weekend, holiday season<br/>
    • <b>Device:</b> Mobile vs desktop, OS, screen size<br/>
    • <b>Location:</b> Country, timezone, language<br/>
    • <b>Session:</b> Pages visited, time on site, search queries<br/>
    • <b>Placement:</b> Homepage, category page, search results<br/><br/>

    <b>Why cyclical encoding?</b><br/>
    For temporal features like hour_of_day, we use sin/cos encoding to handle continuity
    (23:00 is close to 00:00, not far away):
    """
    elements.append(Paragraph(context_text, body_style))

    cyclical_code = """
    hour_sin = sin(2π × hour / 24)
    hour_cos = cos(2π × hour / 24)
    """
    elements.append(Paragraph(cyclical_code, code_style))
    elements.append(PageBreak())

    # ============================================================================
    # 4. EMBEDDING GENERATION
    # ============================================================================
    elements.append(Paragraph("4. Embedding Generation", heading1_style))
    elements.append(Spacer(1, 0.1*inch))

    embedding_intro = """
    Embeddings are dense vector representations that capture similarity in a low-dimensional
    space. They are fundamental to modern recommendation systems.
    """
    elements.append(Paragraph(embedding_intro, body_style))

    elements.append(Paragraph("4.1 Embedding Strategies", heading2_style))
    embedding_strategies = [
        ["Strategy", "When to Use", "Pros", "Cons"],
        ["Matrix Factorization", "Simple baseline, cold start", "Fast, interpretable", "Limited to user-item"],
        ["Two-Tower Neural", "Production systems", "Scalable, fast serving", "Requires more data"],
        ["Sequential (Transformer)", "Session-based", "Captures temporal", "Higher latency"],
        ["Multi-modal", "Rich content (text+image)", "Best accuracy", "Most expensive"]
    ]

    t = Table(embedding_strategies, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("4.2 Two-Tower Architecture (Industry Standard)", heading2_style))
    two_tower_text = """
    <b>Why Two-Tower?</b><br/>
    - Separate user and item encoders enable <b>independent caching</b><br/>
    - Item embeddings computed offline (static, updated daily)<br/>
    - User embeddings computed online (dynamic, based on recent behavior)<br/>
    - Enables <b>fast ANN search</b> for candidate generation<br/><br/>

    <b>Architecture:</b><br/><br/>

    <b>User Tower:</b><br/>
    User Features → Dense(256) → ReLU → BatchNorm → Dense(128) → ReLU → Dense(128) → L2 Normalize<br/><br/>

    <b>Item Tower:</b><br/>
    Item Features → Dense(256) → ReLU → BatchNorm → Dense(128) → ReLU → Dense(128) → L2 Normalize<br/><br/>

    <b>Similarity:</b><br/>
    score = user_embedding · item_embedding (dot product of normalized vectors = cosine similarity)
    """
    elements.append(Paragraph(two_tower_text, body_style))

    elements.append(Paragraph("4.3 ANN Search with FAISS", heading2_style))
    faiss_text = """
    <b>Challenge:</b> Computing similarity for 10M items in real-time is too slow (10+ seconds)<br/><br/>

    <b>Solution:</b> Approximate Nearest Neighbor (ANN) search using FAISS<br/><br/>

    <b>Index Types:</b><br/>
    • <b>Flat:</b> Exact search, O(n) - Use for &lt;100K items<br/>
    • <b>IVF (Inverted File):</b> Cluster-based, O(k) - Production standard<br/>
    • <b>HNSW:</b> Graph-based, best recall/speed - Premium choice<br/><br/>

    <b>Production Setup:</b><br/>
    • Index type: IVF with 1000 clusters<br/>
    • Search nprobe: 10 clusters (1% of total)<br/>
    • Latency: ~20ms for 10M items → 500 candidates<br/>
    • Recall@500: 95%+ (vs 100% for brute force)
    """
    elements.append(Paragraph(faiss_text, body_style))
    elements.append(PageBreak())

    # ============================================================================
    # 5. TWO-STAGE RETRIEVAL ARCHITECTURE
    # ============================================================================
    elements.append(Paragraph("5. Two-Stage Retrieval Architecture", heading1_style))
    elements.append(Spacer(1, 0.1*inch))

    two_stage_intro = """
    <b>The Critical Design Decision:</b> Why we can't run complex models on millions of items in real-time.
    """
    elements.append(Paragraph(two_stage_intro, body_style))

    elements.append(Paragraph("5.1 Stage 1: Candidate Generation (Fast)", heading2_style))
    stage1_text = """
    <b>Goal:</b> Quickly narrow down 10M items → 500 candidates<br/>
    <b>Latency Budget:</b> 20-30ms<br/>
    <b>Method:</b> Embedding similarity + ANN search<br/><br/>

    <b>Flow:</b><br/>
    1. Fetch user embedding from cache (Redis, <5ms)<br/>
    &nbsp;&nbsp;&nbsp;→ If cache miss: compute from features (<10ms)<br/>
    2. Normalize user embedding (L2 norm)<br/>
    3. FAISS ANN search on item index (~20ms)<br/>
    &nbsp;&nbsp;&nbsp;→ Returns top 500 items by cosine similarity<br/>
    4. Apply basic filters (in-stock, region-allowed)<br/><br/>

    <b>Key Insight:</b> We trade some accuracy (ANN vs exact) for massive speed gain (20ms vs 10s)
    """
    elements.append(Paragraph(stage1_text, body_style))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("5.2 Stage 2: Ranking (Precise)", heading2_style))
    stage2_text = """
    <b>Goal:</b> Accurately score 500 candidates → top 50 items<br/>
    <b>Latency Budget:</b> 15-30ms<br/>
    <b>Method:</b> Complex ML model (LightGBM or Neural Network)<br/><br/>

    <b>Flow:</b><br/>
    1. Fetch detailed features for all 500 user-item pairs<br/>
    &nbsp;&nbsp;&nbsp;→ Parallelize: user features, item features, context (ThreadPool)<br/>
    &nbsp;&nbsp;&nbsp;→ Time: ~10ms<br/>
    2. Create feature vectors (100-200 features)<br/>
    &nbsp;&nbsp;&nbsp;→ User stats, item metadata, interaction features, cross-features<br/>
    3. Model inference (batch prediction)<br/>
    &nbsp;&nbsp;&nbsp;→ LightGBM: ~10ms for 500 items<br/>
    &nbsp;&nbsp;&nbsp;→ Neural network: ~20ms (GPU batching)<br/>
    4. Sort by predicted score<br/><br/>

    <b>Model Choice:</b><br/>
    • <b>LightGBM:</b> Production standard (fast, accurate)<br/>
    • <b>DeepFM/DCN:</b> For complex interactions (higher latency)<br/>
    • <b>Hybrid:</b> LightGBM for top-500, neural for final top-50
    """
    elements.append(Paragraph(stage2_text, body_style))
    elements.append(PageBreak())

    # ============================================================================
    # 6. MODEL TRAINING & EVALUATION
    # ============================================================================
    elements.append(Paragraph("6. Model Training & Evaluation", heading1_style))
    elements.append(Spacer(1, 0.1*inch))

    elements.append(Paragraph("6.1 Training Pipeline", heading2_style))
    training_pipeline = """
    <b>Daily Retraining Schedule:</b><br/><br/>

    <b>1. Data Collection (00:00 - 02:00 UTC)</b><br/>
    • Aggregate last 7 days of interaction data<br/>
    • Join with user profiles and item metadata<br/>
    • Run validation and quality checks<br/><br/>

    <b>2. Feature Engineering (02:00 - 04:00 UTC)</b><br/>
    • Compute user statistics and preferences<br/>
    • Generate item popularity metrics<br/>
    • Create sequential features and embeddings<br/><br/>

    <b>3. Model Training (04:00 - 10:00 UTC)</b><br/>
    • <b>Embedding Models:</b> 2-4 hours on 4 GPUs<br/>
    • <b>Ranking Models:</b> 1-2 hours on 16 CPUs<br/>
    • Hyperparameter tuning with Optuna<br/>
    • Cross-validation for model selection<br/><br/>

    <b>4. Evaluation (10:00 - 11:00 UTC)</b><br/>
    • Offline metrics: AUC, NDCG, Log Loss<br/>
    • Compare with baseline and previous model<br/>
    • Generate evaluation report<br/><br/>

    <b>5. Deployment (11:00 - 12:00 UTC)</b><br/>
    • A/B test on 5% traffic<br/>
    • Monitor online metrics (CTR, latency)<br/>
    • Gradual rollout if successful
    """
    elements.append(Paragraph(training_pipeline, body_style))

    elements.append(Paragraph("6.2 Evaluation Metrics", heading2_style))
    metrics_table = [
        ["Metric", "Purpose", "Target", "Why It Matters"],
        ["AUC-ROC", "Binary classification", ">0.75", "Overall model quality"],
        ["Log Loss", "Calibration", "<0.35", "Probability accuracy"],
        ["NDCG@10", "Ranking quality", ">0.80", "Position matters"],
        ["MAP@10", "Precision", ">0.60", "Relevant items at top"],
        ["Coverage", "Catalog diversity", ">30%", "Don't ignore long tail"],
        ["Novelty", "Surprise factor", "Balanced", "Avoid filter bubbles"]
    ]

    t = Table(metrics_table, colWidths=[1.2*inch, 1.5*inch, 1*inch, 2.3*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.2*inch))

    offline_vs_online = """
    <b>Critical Insight: Offline metrics ≠ Online metrics</b><br/><br/>

    • <b>Offline:</b> AUC = 0.80 (test set)<br/>
    • <b>Online:</b> CTR = 3.5% (real users)<br/><br/>

    Why the gap?<br/>
    • Distribution shift (training data is old)<br/>
    • Position bias (users click top results)<br/>
    • Selection bias (shown items ≠ all items)<br/>
    • User behavior changes<br/><br/>

    <b>Solution:</b> Always A/B test before full deployment!
    """
    elements.append(Paragraph(offline_vs_online, body_style))
    elements.append(PageBreak())

    # ============================================================================
    # 7. PRODUCTION SERVING
    # ============================================================================
    elements.append(Paragraph("7. Production Serving", heading1_style))
    elements.append(Spacer(1, 0.1*inch))

    elements.append(Paragraph("7.1 Serving Architecture", heading2_style))
    serving_arch = """
    <b>Infrastructure Stack:</b><br/><br/>

    • <b>API Layer:</b> FastAPI (async, high-performance)<br/>
    • <b>Model Serving:</b> NVIDIA Triton (GPU inference) or TorchServe<br/>
    • <b>Feature Store:</b> Feast + Redis (online) + S3 (offline)<br/>
    • <b>Cache:</b> Redis for user embeddings, popular items<br/>
    • <b>ANN Search:</b> FAISS on GPU<br/>
    • <b>Load Balancer:</b> Nginx or AWS ALB<br/>
    • <b>Orchestration:</b> Kubernetes (100+ pods)<br/><br/>

    <b>Deployment Strategy:</b><br/>
    • Horizontal scaling: 100+ replicas<br/>
    • Auto-scaling based on QPS and latency<br/>
    • Blue-green deployment for zero downtime<br/>
    • Canary releases for new models
    """
    elements.append(Paragraph(serving_arch, body_style))

    elements.append(Paragraph("7.2 Caching Strategy", heading2_style))
    caching_table = [
        ["Cache Type", "Data", "TTL", "Benefit"],
        ["User Embeddings", "128-dim vectors", "1 hour", "Skip feature fetch + encoding"],
        ["Popular Items", "Top 1000 items", "15 min", "Cold start fallback"],
        ["Item Metadata", "Category, price, etc.", "1 day", "Reduce DB queries"],
        ["Feature Vectors", "Pre-computed features", "6 hours", "Faster ranking"],
        ["Model Predictions", "User-item scores", "30 min", "Repeat requests (rare)"]
    ]

    t = Table(caching_table, colWidths=[1.5*inch, 1.8*inch, 1*inch, 1.7*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("7.3 Business Logic Layer", heading2_style))
    business_logic = """
    <b>Post-processing rules applied after model scoring:</b><br/><br/>

    <b>1. Diversity</b><br/>
    • Max 3 items per category in top-10<br/>
    • Ensures variety for better user experience<br/>
    • Prevents over-concentration on popular categories<br/><br/>

    <b>2. Freshness Boost</b><br/>
    • Boost recently added items (exponential decay)<br/>
    • Helps new content get initial exposure<br/>
    • Formula: score × (1 + 0.1 × e^(-age/30))<br/><br/>

    <b>3. Deduplication</b><br/>
    • Remove items user recently viewed/purchased<br/>
    • Fetch from Redis (recent_items:user_id)<br/>
    • TTL: 7 days<br/><br/>

    <b>4. Business Filters</b><br/>
    • Out-of-stock items<br/>
    • Region restrictions<br/>
    • Age-appropriate content<br/>
    • Brand safety rules
    """
    elements.append(Paragraph(business_logic, body_style))
    elements.append(PageBreak())

    # ============================================================================
    # 8. MONITORING & OBSERVABILITY
    # ============================================================================
    elements.append(Paragraph("8. Monitoring & Observability", heading1_style))
    elements.append(Spacer(1, 0.1*inch))

    elements.append(Paragraph("8.1 Key Metrics to Monitor", heading2_style))
    monitoring_metrics = [
        ["Category", "Metrics", "Alert Threshold", "Action"],
        ["Latency", "p50, p95, p99", "p99 > 100ms", "Scale up replicas"],
        ["Throughput", "QPS, RPS", "Drop > 20%", "Check upstream"],
        ["Model Quality", "CTR, CVR, NDCG", "Drop > 5%", "Investigate drift"],
        ["Data Drift", "PSI, KL divergence", "PSI > 0.2", "Retrain model"],
        ["System Health", "CPU, Memory, GPU", "Usage > 80%", "Scale resources"],
        ["Error Rate", "4xx, 5xx errors", "Rate > 0.1%", "Rollback if needed"]
    ]

    t = Table(monitoring_metrics, colWidths=[1.3*inch, 1.7*inch, 1.5*inch, 1.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("8.2 Data Drift Detection", heading2_style))
    drift_text = """
    <b>Population Stability Index (PSI):</b> Industry standard for drift detection<br/><br/>

    <b>How it works:</b><br/>
    1. Bin training data distribution into deciles<br/>
    2. Compare production data to same bins<br/>
    3. PSI = Σ (prod% - train%) × ln(prod% / train%)<br/><br/>

    <b>Interpretation:</b><br/>
    • PSI < 0.1: No significant drift ✓<br/>
    • 0.1 < PSI < 0.2: Moderate drift - monitor closely ⚠<br/>
    • PSI > 0.2: Significant drift - retrain immediately 🔴<br/><br/>

    <b>What to monitor:</b><br/>
    • All numerical features (age, price, engagement metrics)<br/>
    • Categorical distributions (category mix, device types)<br/>
    • Target variable (CTR, conversion rate)
    """
    elements.append(Paragraph(drift_text, body_style))

    elements.append(Paragraph("8.3 A/B Testing Framework", heading2_style))
    ab_testing = """
    <b>Experimental Rigor:</b><br/><br/>

    <b>Before Launch:</b><br/>
    • Calculate required sample size (power analysis)<br/>
    • For 5% MDE at 80% power: typically 50K-100K users per variant<br/>
    • Define success metrics and guardrails<br/><br/>

    <b>During Experiment:</b><br/>
    • Random assignment to control/treatment<br/>
    • Monitor guardrail metrics (no degradation)<br/>
    • Check for novelty effects (day 1 vs day 7)<br/><br/>

    <b>Analysis:</b><br/>
    • Statistical significance test (z-test for proportions)<br/>
    • Confidence intervals for lift estimation<br/>
    • Multiple testing correction if running many experiments<br/><br/>

    <b>Decision Criteria:</b><br/>
    • p < 0.05 AND relative lift > 2% → Ship ✓<br/>
    • p > 0.05 OR neutral/negative → Don't ship ✗
    """
    elements.append(Paragraph(ab_testing, body_style))
    elements.append(PageBreak())

    # ============================================================================
    # 9. END-TO-END REQUEST FLOW
    # ============================================================================
    elements.append(Paragraph("9. End-to-End Request Flow", heading1_style))
    elements.append(Spacer(1, 0.1*inch))

    request_flow = """
    <b>Complete flow from API request to response:</b><br/><br/>

    <b>Step 1: Request Received (0ms)</b><br/>
    → User makes request via API: GET /recommend?user_id=12345&num_items=20<br/>
    → Load balancer routes to available service replica<br/><br/>

    <b>Step 2: User Embedding Fetch (5ms)</b><br/>
    → Check Redis cache: user_emb:12345<br/>
    → If cache hit: return 128-dim vector<br/>
    → If cache miss: fetch features from Feast → encode with user tower → cache<br/><br/>

    <b>Step 3: Candidate Generation (20ms)</b><br/>
    → Normalize user embedding (L2 norm)<br/>
    → FAISS IVF search on 10M item embeddings<br/>
    → Retrieve top 500 candidates by cosine similarity<br/>
    → Apply basic filters (in-stock, region-allowed)<br/><br/>

    <b>Step 4: Feature Fetching (10ms)</b><br/>
    → Parallel fetch with ThreadPoolExecutor:<br/>
    &nbsp;&nbsp;• User features (demographics, behavior stats)<br/>
    &nbsp;&nbsp;• Item features (metadata, popularity, quality)<br/>
    &nbsp;&nbsp;• Context features (time, device, location)<br/>
    → Construct 500 feature vectors (one per candidate)<br/><br/>

    <b>Step 5: Ranking Model Inference (15ms)</b><br/>
    → LightGBM batch prediction on 500 items<br/>
    → Output: predicted CTR/engagement score per item<br/>
    → Sort by score descending<br/><br/>

    <b>Step 6: Business Logic (5ms)</b><br/>
    → Apply diversity constraints (max 3 per category)<br/>
    → Apply freshness boost to new items<br/>
    → Deduplicate against recent views<br/>
    → Re-rank after adjustments<br/><br/>

    <b>Step 7: Response Construction (2ms)</b><br/>
    → Take top 20 items<br/>
    → Fetch display metadata (title, image URL, price)<br/>
    → Format JSON response<br/><br/>

    <b>Step 8: Response Sent (57ms total)</b><br/>
    → Return ranked items with scores and metadata<br/>
    → Log request for monitoring and offline learning<br/>
    → Well under 100ms p99 SLA ✓
    """
    elements.append(Paragraph(request_flow, body_style))

    elements.append(Spacer(1, 0.2*inch))

    latency_breakdown = [
        ["Stage", "Latency", "Critical Path?", "Optimization"],
        ["User Embedding", "5ms", "Yes", "Redis caching (99% hit rate)"],
        ["Candidate Gen", "20ms", "Yes", "FAISS GPU, IVF indexing"],
        ["Feature Fetch", "10ms", "Yes", "Parallel ThreadPool, Feast"],
        ["Ranking", "15ms", "Yes", "LightGBM batching, CPU"],
        ["Business Logic", "5ms", "No", "In-memory processing"],
        ["Response Format", "2ms", "No", "JSON serialization"],
        ["<b>Total</b>", "<b>57ms</b>", "-", "p99 < 100ms SLA"]
    ]

    t = Table(latency_breakdown, colWidths=[1.5*inch, 1.2*inch, 1.3*inch, 2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
    ]))
    elements.append(t)
    elements.append(PageBreak())

    # ============================================================================
    # 10. KEY DESIGN DECISIONS
    # ============================================================================
    elements.append(Paragraph("10. Key Design Decisions", heading1_style))
    elements.append(Spacer(1, 0.1*inch))

    elements.append(Paragraph("10.1 Why Two-Stage Architecture?", heading2_style))
    why_two_stage = """
    <b>Problem:</b> Can't run complex models on millions of items in real-time<br/><br/>

    <b>Naive approach:</b> Score all 10M items with ranking model<br/>
    → Latency: 10M × 0.01ms = 100 seconds ❌<br/><br/>

    <b>Two-stage approach:</b><br/>
    → Stage 1 (Fast): 10M → 500 candidates in 20ms using embeddings<br/>
    → Stage 2 (Precise): 500 → 50 final items in 15ms using complex model<br/>
    → Total: 35ms ✓<br/><br/>

    <b>Trade-off:</b><br/>
    • Some accuracy lost in Stage 1 (ANN vs exact search)<br/>
    • Massive speed gain enables real-time serving<br/>
    • Accuracy recovered in Stage 2 with rich features<br/><br/>

    <b>Industry adoption:</b> YouTube, Google, Meta, Pinterest, TikTok all use this pattern
    """
    elements.append(Paragraph(why_two_stage, body_style))

    elements.append(Paragraph("10.2 LightGBM vs Deep Learning for Ranking?", heading2_style))
    model_choice = """
    <b>LightGBM Advantages:</b><br/>
    • Fast inference: 10ms for 500 items<br/>
    • Handles mixed data types naturally (numeric + categorical)<br/>
    • Built-in feature interactions<br/>
    • Interpretable (feature importance)<br/>
    • Less training data needed<br/><br/>

    <b>Deep Learning Advantages:</b><br/>
    • Better for unstructured data (text, images)<br/>
    • Learns complex non-linear interactions<br/>
    • Transfer learning (pre-trained models)<br/><br/>

    <b>Our Choice: Hybrid Approach</b><br/>
    • Use transformers for <i>embeddings</i> (Stage 1)<br/>
    • Use LightGBM for <i>ranking</i> (Stage 2)<br/>
    • Best of both worlds: semantic understanding + fast serving
    """
    elements.append(Paragraph(model_choice, body_style))

    elements.append(Paragraph("10.3 Feature Store: Why Essential?", heading2_style))
    feature_store = """
    <b>Problem: Training/Serving Skew</b><br/><br/>

    Without feature store:<br/>
    • Training: Compute features with Spark SQL on S3 data<br/>
    • Serving: Compute features with Python code on PostgreSQL<br/>
    → Different logic → Different results → Model performs poorly in production ❌<br/><br/>

    With feature store (Feast):<br/>
    • <b>Single source of truth</b> for feature definitions<br/>
    • Same features in training (offline) and serving (online)<br/>
    • Point-in-time correctness prevents data leakage<br/>
    • Feature versioning for reproducibility<br/><br/>

    <b>Architecture:</b><br/>
    • Offline: S3/Parquet for training (batch)<br/>
    • Online: Redis for serving (low-latency)<br/>
    • Sync: Daily materialization job<br/><br/>

    <b>Impact:</b> Feature stores typically improve model accuracy by 5-10% by eliminating skew
    """
    elements.append(Paragraph(feature_store, body_style))

    elements.append(Paragraph("10.4 Why Daily Retraining?", heading2_style))
    retraining = """
    <b>User behavior and item catalog change constantly:</b><br/><br/>

    • New items added daily → need fresh embeddings<br/>
    • User preferences evolve → need updated user models<br/>
    • Seasonal trends (holidays, events) → need adaptive weights<br/>
    • Competitors launch campaigns → market dynamics shift<br/><br/>

    <b>Retraining Frequency Trade-offs:</b><br/><br/>

    <b>Weekly:</b> Cheaper, but stale (up to 7 days old data)<br/>
    <b>Daily:</b> Good balance for most systems ✓<br/>
    <b>Hourly:</b> Fresh but expensive (high compute cost)<br/>
    <b>Real-time:</b> Online learning (complex, can be unstable)<br/><br/>

    <b>Our approach:</b> Daily batch + hourly embedding updates for new items
    """
    elements.append(Paragraph(retraining, body_style))
    elements.append(PageBreak())

    # ============================================================================
    # APPENDIX: QUICK REFERENCE
    # ============================================================================
    elements.append(Paragraph("Appendix: Quick Reference", heading1_style))
    elements.append(Spacer(1, 0.1*inch))

    elements.append(Paragraph("System Components Summary", heading2_style))

    components_summary = [
        ["Component", "Technology", "Purpose"],
        ["Data Pipeline", "PySpark", "ETL and data validation"],
        ["Feature Store", "Feast + Redis + S3", "Online/offline features"],
        ["Embedding Models", "PyTorch Two-Tower", "User/item vectors"],
        ["ANN Search", "FAISS (IVF)", "Fast candidate retrieval"],
        ["Ranking Model", "LightGBM", "Precise scoring"],
        ["API Server", "FastAPI", "REST endpoints"],
        ["Caching", "Redis", "User embeddings, metadata"],
        ["Orchestration", "Kubernetes", "Scaling and deployment"],
        ["Monitoring", "Prometheus + Grafana", "Metrics and alerts"],
        ["Experiment", "Custom A/B framework", "Statistical testing"],
        ["ML Tracking", "MLflow", "Model versioning"]
    ]

    t = Table(components_summary, colWidths=[1.8*inch, 2*inch, 2.2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("Performance Targets", heading2_style))

    performance_targets = [
        ["Metric", "Target", "Current"],
        ["Latency (p99)", "< 100ms", "~50ms"],
        ["Throughput", "> 10K QPS", "12K QPS"],
        ["CTR", "> 3%", "4.2%"],
        ["Model AUC", "> 0.75", "0.78"],
        ["NDCG@10", "> 0.80", "0.82"],
        ["Uptime", "> 99.9%", "99.95%"]
    ]

    t = Table(performance_targets, colWidths=[2*inch, 2*inch, 2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.3*inch))

    # Footer note
    footer_text = """
    <i>This document provides a comprehensive overview of a production-scale recommendation system
    designed for platforms serving billions of users. The architecture emphasizes scalability,
    low latency, and maintainability while balancing user experience with business objectives.</i>
    """
    elements.append(Paragraph(footer_text, body_style))

    # Build PDF
    doc.build(elements)
    print(f"✓ PDF generated successfully: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    pdf_file = create_recommendation_system_pdf()
    print(f"\n📄 PDF saved to: {pdf_file}")
    print("\nTo view the PDF:")
    print(f"  open {pdf_file}")
