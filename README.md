# Enterprise Email Analysis System

A sophisticated, production-ready email analysis platform combining machine learning, agentic AI, RAG (Retrieval-Augmented Generation), and multi-agent collaboration for intelligent phishing detection and email processing.

[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Angular](https://img.shields.io/badge/Angular-18.2-DD0031?logo=angular)](https://angular.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791?logo=postgresql)](https://www.postgresql.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-000000)](https://ollama.ai/)


## 🎥 Demo Video

[![Watch the video](https://img.youtube.com/vi/8xxYOO3r-44/maxresdefault.jpg)]
(https://youtu.be/8xxYOO3r-44)


## Current Status - Fully Implemented Features

### 🚀 Key Automation Features

**Zero-Touch Email Processing:**
- ✅ **Automatic Email Fetching**: Continuously imports emails from MailPit
- ✅ **Automatic Phishing Detection**: ML ensemble analyzes every email (82.10% accuracy)
- ✅ **Automatic Summarization**: LLM generates concise summaries for all emails
- ✅ **Automatic Badge Assignment**: AI categorizes emails into 8 intelligent groups
- ✅ **Automatic CTA Extraction**: Detects and lists action items from email content
- ✅ **Automatic Reply Generation**: Creates 3 quick reply drafts per email
- ✅ **Automatic RAG Enrichment**: Adds wiki context and employee information
- ✅ **Automatic Filtering**: Daily Inbox Digest groups emails by category
- ✅ **Real-time Updates**: SSE streams all changes to frontend instantly

**User Experience:**
- **10,000 emails processed** automatically on startup
- **Badge-based filtering** - Click to view RISK, MEETING, VIP, etc.
- **Action item tracking** - Never miss important CTAs
- **Smart replies ready** - Formal, Friendly, Brief tones pre-generated
- **Context-aware** - Wiki knowledge and employee data auto-injected
- **3-5 seconds per email** - From arrival to fully enriched
- **24/7 continuous operation** - Background services always running

---

### Core Capabilities

#### 1. Advanced Phishing Detection (82.10% Accuracy)
- **Ensemble ML Model**: 3 models with OR-based voting strategy
  - Naive Bayes: 76.50% accuracy
  - Random Forest: 71.70% accuracy
  - Fine-tuned LLM: 59% accuracy
- **Overall Performance**: 82.10% accuracy, 81.90% F1-score
- **Training Data**: 10,000 real-world emails (60% phishing, 40% legitimate)
- **Source**: Kaggle Phishing and Legitimate Emails Dataset

#### 2. Agentic Team Collaboration (3 Specialized Workflows)
Multi-agent debate system where specialized teams analyze complex emails through 3-round discussions with dedicated tools and workflows:

**Fully Implemented Specialized Teams:**

1. **Fraud Investigation Unit** (`fraud_workflow.py`)
   - **Agents**: Fraud Detection Specialist, Forensic Analyst, Legal Advisor, Security Director
   - **Tools**: Transaction analysis, Risk assessment, Investigation tools, IP geolocation
   - **Patterns**: Doer-Checker validation, Debate, Iterative deepening (3 iterations)
   - **Capabilities**: Phishing investigation, BEC detection, Account compromise analysis
   - **Use Cases**: Suspicious transactions, fraud detection, security threat analysis

2. **Compliance & Regulatory Affairs** (`compliance_workflow.py`)
   - **Agents**: Compliance Officer, Legal Counsel, Internal Auditor, Regulatory Liaison
   - **Tools**: OFAC sanctions (free), AML/KYC, Regulatory assessment, Policy enforcement, Entity resolver
   - **Patterns**: Multi-agent debate, Compliance verification
   - **Capabilities**: Company compliance checks, Sanctions screening, Regulatory assessment
   - **Use Cases**: Vendor verification, Client onboarding, Regulatory compliance

3. **Investment Research Team** (`investment_workflow.py`)
   - **Agents**: Financial Analyst, Research Analyst, Filings Analyst, Investment Advisor
   - **Tools**: SEC EDGAR (10-K/10-Q), Google search (Serper), Web scraping (Browserless), Calculator
   - **Patterns**: Research synthesis, Data aggregation
   - **Capabilities**: SEC filing analysis, Company research, Financial metrics calculation
   - **Use Cases**: Investment analysis, Due diligence, Market research

**Additional Teams** (Defined in `agentic_teams.py` - Basic implementation):
- Credit Risk Committee
- Wealth Management Advisory
- Corporate Banking Team
- Operations & Quality

**Multi-Agent Collaboration Patterns:**

1. **Doer-Checker Pattern** (Primary - Used in Fraud workflow):
   - **Doer Agent**: Performs initial analysis and makes findings
   - **Checker Agent**: Validates findings, challenges assumptions, identifies gaps
   - **Iteration**: Up to 3 rounds of refinement between doer and checker
   - **Validation**: Ensures high-quality, cross-verified results
   - **Use Case**: Fraud detection, high-stakes decision making

2. **Multi-Round Debate Pattern** (Available in all workflows):
   - **Round 1**: Initial assessment - each agent provides their specialized perspective
   - **Round 2**: Challenge and debate - agents critique and question each other's views
   - **Round 3**: Final synthesis - agents defend positions or concede based on evidence
   - **Decision**: Final decision maker synthesizes all viewpoints with concrete action items
   - **Use Case**: Complex analysis requiring multiple perspectives

3. **Research Synthesis Pattern** (Investment workflow):
   - **Parallel Research**: Multiple agents gather data from different sources simultaneously
   - **Data Aggregation**: Findings combined and cross-referenced
   - **Expert Analysis**: Each agent contributes domain expertise
   - **Final Report**: Comprehensive synthesis with citations and metrics

**Real-time Updates**: Server-Sent Events stream each interaction to frontend for live visualization of agent collaboration

#### 3. RAG System (Retrieval-Augmented Generation)
Contextual enrichment combining internal knowledge base and employee directory:

**Components:**
- **Vector Store**: ChromaDB with sentence-transformers embeddings (all-MiniLM-L6-v2)
- **Wiki Integration**: OtterWiki knowledge base with automatic indexing
- **Employee Directory**: Real-time lookup for sender information (mobile, phone, department, location)
- **Confidence Scoring**: Relevance-based enrichment with similarity thresholds
- **Async Processing**: Non-blocking enrichment with graceful fallbacks

**Enrichment Capabilities:**
- Contextual wiki knowledge injection
- Employee profile augmentation
- Department and location context
- Contact information retrieval

#### 4. LLM Integration with Automatic Fallback
Dual-mode LLM system supporting both cloud and offline operation:

**Primary (Ollama - Offline)**:
- Model: Phi3 (lightweight, efficient)
- Use case: Email summarization, badge detection, CTA extraction
- Fully offline operation

**Secondary (OpenAI - Optional)**:
- Model: GPT-4o-mini (fast, cost-effective)
- Use case: Agentic team discussions (higher quality)
- Automatic fallback to Ollama if unavailable

**Smart Fallback Logic**:
- Automatically uses Ollama when:
  - No `.env` file exists
  - `OPENAI_API_KEY` not set or empty
  - API key contains placeholder values
  - OpenAI API fails at runtime (network, rate limits)
- Zero configuration needed for offline mode
- Seamless cloud-to-local switching

#### 5. AI-Powered Email Processing
Comprehensive LLM-based email analysis with automatic enrichment:

- **Smart Summarization**: Concise email summaries (2-3 sentences) generated by Ollama phi3
- **Badge Detection**: Auto-categorization into 8 types for intelligent filtering
  - MEETING - Calendar events and scheduling
  - RISK - Phishing/security threats
  - EXTERNAL - Emails from external sources
  - AUTOMATED - Auto-generated notifications
  - VIP - Important contacts
  - FOLLOW_UP - Requires action or response
  - NEWSLETTER - Marketing/bulk emails
  - FINANCE - Financial transactions
- **Call-to-Action (CTA) Extraction**: Automatic detection and listing of actionable items
  - Extracts specific actions requested in emails (e.g., "Review the attached proposal", "Schedule a meeting")
  - Displayed as a dedicated list in email details
  - Helps prioritize email responses
  - Integrated with badge system for FOLLOW_UP classification
- **Quick Reply Drafts**: 3 AI-generated response tones
  - **Formal**: Professional business tone
  - **Friendly**: Casual but respectful tone
  - **Brief**: Concise, to-the-point responses
- **Automatic Processing**: Background analyzer continuously processes all emails
  - LLM processing triggers automatically for new emails
  - No manual intervention required
  - Batch processing support for efficiency

#### 6. Background Processing & Real-Time Updates
Fully automated email monitoring and analysis pipeline:

**Email Fetcher Service** (`email-fetcher/`):
- Continuous polling from MailPit SMTP server
- Automatically imports new emails into PostgreSQL database
- Configurable fetch limit and interval
- Runs in background (Docker: unless-stopped)
- Preserves email metadata (sender, recipient, subject, body, timestamps)

**Email Analyzer Service** (`email-analyzer/`) - Intelligent Auto-Processing:
- **Automatic ML Analysis**: Runs phishing detection workflows on all unprocessed emails
  - Ensemble model prediction (Naive Bayes + Random Forest + Fine-tuned LLM)
  - Confidence scoring (0-100%)
  - Risk indicator extraction
  - Phishing type classification
- **Automatic LLM Processing**: Generates AI-powered insights for every email
  - Summary generation (2-3 sentences)
  - Badge assignment (8 categories)
  - Call-to-action extraction
  - Quick reply draft generation (3 tones)
- **Automatic RAG Enrichment**: Contextual data augmentation
  - Wiki knowledge injection (OtterWiki integration)
  - Employee directory lookup (phone, department, location)
  - Relevance-based confidence scoring
- **Batch Processing**: Processes 100 emails per batch
- **Configurable Delays**: Prevents resource exhaustion
- **Continuous Operation**: Runs 24/7 in background mode
- **Status**: All processing happens automatically without user intervention

**Server-Sent Events (SSE)**: Real-time updates streamed to frontend
- Email processing status (fetching, analyzing, enriching)
- Workflow results (ML predictions, confidence scores)
- Agentic team discussions (round-by-round debate updates)
- Analysis progress (percentage completion)
- Task tracking with unique task IDs
- Auto-reconnection on connection loss
- <100ms latency for real-time experience

#### 7. Enterprise Angular Web Application
Modern Angular 18 application with professional UI and state management:

**Pages:**

1. **Dashboard** (`/`) - Real-time Statistics & Monitoring
   - Total emails processed count
   - Phishing detection rate (percentage)
   - Risk email count with alerts
   - Average confidence score
   - Interactive charts (Chart.js)
   - Quick action buttons
   - Real-time updates via SSE

2. **Mailbox** (`/mailbox`) - Comprehensive Email Management
   - Paginated email list with search/filter
   - Email details modal with full content
   - ML workflow results display (confidence scores, risk indicators)
   - Quick reply integration (3 tones)
   - Badge indicators for categorization
   - Call-to-action list display
   - Email enrichment data (wiki + employee info)
   - Team assignment interface

3. **Daily Inbox Digest** (`/daily-inbox-digest`) - **Intelligent Email Organization**
   - **Automatic Badge-Based Filtering**: Emails auto-categorized into 8 groups
     - **MEETING** - Calendar events, scheduling requests
     - **RISK** - Phishing attempts, security threats (highlighted in red)
     - **EXTERNAL** - Communications from outside the organization
     - **AUTOMATED** - System notifications, auto-generated emails
     - **VIP** - Messages from important contacts
     - **FOLLOW_UP** - Emails requiring action or response
     - **NEWSLETTER** - Marketing emails, bulk communications
     - **FINANCE** - Transaction notifications, financial updates
   - **Smart Grouping**: Each badge category displays email count
   - **Priority Sorting**: Risk emails prominently displayed at top
   - **One-Click Navigation**: Click badge to filter emails
   - **Visual Indicators**: Color-coded badges for quick scanning
   - **Call-to-Action Summary**: Aggregated action items across all emails
   - **Workflow Integration**: Shows phishing detection results inline
   - **Zero Manual Tagging**: All categorization done automatically by LLM

4. **Agentic Teams** (`/agentic-teams`) - Real-time Multi-Agent Collaboration
   - Live team discussion visualization
   - Round-by-round debate display
   - Agent responses with role indicators
   - Final decision synthesis
   - Action items extraction
   - SSE-powered real-time updates
   - Team selection interface

**Technology:**
- **Framework**: Angular 18.2.0
- **State Management**: NgRx 18.1.1 with effects and selectors
- **UI**: Bootstrap 5.3.8 with Material Design
- **Charts**: Chart.js 4.5.1
- **Icons**: Material Icons, Font Awesome
- **Real-time**: Server-Sent Events integration
- **Markdown**: Marked 17.0.0 for rich text rendering

**UI Features:**
- Centralized state management with NgRx
- Real-time SSE updates across all pages
- Responsive design (mobile-friendly)
- Badge-based automatic categorization
- Workflow result panels with confidence scores
- Quick reply integration (3 AI-generated tones)
- Entity-based normalized state
- Advanced filtering and search
- Infinite scroll support
- Loading states and error handling

#### 8. Pluggable Tools Framework (13 Specialized Tools)
Extensible tool architecture for agentic workflows with graceful fallbacks:

**Available Tools:**
- **Investigation Tools** - OFAC sanctions screening (free, no API key), business verification, fraud detection
- **Sanctions Tools** - Multi-database sanctions screening (OFAC SDN, UN, EU, UK, PEP lists)
- **AML Tools** - AML/KYC compliance checks, risk classification, customer due diligence
- **Policy Compliance Tools** - Internal policy enforcement, compliance rule checking
- **Regulatory Tools** - Regulatory requirement assessment, compliance validation
- **Risk Tools** - IP geolocation & threat intelligence, VPN/Proxy/TOR detection
- **Transaction Tools** - Transaction pattern analysis, anomaly detection
- **SEC Tools** - SEC EDGAR filing retrieval (10-K, 10-Q reports)
- **Entity Resolver** - Entity name standardization, fuzzy matching, duplicate detection
- **Search Tools** - Internet search integration (Serper API with fallback to mock)
- **Browser Tools** - Website scraping and content extraction (Browserless integration)
- **Calculator Tools** - Financial metric calculations, ratio analysis
- **Serper Search Plugin** - Advanced search plugin for MCP integration

**Architecture:**
- **BaseTool** class with standardized interface
- **ToolRegistry** for dynamic discovery and registration
- **Automatic Fallbacks**: All external APIs optional with mock/free alternatives
- **Graceful Degradation**: System continues working when APIs unavailable

**API Keys (All Optional):**
- `OPENAI_API_KEY` - Cloud LLM (fallback: Ollama)
- `SERPER_API_KEY` - Web search (fallback: mock results)
- `BROWSERLESS_API_KEY` - JS scraping (fallback: HTTP)
- `SEC_API_KEY` - SEC filings (fallback: free EDGAR)
- `IPGEOLOCATION_API_KEY` - IP threat intel (fallback: mock)
- `ABSTRACTAPI_EMAIL_KEY` - Email validation (fallback: format check)

📖 **For detailed tool documentation with usage examples and API integration, see [Tools Framework Guide](docs/TOOLS_FRAMEWORK.md)**

#### 9. Specialized Workflows with Multi-Agent Patterns

**Fraud Detection Workflow** (`fraud_workflow.py`):
- **Agents**: 4 specialized fraud investigators
  - Fraud Detection Specialist (Doer)
  - Forensic Analyst (Checker)
  - Legal Advisor
  - Security Director
- **Primary Pattern**: Doer-Checker with iterative refinement (up to 3 iterations)
- **Alternative Pattern**: Multi-agent debate when needed
- **Capabilities**: Phishing investigation, BEC detection, Account compromise analysis
- **Tools**: Transaction analysis, Risk assessment, Investigation tools, IP geolocation
- **Analysis Methods**:
  - `analyze_email_for_fraud()` - Email-level fraud detection
  - `_investigate_phishing()` - Phishing-specific investigation
  - `_investigate_bec()` - Business Email Compromise detection
  - `_investigate_account_compromise()` - Account takeover detection

**Compliance Workflow** (`compliance_workflow.py`):
- **Agents**: 4 specialized compliance officers
  - Compliance Officer
  - Legal Counsel
  - Internal Auditor
  - Regulatory Liaison
- **Primary Pattern**: Multi-agent debate
- **Capabilities**: Company compliance checks, OFAC sanctions screening, AML/KYC verification, Regulatory assessment
- **Tools**: Regulatory tools, AML tools, Sanctions tools (free OFAC), Entity Resolver, Policy tools
- **Data Sources**: OpenSanctions (free, no API key), AML databases, Regulatory frameworks

**Investment Research Workflow** (`investment_workflow.py`):
- **Agents**: 4 investment research specialists
  - Financial Analyst
  - Research Analyst
  - Filings Analyst
  - Investment Advisor
- **Primary Pattern**: Research synthesis with parallel data gathering
- **Data Sources**: SEC EDGAR (10-K, 10-Q filings), Google search (via Serper), Company websites (via Browserless)
- **Tools**: Search tools, Browser tools, SEC tools, Calculator tools
- **Capabilities**: Financial metrics calculation, SEC filing analysis, Company research, Market analysis

**Orchestration & Infrastructure**:
- **File**: `backend/agentic_teams.py`
- **Class**: `AgenticTeamOrchestrator`
- **Team Detection**: Rule-based + LLM-based suggestion
- **Pattern Support**: Doer-Checker, Multi-round debate, Research synthesis
- **Real-time Updates**: SSE progress updates for all patterns
- **Tool Integration**: Dynamic tool registry with 13 specialized tools

📖 **For complete workflow documentation with API examples and best practices, see [Agentic Workflows Guide](docs/AGENTIC_WORKFLOWS.md)**

### Architecture

#### Docker Services (14 Containers)

| Service | Technology | Port | Purpose |
|---------|-----------|------|---------|
| **postgres** | PostgreSQL 15 | 5432 | Main database |
| **ollama** | Ollama (Phi3, Tinyllama) | 11434 | Local LLM inference |
| **backend** | FastAPI + Python 3.11 | 8000 | REST API & orchestration |
| **frontend-angular** | Angular 18 + Nginx | 80/443 | Modern web UI with NgRx |
| **mailpit** | MailPit | 1025/8025 | SMTP + POP3 + Web UI |
| **otterwiki** | OtterWiki | 9000 | Internal knowledge base (RAG) |
| **otterwiki-init** | OtterWiki | - | Wiki initialization (ephemeral) |
| **email-seeder** | Python | - | Dataset loader (10,000 emails) |
| **email-fetcher** | Python | - | Continuous email fetching |
| **email-analyzer** | Python | - | Background ML processing |
| **model-naive-bayes** | Python + scikit-learn | 8002 | Naive Bayes model API (76.50%) |
| **model-random-forest** | Python + scikit-learn | 8004 | Random Forest model API (71.70%) |
| **model-fine-tuned-llm** | Python + Transformers | 8006 | Fine-tuned LLM model API (59%) |
| **employee-directory** | FastAPI | 8100 | Employee lookup service (HR API) |

#### Database Schema

**Tables:**
- `emails` - Email content, metadata, summaries, badges, CTAs, quick replies, enrichment data
- `workflow_results` - ML model predictions with confidence scores
- `workflow_configs` - Workflow definitions and status
- `teams` - Agentic team configurations
- `team_agents` - Individual agent definitions with personalities

**Migrations:**
- Liquibase-managed schema versioning
- Automatic migration on startup
- Changelog: `backend/db/changelog/`

📖 **For complete database schema documentation, see [Database Schema Guide](docs/database-schema.md)**

#### API Endpoints (46 Endpoints)

**System:**
- `GET /` - API information and welcome message
- `GET /health` - Health check endpoint
- `GET /api/mailpit/stats` - MailPit server statistics

**Email Management:**
- `GET /api/emails` - List all emails with summaries, badges, quick replies
- `GET /api/emails/{email_id}` - Get email details with workflow results
- `POST /api/emails/fetch` - Fetch new emails from MailPit
- `POST /api/emails/update-bodies` - Update email bodies from MailPit
- `DELETE /api/emails/{email_id}` - Delete specific email

**Phishing Detection & Workflows:**
- `POST /api/emails/{email_id}/process` - Run ML phishing detection workflows
- `POST /api/emails/process-all` - Batch process all unprocessed emails
- `GET /api/workflows` - List available workflow configurations
- `GET /api/emails/{email_id}/workflow-status` - Get workflow execution status

**LLM Processing & Enrichment:**
- `POST /api/emails/{email_id}/process-llm` - Generate summary, badges, CTAs, quick replies
- `POST /api/emails/process-all-llm` - Batch LLM processing for all emails
- `POST /api/emails/{email_id}/enrich` - RAG enrichment (wiki + employee directory)

**Inbox Management & Statistics:**
- `GET /api/inbox-digest` - **Intelligent Email Digest with Automatic Filtering**
  - Returns emails automatically grouped by AI-assigned badges (8 categories)
  - Includes email count per category
  - Pre-filtered for instant display in Daily Inbox Digest page
  - Supports priority sorting (RISK emails first)
- `GET /api/daily-summary` - Aggregated daily statistics
  - Total emails processed
  - Breakdown by badge category
  - Phishing detection summary
  - Action items count
- `GET /api/statistics` - Overall system metrics
  - Total emails in database
  - Phishing detection rate
  - Average confidence scores
- `GET /api/dashboard/enriched-stats` - Enhanced statistics with enrichment data
  - Wiki enrichment coverage
  - Employee directory match rate
  - RAG performance metrics

**Agentic Teams - Team Management:**
- `GET /api/agentic/teams` - List all available teams
- `GET /api/agentic/teams/{team_key}/config` - Get team configuration
- `POST /api/agentic/teams/{team_key}/config` - Update team configuration
- `POST /api/agentic/teams` - Create new team
- `PUT /api/agentic/teams/{team_key}` - Update existing team
- `DELETE /api/agentic/teams/{team_key}` - Delete team (soft delete)

**Agentic Teams - Email Analysis:**
- `POST /api/agentic/emails/{email_id}/process` - Start multi-agent team discussion
- `GET /api/agentic/tasks/{task_id}` - Poll discussion progress and results
- `POST /api/agentic/direct-query` - Direct query to agentic system
- `POST /api/agentic/task/{task_id}/chat` - Continue chat with task
- `GET /api/agentic/emails/{email_id}/team` - Get team assignment for email
- `GET /api/agentic/simulate-discussion` - Simulate team discussion
- `POST /api/emails/{email_id}/suggest-team` - LLM-based team suggestion
- `POST /api/emails/{email_id}/assign-team` - Manually assign email to team

**Tools Management:**
- `GET /api/teams/{team_key}/tools` - Get tools available for specific team
- `POST /api/tools/test` - Test tool execution
- `GET /api/tools/registry` - Get complete tool registry
- `GET /api/tools/registry/stats` - Get tool registry statistics
- `GET /api/tools/search/capability/{capability}` - Search tools by capability
- `GET /api/tools/search/category/{category}` - Search tools by category
- `GET /api/tools/{tool_name}` - Get specific tool details
- `POST /api/tools/registry/assign` - Assign tools to team
- `GET /api/tools/{tool_name}/readiness` - Check tool readiness status

**Background Services (Email Fetcher):**
- `POST /api/fetcher/start` - Start continuous email fetching
- `POST /api/fetcher/stop` - Stop email fetcher
- `GET /api/fetcher/status` - Check email fetcher status

**Real-time Events:**
- `GET /api/events` - Server-Sent Events stream for real-time updates

## How It Works - Automatic Email Analysis Pipeline

The system operates as a fully automated email intelligence platform. Here's the complete workflow:

### 1. Email Ingestion (Automatic)
```
MailPit SMTP Server (10,000 pre-seeded emails)
         ↓
Email Fetcher Service (continuous polling)
         ↓
PostgreSQL Database (email storage with metadata)
```
- Email Seeder loads 10,000 real phishing/legitimate emails on startup
- Email Fetcher continuously polls MailPit and imports new emails
- All emails stored with complete metadata (sender, recipient, subject, body, timestamps)

### 2. ML-Based Phishing Detection (Automatic)
```
Unprocessed Emails in Database
         ↓
Email Analyzer Service (batch: 100 emails)
         ↓
Ensemble ML Model (3 models, OR-based voting)
  ├─ Naive Bayes API (Port 8002) → 76.50% accuracy
  ├─ Random Forest API (Port 8004) → 71.70% accuracy
  └─ Fine-tuned LLM API (Port 8006) → 59% accuracy
         ↓
Workflow Results (confidence score, risk indicators, phishing type)
         ↓
Database Update (is_phishing flag, workflow_results table)
```
- Automatic ensemble prediction: If ANY model flags email → marked as phishing
- Overall accuracy: 82.10%, F1-score: 81.90%
- Inference time: <100ms per email

### 3. LLM-Powered Email Enrichment (Automatic)
```
Emails with ML Results
         ↓
Ollama LLM Service (phi3 model, local inference)
         ↓
Parallel Processing:
  ├─ Summary Generation (2-3 sentences)
  ├─ Badge Detection (8 categories: MEETING, RISK, EXTERNAL, etc.)
  ├─ Call-to-Action Extraction (actionable items list)
  └─ Quick Reply Drafts (Formal, Friendly, Brief tones)
         ↓
Database Update (summary, badges, call_to_actions, quick_reply_drafts)
```
- Processing time: 2-3 seconds per email
- Fully offline operation (no API keys required)
- Automatic fallback: OpenAI → Ollama if configured

### 4. RAG-Based Context Enrichment (Automatic)
```
Emails with LLM Processing
         ↓
Parallel RAG Enrichment:
  ├─ OtterWiki (ChromaDB vector search)
  │   └─ Sentence-transformers embeddings (all-MiniLM-L6-v2)
  │   └─ Relevant wiki articles injection
  └─ Employee Directory API (Port 8100)
      └─ Sender lookup (phone, mobile, department, location)
         ↓
Enriched Email Data (wiki context + employee info)
         ↓
Database Update (enriched_data, wiki_enriched, phone_enriched flags)
```
- Enrichment time: <500ms per email
- Confidence-based filtering (similarity threshold)
- Graceful degradation if services unavailable

### 5. Automatic Badge-Based Categorization
```
Fully Processed Emails
         ↓
Badge Assignment by LLM:
  ├─ MEETING (calendar/scheduling keywords)
  ├─ RISK (phishing detected OR security threats)
  ├─ EXTERNAL (external domain analysis)
  ├─ AUTOMATED (auto-generated patterns)
  ├─ VIP (important sender detection)
  ├─ FOLLOW_UP (action required analysis)
  ├─ NEWSLETTER (bulk/marketing patterns)
  └─ FINANCE (transaction/financial keywords)
         ↓
Daily Inbox Digest (automatic grouping by badge)
```
- **Zero manual tagging required**
- Automatic filtering in Daily Inbox Digest page
- Priority sorting (RISK emails at top)
- Real-time badge updates via SSE

### 6. User Interface Display (Real-time)
```
Angular Frontend (NgRx State Management)
         ↓
Real-time SSE Updates:
  ├─ Dashboard (statistics, charts)
  ├─ Mailbox (email list with all enrichments)
  ├─ Daily Inbox Digest (badge-grouped emails)
  │   ├─ Click badge → filter emails
  │   ├─ View email → see summary, CTAs, quick replies
  │   └─ Risk emails highlighted in red
  └─ Agentic Teams (multi-agent analysis on-demand)
```

### 7. Agentic Workflow Analysis (On-Demand / Manual Trigger)
For complex emails requiring deep analysis, users can trigger specialized multi-agent workflows:

```
User Selects Email → Assign to Specialized Team
         ↓
Team Selection (Automatic LLM Suggestion or Manual):
  ├─ Fraud Investigation Unit
  ├─ Compliance & Regulatory Affairs
  └─ Investment Research Team
         ↓
Multi-Agent Collaboration:
  ├─ Pattern 1: Doer-Checker (Fraud)
  │   ├─ Doer Agent: Initial analysis
  │   ├─ Checker Agent: Validation & challenge
  │   └─ Iteration: Up to 3 rounds of refinement
  │
  ├─ Pattern 2: Multi-Round Debate (Compliance)
  │   ├─ Round 1: Initial perspectives (4 agents)
  │   ├─ Round 2: Challenge & critique
  │   └─ Round 3: Final synthesis
  │
  └─ Pattern 3: Research Synthesis (Investment)
      ├─ Parallel data gathering (SEC, Web, Search)
      ├─ Expert analysis from each agent
      └─ Comprehensive report generation
         ↓
Tool Execution (13 Specialized Tools):
  ├─ Investigation: OFAC screening, fraud detection
  ├─ Sanctions: Multi-database screening
  ├─ AML: Compliance checks
  ├─ SEC: Filing retrieval (10-K, 10-Q)
  ├─ Search: Internet research (Serper API)
  ├─ Browser: Website scraping
  ├─ Risk: IP geolocation, threat intel
  └─ [+6 more tools]
         ↓
Real-time Streaming (SSE):
  ├─ Round-by-round agent responses
  ├─ Tool execution results
  ├─ Debate progression
  └─ Live updates in Agentic Teams page
         ↓
Final Decision & Action Items:
  ├─ Decision maker synthesizes all findings
  ├─ Concrete action items extracted
  ├─ Comprehensive analysis report
  └─ Stored in database for future reference
```

**Agentic Workflow Characteristics:**
- **Trigger**: Manual user action (not automatic)
- **Processing Time**: 30-60 seconds (3 rounds, multiple agents)
- **Real-time Visibility**: Live SSE streaming of agent discussions
- **LLM Requirements**: Prefers OpenAI (gpt-4o-mini), falls back to Ollama (tinyllama)
- **Use Cases**: Complex fraud analysis, regulatory compliance, investment due diligence
- **Output**: Detailed multi-perspective analysis with action items

📖 **For complete details, see [Agentic Workflows Guide](docs/AGENTIC_WORKFLOWS.md) and [Tools Framework Guide](docs/TOOLS_FRAMEWORK.md)**

### Complete Processing Timeline
- **Email arrival to full enrichment** (Automatic): ~3-5 seconds
- **Agentic workflow analysis** (On-demand): ~30-60 seconds
- **No user intervention required**: Everything automatic except agentic workflows
- **Continuous operation**: 24/7 background processing for ML/LLM/RAG
- **Scalability**: Processes 10+ emails/second (automatic pipeline)

---

## Quick Start

### Prerequisites

- Docker and Docker Compose plugin
- At least 4GB available RAM
- Ports available: 80, 443, 1025, 5432, 8000, 8002, 8004, 8006, 8025, 8100, 9000, 11434

### Installation

**1. Clone the repository:**
```bash
git clone <repository-url>
cd ai-cup-2025
```

**2. (Optional) Configure OpenAI for Agentic Teams:**
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
# If not configured, automatically falls back to Ollama
```

**3. Start all services:**
```bash
./start.sh
```

The startup script will:
- Check Docker availability
- Detect and handle port conflicts
- Build images if needed
- Start all services in correct order
- Display access URLs when ready

### First Use

**The system processes emails automatically - just wait a few minutes after startup!**

1. **Access Dashboard**: http://localhost
   - View real-time statistics as emails are processed
   - Charts update automatically via SSE
   - Monitor phishing detection rate and confidence scores

2. **Automatic Processing** (No action required):
   - Email Seeder loads 10,000 emails into MailPit on startup
   - Email Fetcher automatically imports all emails to database
   - Email Analyzer processes ALL emails in background:
     - ✓ ML phishing detection (ensemble model)
     - ✓ LLM summary generation
     - ✓ Badge categorization (8 types)
     - ✓ Call-to-action extraction
     - ✓ Quick reply drafts
     - ✓ RAG enrichment (wiki + employee data)
   - **Processing time**: 3-5 seconds per email, runs continuously

3. **View Daily Inbox Digest**: http://localhost/daily-inbox-digest
   - **RISK** - Phishing emails detected (highlighted)
   - **MEETING** - Calendar and scheduling emails
   - **FOLLOW_UP** - Emails with action items
   - **VIP** - Important contacts
   - **EXTERNAL** - Outside communications
   - **AUTOMATED** - System notifications
   - **NEWSLETTER** - Marketing emails
   - **FINANCE** - Transaction notifications
   - Click any badge to filter and view emails in that category
   - View aggregated call-to-action items

4. **Explore Email Details**: Click any email to see:
   - AI-generated summary (2-3 sentences)
   - Phishing detection results with confidence score
   - Call-to-action list (extracted action items)
   - Quick reply drafts (3 tones: Formal, Friendly, Brief)
   - RAG enrichment (wiki context + employee info)
   - Risk indicators and workflow analysis

5. **Try Agentic Teams** (Optional - Manual Trigger):
   - Assign complex emails to specialized teams
   - Watch real-time multi-agent debate (3 rounds)
   - Get comprehensive analysis with action items
   - Requires OpenAI API key (or uses Ollama fallback)

### Access URLs

**Frontend Application (Angular 18):**
- **Dashboard**: http://localhost or https://localhost
- **Mailbox**: http://localhost/mailbox
- **Daily Inbox Digest**: http://localhost/daily-inbox-digest
- **Agentic Teams**: http://localhost/agentic-teams

**Backend Services:**
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health
- **MailPit Web UI**: http://localhost:8025
- **OtterWiki (RAG)**: http://localhost:9000
- **Employee Directory API**: http://localhost:8100/docs
- **Naive Bayes Model**: http://localhost:8002
- **Random Forest Model**: http://localhost:8004
- **Fine-tuned LLM Model**: http://localhost:8006
- **Ollama Service**: http://localhost:11434

## Configuration

### Environment Variables

**LLM Configuration** (Optional - Automatic Fallback):
```bash
OPENAI_API_KEY=your_key_here        # Optional, falls back to Ollama
OPENAI_MODEL=gpt-4o-mini            # Default model for agentic teams
```

**Investment Team Tools** (Optional):
```bash
SERPER_API_KEY=your_key_here        # Google search via Serper API (fallback: mock)
SEC_API_KEY=your_key_here           # SEC EDGAR filings (fallback: free EDGAR)
BROWSERLESS_API_KEY=your_key_here   # JS-enabled web scraping (fallback: HTTP)
```

**Fraud Detection Tools** (Optional):
```bash
IPGEOLOCATION_API_KEY=your_key_here # IP threat intelligence (fallback: mock)
ABSTRACTAPI_EMAIL_KEY=your_key_here # Email validation (fallback: format check)
```

**MCP Configuration** (Optional):
```bash
MCP_ENABLED=false                   # Redundant fallback layer
MCP_FRAUD_CASES_PATH=/app/fraud_cases # Fraud case files
```

**Service Configuration** (Auto-configured in Docker):
```bash
DATABASE_URL=postgresql://mailbox_user:mailbox_pass@postgres:5432/mailbox_db
OLLAMA_HOST=ollama                  # Ollama service hostname
OLLAMA_PORT=11434                   # Ollama port
OLLAMA_MODEL=phi3                   # Local LLM model for email processing
MAILPIT_HOST=mailpit                # MailPit hostname
MAILPIT_PORT=8025                   # MailPit API port
MAILPIT_SMTP_PORT=1025              # MailPit SMTP port
DISABLE_WIKI_ENRICHMENT=false       # Enable/disable RAG enrichment
```

### LLM Mode Selection

The system automatically selects the appropriate LLM:

**Scenario 1: No OpenAI Key**
- Agentic Teams: Ollama (tinyllama:latest)
- Email Processing: Ollama (phi3)
- Status: Fully offline operation

**Scenario 2: Valid OpenAI Key**
- Agentic Teams: OpenAI (gpt-4o-mini)
- Email Processing: Ollama (phi3)
- Status: Hybrid operation

**Scenario 3: OpenAI Fails**
- Automatic fallback to Ollama
- Status: Resilient operation

## Management Scripts

### start.sh - Start Application
```bash
./start.sh
```
Intelligent startup with port conflict detection and status display.

### stop.sh - Stop Application
```bash
./stop.sh              # Stop containers
./stop.sh -v           # Stop and remove volumes (deletes data)
./stop.sh -i           # Stop and remove images (requires rebuild)
./stop.sh -a           # Stop and remove everything
```

### logs.sh - View Logs
```bash
./logs.sh              # Show all logs
./logs.sh -f           # Follow all logs (live)
./logs.sh backend      # Show backend logs only
./logs.sh -f backend   # Follow backend logs
```

Available services: `postgres`, `backend`, `frontend-angular`, `mailpit`, `ollama`, `otterwiki`, `otterwiki-init`, `email-seeder`, `email-fetcher`, `email-analyzer`, `model-naive-bayes`, `model-random-forest`, `model-fine-tuned-llm`, `employee-directory`

## Dataset

**Phishing and Legitimate Emails Dataset** by Kuladeep19
- **Source**: https://www.kaggle.com/datasets/kuladeep19/phishing-and-legitimate-emails-dataset
- **Size**: 10,000 emails with ground truth labels
- **Distribution**: 60% phishing (6,000), 40% legitimate (4,000)
- **Categories**:
  - Credential harvesting
  - Authority scams
  - Romance/dating scams
  - Financial fraud
  - Legitimate emails

**Email Seeder:**
- Automatically loads dataset on startup
- Sends all emails to MailPit SMTP server
- Preserves labels in custom headers (X-Email-Label, X-Phishing-Type)
- Generates contextual sender addresses

## Project Structure

```
ai-cup-2025/
├── docker-compose.yml           # Service orchestration (14 services)
├── .env.example                 # Environment template
├── start.sh                     # Intelligent startup script (4KB)
├── stop.sh                      # Shutdown script with cleanup (4KB)
├── logs.sh                      # Log viewer (2.6KB)
├── clean-start.sh              # Fresh installation (9.3KB)
├── README.md                    # This file (comprehensive documentation)
├── frontend-angular/            # Angular 18 web application
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── angular.json
│   ├── package.json
│   └── src/
│       └── app/
│           ├── core/           # Services, guards, models
│           │   ├── models/
│           │   │   ├── email.model.ts
│           │   │   ├── statistics.model.ts
│           │   │   └── workflow.model.ts
│           │   └── services/
│           │       ├── api.service.ts
│           │       ├── email.service.ts
│           │       ├── page-state.service.ts
│           │       └── sse.service.ts
│           ├── features/       # Feature modules
│           │   ├── dashboard/
│           │   ├── mailbox/
│           │   ├── daily-inbox-digest/
│           │   └── agentic-teams/
│           ├── shared/         # Shared components
│           │   └── components/
│           │       ├── navbar/
│           │       └── sidebar/
│           └── store/          # NgRx state management
│               ├── emails/
│               ├── statistics/
│               └── workflows/
├── backend/                     # FastAPI application (141 Python files)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                 # API endpoints 
│   ├── models.py               # Database ORM models
│   ├── database.py             # DB connection
│   ├── workflows.py            # ML workflows
│   ├── agentic_teams.py        # Multi-agent orchestration
│   ├── fraud_workflow.py       # Fraud detection workflow 
│   ├── compliance_workflow.py  # Compliance workflow 
│   ├── investment_workflow.py  # Investment research 
│   ├── wiki_enrichment.py      # RAG system
│   ├── llm_service.py          # LLM integration
│   ├── mailpit_client.py       # MailPit integration
│   ├── events.py               # SSE event broadcaster
│   ├── tools/                  # Pluggable tools framework (13 tools)
│   │   ├── base_tool.py
│   │   ├── tool_registry.py
│   │   ├── investigation_tools.py
│   │   ├── sanctions_tools.py
│   │   ├── aml_tools.py
│   │   ├── policy_compliance_tools.py
│   │   ├── regulatory_tools.py
│   │   ├── risk_tools.py
│   │   ├── transaction_tools.py
│   │   ├── sec_tools.py
│   │   ├── entity_resolver.py
│   │   ├── search_tools.py
│   │   ├── browser_tools.py
│   │   ├── calculator_tools.py
│   │   └── serper_search_plugin.py
│   ├── test_fallback.py        # Fallback testing
│   ├── test_compliance.py      # Compliance testing
│   └── db/
│       └── changelog/          # Liquibase migrations (9 changesets)
├── scripts/                     # Automation scripts
│   ├── check-ports.sh
│   ├── deploy.sh
│   ├── ollama-entrypoint.sh
│   ├── otterwiki-auto-init.sh
│   ├── wiki-init.sh
│   ├── generate_wiki_pages.py (40KB)
│   └── [12+ test scripts]
├── regression-tests/            # Playwright UI tests
│   ├── dashboard.spec.js
│   └── package.json
├── email-seeder/               # Dataset loader
│   ├── Dockerfile
│   ├── seed_emails.py
│   └── phishing_emails.csv (10,000 emails)
├── email-fetcher/              # Background fetcher
│   ├── Dockerfile
│   └── fetch_emails.py
├── email-analyzer/             # Background analyzer
│   ├── Dockerfile
│   └── analyze_emails.py
├── model-naive-bayes/          # Naive Bayes service (76.50%)
│   ├── Dockerfile
│   ├── train.py
│   └── serve.py
├── model-random-forest/        # Random Forest service (71.70%)
│   ├── Dockerfile
│   ├── train.py
│   └── serve.py
├── model-fine-tuned-llm/       # Fine-tuned LLM service (59%)
│   ├── Dockerfile
│   └── serve.py
├── employee-directory/         # Employee API
│   ├── Dockerfile
│   └── main.py
└── postgres/
    └── init.sql                # DB initialization
```

## Performance Metrics

### ML Model Performance
- **Ensemble Accuracy**: 82.10%
- **Ensemble F1-Score**: 81.90%
- **Ensemble Precision**: 81.90%
- **Ensemble Recall**: 82.00%
- **Training Time**: ~5-10 minutes (all models)
- **Inference Time**: <100ms per email

### System Performance

**Email Processing Pipeline:**
- **Email Fetching**: Continuous, real-time import from MailPit
- **ML Phishing Detection**: <100ms per email (ensemble of 3 models)
- **LLM Summarization**: 2-3 seconds per email (Ollama phi3)
- **Badge Categorization**: Instant (part of LLM processing)
- **CTA Extraction**: Automatic during LLM processing
- **Quick Reply Generation**: Automatic (3 tones per email)
- **RAG Enrichment**: <500ms per email (wiki + employee lookup)
- **End-to-End**: 3-5 seconds from email arrival to fully enriched
- **Throughput**: 10+ emails/second in batch processing
- **Database**: 10,000+ emails indexed with full-text search
- **Real-time Updates**: <100ms SSE latency

**Automatic Features:**
- **Badge Assignment Rate**: 100% of emails auto-categorized
- **CTA Extraction Rate**: Automatic for all emails with action items
- **Summary Generation**: 100% of emails (2-3 sentences each)
- **Quick Reply Coverage**: 100% (3 tones: Formal, Friendly, Brief)
- **RAG Enrichment**: Automatic for all processable emails
- **Phishing Detection**: 100% of emails analyzed by ensemble model

**Agentic Teams (On-Demand):**
- **Multi-Agent Discussion**: 30-60 seconds (3 rounds, 4 agents per team)
- **Team Detection**: Automatic LLM-based suggestion
- **Real-time Streaming**: Round-by-round updates via SSE

### Scalability
- **Tested Load**: 10,000 emails
- **Concurrent Users**: 10+ simultaneous connections
- **Background Processing**: Automatic queue management
- **Memory Usage**: ~3GB total (all services)
- **Disk Usage**: ~5GB (including models and database)

## Technology Stack

### Backend
| Component | Technology | Version |
|-----------|-----------|---------|
| Framework | FastAPI | 0.110+ |
| Language | Python | 3.11 |
| Web Server | Uvicorn | 0.27+ |
| ORM | SQLAlchemy | 2.0.23 |
| Database Migrations | Liquibase | XML-based |
| Async/HTTP | asyncio, httpx | Latest |
| LLM - Offline | Ollama | phi3, tinyllama |
| LLM - Cloud | OpenAI SDK | 1.50+ |
| RAG/Vector Store | ChromaDB | 0.4.24 |
| Embeddings | sentence-transformers | 2.5.1 (all-MiniLM-L6-v2) |
| Deep Learning | PyTorch, Transformers | 2.2.2, 4.38.2 |
| ML Models | scikit-learn | pickle-based |
| Data Processing | NumPy, Pandas | 1.26.4 |
| Web Scraping | BeautifulSoup4 | 4.12.3 |
| Agent Framework | LangGraph | For agentic workflows |
| MCP Protocol | mcp | 1.0+ |

### Frontend
| Component | Technology | Version |
|-----------|-----------|---------|
| Framework | Angular | 18.2.0 |
| State Management | NgRx | 18.1.1 |
| Styling | Bootstrap | 5.3.8 |
| CSS Framework | Material Design | via Bootstrap |
| Charts | Chart.js | 4.5.1 |
| Icons | Material Icons, Font Awesome | Latest |
| HTTP Client | Angular HttpClient | 18.2.0 |
| Markdown Parser | marked | 17.0.0 |

### Infrastructure
| Service | Technology | Purpose |
|---------|-----------|---------|
| Containerization | Docker Compose | Multi-service orchestration |
| Database | PostgreSQL 15 | Main data store |
| Web Server | Nginx | Frontend serving + HTTPS |
| Email Server | MailPit | SMTP + POP3 + Web UI |
| Wiki | OtterWiki | Knowledge base (RAG) |
| LLM Inference | Ollama | Local model serving |
| Network | Bridge network | Service-to-service communication |

## Development

### Testing
```bash
# Run fallback mechanism tests
docker compose exec backend python3 /app/test_fallback.py

# Check service health
curl http://localhost:8000/health

# Test ML endpoint
curl -X POST http://localhost:8000/api/emails/1/process

# Test agentic teams
curl -X POST http://localhost:8000/api/agentic/emails/1/process?team=fraud
```

### Debugging
```bash
# View backend logs
./logs.sh -f backend

# Check database
docker compose exec postgres psql -U mailbox_user -d mailbox -c "SELECT COUNT(*) FROM emails;"

# Restart single service
docker compose restart backend

# Access Ollama directly
docker compose exec ollama ollama list
```

### Rebuilding
```bash
# Rebuild specific service
docker compose build backend

# Rebuild all services
docker compose build

# Force rebuild
docker compose build --no-cache
```

## Troubleshooting

### Common Issues

**1. Ports already in use:**
```bash
# The start.sh script automatically detects conflicts
# Or manually check:
./stop.sh
./start.sh
```

**2. Ollama model not downloaded:**
```bash
# Check Ollama status
docker compose exec ollama ollama list

# Manually pull model
docker compose exec ollama ollama pull phi3
```

**3. Database migration fails:**
```bash
# Check migration logs
./logs.sh backend | grep liquibase

# Reset database (WARNING: deletes data)
./stop.sh -v
./start.sh
```

**4. OpenAI fallback not working:**
```bash
# Verify configuration
docker compose exec backend env | grep OPENAI

# Check backend logs for fallback messages
./logs.sh backend | grep "Orchestrator"
```

## Contributing

### Development Workflow
1. Make changes to code
2. Test locally with Docker Compose
3. Update documentation if needed
4. Commit changes with descriptive messages

### Code Style
- Backend: PEP 8 (Python)
- Frontend: Standard JavaScript
- Comments: Clear, concise explanations
- Documentation: Markdown with examples

## License

MIT License - See LICENSE file for details

## Detailed Documentation

For in-depth technical documentation, see the specialized guides in the `/docs` directory:

### Core Documentation

**[Agentic Workflows Guide](docs/AGENTIC_WORKFLOWS.md)** - Complete guide to multi-agent collaboration
- Architecture overview and orchestration
- Collaboration patterns (Doer-Checker, Multi-Round Debate, Research Synthesis)
- Fraud Investigation Workflow
- Compliance & Regulatory Workflow
- Investment Research Workflow
- API usage and real-time streaming (SSE)
- Configuration and best practices
- Performance metrics and optimization

**[Tools Framework Guide](docs/TOOLS_FRAMEWORK.md)** - Complete guide to pluggable tools system
- Architecture and design principles
- All 13 specialized tools documented:
  - Investigation Tools (OFAC screening, fraud detection)
  - Sanctions Tools (Multi-database screening)
  - AML Tools (KYC risk assessment)
  - Policy Compliance Tools
  - Regulatory Tools
  - Risk Tools (IP geolocation, threat intel)
  - Transaction Tools
  - SEC Tools (EDGAR filings)
  - Entity Resolver
  - Search Tools (Serper API)
  - Browser Tools (Browserless)
  - Calculator Tools
  - Serper Search Plugin
- API integration and tool registry
- Usage examples and development guide
- Environment variables and fallback strategies
- Performance benchmarks

**[Database Schema](docs/database-schema.md)** - Database design and migrations
- Complete schema documentation
- Entity relationships
- Liquibase migration history
- Table descriptions and indexes

### Additional Resources

**[ML Model Accuracy Results](docs/ML_MODEL_ACCURACY_RESULTS.txt)** - Ensemble model performance metrics
- Individual model accuracy scores
- Ensemble voting results (82.10% accuracy)
- Test dataset statistics

## Support

For issues, questions, or contributions:
- **API Documentation**: http://localhost:8000/docs (when running)
- **Agentic Workflows**: See [docs/AGENTIC_WORKFLOWS.md](docs/AGENTIC_WORKFLOWS.md)
- **Tools Framework**: See [docs/TOOLS_FRAMEWORK.md](docs/TOOLS_FRAMEWORK.md)
- **Database Schema**: See [docs/database-schema.md](docs/database-schema.md)
- **Logs**: Use `./logs.sh` for debugging

## Acknowledgments

- **Dataset**: Kuladeep19 - Phishing and Legitimate Emails Dataset (Kaggle)
- **UI Template**: Material Dashboard by Creative Tim
- **LLM**: Ollama project, OpenAI
- **Framework**: FastAPI, LangGraph

---

## Project Metrics & Statistics

### Codebase
- **Total Python Files**: 141 files
- **Backend Main Module**: (`main.py`)
- **Agentic Teams Module**: (`agentic_teams.py`)
- **Fraud Workflow**: (`fraud_workflow.py`)
- **Compliance Workflow**: (`compliance_workflow.py`)
- **Investment Workflow**: (`investment_workflow.py`)
- **Tools Framework**: 13 specialized tools with extensible architecture

### API & Features
- **API Endpoints**: 46 REST endpoints across 8 categories
- **Specialized Workflows**: 3 fully implemented (Fraud, Compliance, Investment)
- **Individual Agents**: 12 specialized agents (4 per workflow)
- **Additional Team Definitions**: 4 basic teams in agentic_teams.py
- **Pluggable Tools**: 13 specialized tools with automatic fallbacks
- **Database Tables**: 5 main tables
- **Liquibase Migrations**: 9 changesets
- **Docker Services**: 14 containers

### Recent Development Milestones

**Angular Migration (Latest)**:
- Migrated frontend from vanilla JavaScript to Angular 18
- Implemented NgRx state management for centralized state
- Created modular architecture with feature modules
- Enhanced real-time updates with improved SSE integration

**Tools Framework Implementation**:
- Built extensible pluggable tools architecture
- Implemented 13 specialized tools with automatic fallbacks
- Added graceful degradation for all external APIs
- Created tool registry for dynamic discovery
- Zero hard dependencies - all APIs optional

**Compliance Workflow**:
- Implemented OFAC sanctions screening (free, no API key)
- Added AML/KYC compliance checks
- Created regulatory assessment workflow
- Integrated 4 specialized compliance agents

**Fraud Detection Enhancement**:
- Built multi-pattern fraud workflow
- Implemented doer-checker validation pattern
- Added iterative deepening (up to 3 iterations)
- Created phishing, BEC, and account compromise detection

**Investment Research Team**:
- Integrated SEC EDGAR API for 10-K/10-Q filings
- Added Serper API for Google search capability
- Implemented Browserless for JS-enabled web scraping
- Created 4 specialized investment research agents

**Production Readiness**:
- Comprehensive test suite (Playwright + Python tests)
- 82.10% ML ensemble accuracy achieved
- Real-time SSE for all agentic workflows
- Automatic service health checks
- Complete feature documentation in README

---

