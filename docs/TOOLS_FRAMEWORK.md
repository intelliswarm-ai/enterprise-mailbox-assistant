# Tools Framework Documentation

## Overview

The emterprise mailbox platform features an extensible pluggable tools framework with **13 specialized tools** that enable agentic workflows to interact with external systems, APIs, and data sources. All tools are designed with **automatic fallback mechanisms** and **graceful degradation** to ensure zero hard dependencies.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Tool Categories](#tool-categories)
3. [Individual Tools](#individual-tools)
4. [API Integration](#api-integration)
5. [Tool Registry](#tool-registry)
6. [Usage Examples](#usage-examples)
7. [Development Guide](#development-guide)

---

## Architecture

### Core Components

**Base Tool Class**: `backend/tools/base_tool.py`
```python
class BaseTool:
    name: str                    # Tool identifier
    description: str             # Tool purpose
    required_args: List[str]     # Required parameters

    async def execute(self, **kwargs) -> Dict
        # Tool execution logic
        # Returns: {"success": bool, "data": Any, "error": str}
```

**Tool Registry**: `backend/tools/tool_registry.py`
- Dynamic tool discovery and registration
- Capability-based tool search
- Category-based organization
- Readiness checking
- Team-tool assignment

**Directory Structure**:
```
backend/tools/
├── base_tool.py              # Base class
├── tool_registry.py          # Registry system
├── investigation_tools.py    # OFAC, fraud detection
├── sanctions_tools.py        # Multi-database screening
├── aml_tools.py             # AML/KYC compliance
├── policy_compliance_tools.py # Policy enforcement
├── regulatory_tools.py       # Regulatory assessment
├── risk_tools.py            # IP geolocation, threat intel
├── transaction_tools.py      # Transaction analysis
├── sec_tools.py             # SEC EDGAR filings
├── entity_resolver.py       # Name standardization
├── search_tools.py          # Internet search
├── browser_tools.py         # Web scraping
├── calculator_tools.py      # Financial calculations
└── serper_search_plugin.py  # Advanced search plugin
```

### Design Principles

1. **Zero Hard Dependencies**: All external APIs have fallback mechanisms
2. **Graceful Degradation**: System continues working when APIs unavailable
3. **Automatic Fallbacks**: Built-in mock responses for all tools
4. **Extensible Architecture**: Easy to add new tools
5. **Type Safety**: Clear input/output contracts
6. **Async Support**: Non-blocking execution

---

## Tool Categories

### 1. Investigation & Screening
- **Investigation Tools**: OFAC sanctions, business verification, fraud detection
- **Sanctions Tools**: Multi-database sanctions screening
- **AML Tools**: AML/KYC compliance checks

### 2. Compliance & Policy
- **Policy Compliance Tools**: Internal policy enforcement
- **Regulatory Tools**: Regulatory requirement assessment

### 3. Risk & Security
- **Risk Tools**: IP geolocation, VPN/Proxy/TOR detection
- **Transaction Tools**: Transaction pattern analysis

### 4. Research & Data
- **SEC Tools**: SEC EDGAR filing retrieval
- **Search Tools**: Internet search integration
- **Browser Tools**: Website scraping

### 5. Analytics & Processing
- **Entity Resolver**: Entity name standardization
- **Calculator Tools**: Financial metric calculations
- **Serper Search Plugin**: Advanced search capabilities

---

## Individual Tools

### 1. Investigation Tools

**File**: `backend/tools/investigation_tools.py`

**Purpose**: OFAC sanctions screening, business verification, fraud detection

**Capabilities**:
- OFAC sanctions screening (FREE, no API key)
- Business registration verification
- Email fraud pattern detection
- Merchant reputation checking

**Key Functions**:

#### OFAC Sanctions Screening
```python
{
  "tool": "ofac_screening",
  "args": {
    "entity_name": "John Doe",
    "entity_type": "individual"  # or "company"
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "match_found": false,
    "confidence": 0.0,
    "lists_checked": ["OFAC SDN", "UN", "EU", "UK"],
    "source": "OpenSanctions API"
  }
}
```

**Fallback**: Always uses free OpenSanctions API (no fallback needed)

#### Business Verification
```python
{
  "tool": "verify_business",
  "args": {
    "business_name": "ACME Corp",
    "jurisdiction": "US"
  }
}
```

**Fallback**: Mock response with basic validation

---

### 2. Sanctions Tools

**File**: `backend/tools/sanctions_tools.py`

**Purpose**: Multi-database sanctions and PEP screening

**Databases Checked**:
- OFAC SDN (Specially Designated Nationals)
- UN Security Council Consolidated List
- EU Consolidated Financial Sanctions List
- UK HM Treasury Sanctions List
- PEP (Politically Exposed Persons) Database

**Key Functions**:

#### Multi-Database Screening
```python
{
  "tool": "sanctions_screening",
  "args": {
    "name": "Entity Name",
    "type": "individual",  # or "company"
    "databases": ["OFAC", "UN", "EU", "UK", "PEP"]
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "matches": [
      {
        "database": "OFAC",
        "name": "Similar Name",
        "confidence": 0.85,
        "list": "SDN",
        "program": "IRAN"
      }
    ],
    "pep_match": false,
    "risk_level": "high"
  }
}
```

**API**: OpenSanctions (free, no key required)

**Fallback**: N/A (always free)

---

### 3. AML Tools

**File**: `backend/tools/aml_tools.py`

**Purpose**: AML/KYC compliance checks and risk classification

**Capabilities**:
- Customer due diligence (CDD)
- Enhanced due diligence (EDD) triggers
- Risk classification (Low/Medium/High)
- KYC verification
- Ongoing monitoring recommendations

**Key Functions**:

#### KYC Risk Assessment
```python
{
  "tool": "kyc_risk_assessment",
  "args": {
    "customer_name": "John Doe",
    "country": "US",
    "business_type": "individual",
    "transaction_volume": 50000
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "risk_level": "medium",
    "risk_factors": [
      "High transaction volume for individual customer",
      "First-time customer"
    ],
    "edd_required": false,
    "recommendations": [
      "Obtain proof of income",
      "Monitor transactions for 90 days"
    ]
  }
}
```

**Fallback**: Rule-based risk classification

---

### 4. Policy Compliance Tools

**File**: `backend/tools/policy_compliance_tools.py`

**Purpose**: Internal policy enforcement and compliance rule checking

**Capabilities**:
- Transaction limit verification
- Approval workflow validation
- Policy rule checking
- Compliance gap identification

**Key Functions**:

#### Policy Verification
```python
{
  "tool": "check_policy_compliance",
  "args": {
    "policy_type": "transaction_limits",
    "transaction_amount": 100000,
    "user_role": "manager"
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "compliant": false,
    "violations": [
      "Transaction exceeds manager approval limit ($50,000)"
    ],
    "required_approvals": ["Senior Manager", "Director"],
    "policy_reference": "POL-FIN-001"
  }
}
```

**Fallback**: Built-in policy rules

---

### 5. Regulatory Tools

**File**: `backend/tools/regulatory_tools.py`

**Purpose**: Regulatory requirement assessment and compliance validation

**Capabilities**:
- Regulatory framework mapping
- Requirement assessment
- Compliance gap analysis
- Regulatory reporting validation

**Key Functions**:

#### Regulatory Assessment
```python
{
  "tool": "assess_regulatory_requirements",
  "args": {
    "jurisdiction": "US",
    "business_type": "financial_services",
    "activity": "wire_transfer"
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "applicable_regulations": [
      "Bank Secrecy Act (BSA)",
      "USA PATRIOT Act",
      "FinCEN Regulations"
    ],
    "requirements": [
      "Customer identification program (CIP)",
      "Suspicious activity reporting (SAR)",
      "Currency transaction reporting (CTR)"
    ],
    "compliance_level": "partial"
  }
}
```

**Fallback**: Built-in regulatory database

---

### 6. Risk Tools

**File**: `backend/tools/risk_tools.py`

**Purpose**: IP geolocation, VPN/Proxy/TOR detection, threat intelligence

**Capabilities**:
- IP geolocation
- VPN/Proxy detection
- TOR exit node detection
- Threat intelligence lookup
- Risk scoring

**Key Functions**:

#### IP Risk Analysis
```python
{
  "tool": "analyze_ip_risk",
  "args": {
    "ip_address": "192.168.1.1"
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "country": "US",
    "city": "New York",
    "is_vpn": false,
    "is_proxy": false,
    "is_tor": false,
    "threat_level": "low",
    "reputation": "clean",
    "asn": "AS15169 Google LLC"
  }
}
```

**API**: IPGeolocation API

**Fallback**: Mock geolocation data with basic validation

---

### 7. Transaction Tools

**File**: `backend/tools/transaction_tools.py`

**Purpose**: Transaction pattern analysis and anomaly detection

**Capabilities**:
- Pattern analysis
- Anomaly detection
- Velocity checking
- Behavioral analysis

**Key Functions**:

#### Transaction Pattern Analysis
```python
{
  "tool": "analyze_transaction_pattern",
  "args": {
    "amount": 10000,
    "currency": "USD",
    "sender": "John Doe",
    "recipient": "Jane Smith",
    "historical_data": [...]
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "anomaly_detected": true,
    "anomaly_type": "amount_deviation",
    "risk_score": 75,
    "explanation": "Transaction amount 300% higher than historical average",
    "recommendations": [
      "Request additional verification",
      "Contact customer for confirmation"
    ]
  }
}
```

**Fallback**: Statistical analysis using historical patterns

---

### 8. SEC Tools

**File**: `backend/tools/sec_tools.py`

**Purpose**: SEC EDGAR filing retrieval and analysis

**Capabilities**:
- Retrieve 10-K (annual reports)
- Retrieve 10-Q (quarterly reports)
- Retrieve 8-K (current reports)
- Extract key sections (MD&A, Risk Factors)
- Search by company name or CIK

**Key Functions**:

#### Retrieve SEC Filing
```python
{
  "tool": "get_sec_filing",
  "args": {
    "company": "APPLE INC",
    "filing_type": "10-K",
    "year": 2024
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "filing_url": "https://www.sec.gov/...",
    "filing_date": "2024-10-31",
    "company_name": "APPLE INC",
    "cik": "0000320193",
    "excerpts": {
      "risk_factors": "...",
      "md_and_a": "..."
    }
  }
}
```

**API**: SEC EDGAR API (optional key), free EDGAR access

**Fallback**: Direct EDGAR HTML parsing

---

### 9. Entity Resolver

**File**: `backend/tools/entity_resolver.py`

**Purpose**: Entity name standardization, fuzzy matching, duplicate detection

**Capabilities**:
- Name standardization
- Fuzzy matching (Levenshtein distance)
- Duplicate detection
- Entity disambiguation

**Key Functions**:

#### Resolve Entity Name
```python
{
  "tool": "resolve_entity",
  "args": {
    "name": "Apple Inc.",
    "candidates": ["APPLE INC", "Apple Computer", "Apple Incorporated"]
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "standardized_name": "APPLE INC",
    "match_confidence": 0.95,
    "matched_candidate": "APPLE INC",
    "alternatives": [
      {"name": "Apple Incorporated", "confidence": 0.85}
    ]
  }
}
```

**Fallback**: Built-in fuzzy matching algorithm

---

### 10. Search Tools

**File**: `backend/tools/search_tools.py`

**Purpose**: Internet search integration via Serper API (Google)

**Capabilities**:
- Google search results
- News search
- Image search
- Snippet extraction

**Key Functions**:

#### Internet Search
```python
{
  "tool": "search_internet",
  "args": {
    "query": "ACME Corp annual revenue 2024",
    "num_results": 10
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "title": "ACME Corp Reports Record Revenue",
        "url": "https://example.com/...",
        "snippet": "ACME Corp announced annual revenue of $5.2B..."
      }
    ],
    "total_results": 1500000
  }
}
```

**API**: Serper API (Google Search)

**Fallback**: Mock search results with generic responses

---

### 11. Browser Tools

**File**: `backend/tools/browser_tools.py`

**Purpose**: Website scraping and content extraction

**Capabilities**:
- JavaScript-enabled scraping (Browserless)
- HTML content extraction
- Screenshot capture
- PDF rendering

**Key Functions**:

#### Scrape Website
```python
{
  "tool": "scrape_website",
  "args": {
    "url": "https://www.acmecorp.com/investors",
    "wait_for": "#earnings-table"
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "content": "<html>...</html>",
    "text": "Extracted text content...",
    "title": "ACME Corp - Investor Relations",
    "scraped_at": "2025-12-10T10:30:00Z"
  }
}
```

**API**: Browserless (Chrome automation)

**Fallback**: HTTP GET with BeautifulSoup parsing

---

### 12. Calculator Tools

**File**: `backend/tools/calculator_tools.py`

**Purpose**: Financial metric calculations and ratio analysis

**Capabilities**:
- Financial ratio calculation (P/E, D/E, Current Ratio)
- Growth rate calculations
- Return calculations (ROI, ROE, ROA)
- Valuation metrics

**Key Functions**:

#### Calculate Financial Ratios
```python
{
  "tool": "calculate_financial_ratios",
  "args": {
    "revenue": 5200000000,
    "net_income": 780000000,
    "total_assets": 15000000000,
    "total_debt": 4500000000,
    "equity": 10500000000,
    "current_assets": 6000000000,
    "current_liabilities": 3000000000
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "profit_margin": 0.15,
    "roa": 0.052,
    "roe": 0.074,
    "debt_to_equity": 0.43,
    "current_ratio": 2.0
  }
}
```

**Fallback**: N/A (pure calculation, always available)

---

### 13. Serper Search Plugin

**File**: `backend/tools/serper_search_plugin.py`

**Purpose**: Advanced search plugin for MCP integration

**Capabilities**:
- Advanced Google search
- News search with date filters
- Academic search
- Patent search

**Key Functions**:

#### Advanced Search
```python
{
  "tool": "serper_advanced_search",
  "args": {
    "query": "artificial intelligence healthcare applications",
    "search_type": "news",
    "date_range": "past_month"
  }
}
```

**API**: Serper API

**Fallback**: Basic search with mock results

---

## API Integration

### Tool Execution Endpoint

**Endpoint**: `POST /api/tools/test`

**Request**:
```json
{
  "tool_name": "ofac_screening",
  "args": {
    "entity_name": "John Doe",
    "entity_type": "individual"
  }
}
```

**Response**:
```json
{
  "success": true,
  "tool": "ofac_screening",
  "execution_time": 0.245,
  "result": {
    "match_found": false,
    "confidence": 0.0
  }
}
```

### Tool Registry Endpoints

**Get All Tools**:
```bash
GET /api/tools/registry
```

**Get Tool Details**:
```bash
GET /api/tools/{tool_name}
```

**Search by Capability**:
```bash
GET /api/tools/search/capability/sanctions_screening
```

**Search by Category**:
```bash
GET /api/tools/search/category/compliance
```

**Check Tool Readiness**:
```bash
GET /api/tools/{tool_name}/readiness
```

**Get Tools for Team**:
```bash
GET /api/teams/{team_key}/tools
```

---

## Tool Registry

### Dynamic Tool Discovery

The tool registry automatically discovers and registers all tools in the `tools/` directory:

```python
from tools.tool_registry import ToolRegistry

registry = ToolRegistry()
registry.register_all_tools()

# Get tool by name
tool = registry.get_tool("ofac_screening")

# Search by capability
tools = registry.search_by_capability("sanctions")

# Search by category
tools = registry.search_by_category("compliance")
```

### Tool Metadata

Each registered tool includes:
- **Name**: Unique identifier
- **Description**: Tool purpose
- **Category**: Tool category
- **Capabilities**: List of capabilities
- **Required Args**: Parameter specifications
- **API Dependencies**: External API requirements
- **Fallback Strategy**: Fallback behavior
- **Readiness Status**: Current availability

### Tool Assignment to Teams

Tools can be assigned to specific teams:

```python
# Fraud team tools
fraud_tools = [
    "investigation_tools",
    "risk_tools",
    "transaction_tools",
    "entity_resolver"
]

# Compliance team tools
compliance_tools = [
    "sanctions_tools",
    "aml_tools",
    "policy_compliance_tools",
    "regulatory_tools"
]

# Investment team tools
investment_tools = [
    "sec_tools",
    "search_tools",
    "browser_tools",
    "calculator_tools"
]
```

---

## Usage Examples

### Example 1: OFAC Screening in Compliance Workflow

```python
from tools.investigation_tools import InvestigationTools

tool = InvestigationTools()
result = await tool.execute(
    action="ofac_screening",
    entity_name="Example Company LLC",
    entity_type="company"
)

if result["success"]:
    if result["data"]["match_found"]:
        print(f"SANCTIONS MATCH FOUND!")
        print(f"Confidence: {result['data']['confidence']}")
        print(f"Lists: {result['data']['lists_checked']}")
    else:
        print("No sanctions match - entity cleared")
```

### Example 2: SEC Filing Retrieval in Investment Workflow

```python
from tools.sec_tools import SECTools

tool = SECTools()
result = await tool.execute(
    action="get_filing",
    company="TESLA INC",
    filing_type="10-K",
    year=2024
)

if result["success"]:
    filing = result["data"]
    print(f"Filing URL: {filing['filing_url']}")
    print(f"Risk Factors: {filing['excerpts']['risk_factors'][:500]}...")
```

### Example 3: Transaction Pattern Analysis in Fraud Workflow

```python
from tools.transaction_tools import TransactionTools

tool = TransactionTools()
result = await tool.execute(
    action="analyze_pattern",
    amount=50000,
    sender="John Doe",
    recipient="External Account",
    historical_avg=5000
)

if result["success"]:
    analysis = result["data"]
    if analysis["anomaly_detected"]:
        print(f"ANOMALY: {analysis['anomaly_type']}")
        print(f"Risk Score: {analysis['risk_score']}")
        print(f"Recommendations: {analysis['recommendations']}")
```

### Example 4: Multi-Tool Chain

```python
# Step 1: Search for company information
search_result = await search_tool.execute(
    action="search",
    query="ACME Corp annual revenue"
)

# Step 2: Scrape company website
website = search_result["data"]["results"][0]["url"]
scrape_result = await browser_tool.execute(
    action="scrape",
    url=website
)

# Step 3: Get SEC filing
sec_result = await sec_tool.execute(
    action="get_filing",
    company="ACME CORP",
    filing_type="10-K"
)

# Step 4: Calculate financial ratios
calc_result = await calculator_tool.execute(
    action="calculate_ratios",
    **extracted_financial_data
)
```

---

## Development Guide

### Creating a New Tool

**Step 1: Create Tool File**

Create `backend/tools/my_new_tool.py`:

```python
from typing import Dict, List
from .base_tool import BaseTool

class MyNewTool(BaseTool):
    def __init__(self):
        self.name = "my_new_tool"
        self.description = "Description of what this tool does"
        self.required_args = ["arg1", "arg2"]
        self.category = "research"  # or compliance, risk, etc.
        self.capabilities = ["capability1", "capability2"]

    async def execute(self, **kwargs) -> Dict:
        """Execute the tool with provided arguments"""
        try:
            # Validate required arguments
            for arg in self.required_args:
                if arg not in kwargs:
                    return {
                        "success": False,
                        "error": f"Missing required argument: {arg}"
                    }

            # Tool logic here
            result = self._perform_action(kwargs)

            return {
                "success": True,
                "data": result
            }

        except Exception as e:
            # Fallback logic
            return {
                "success": True,
                "data": self._fallback_response(kwargs),
                "fallback": True,
                "error": str(e)
            }

    def _perform_action(self, args: Dict) -> Dict:
        """Main tool logic"""
        # Implementation here
        pass

    def _fallback_response(self, args: Dict) -> Dict:
        """Fallback when main logic fails"""
        # Mock or degraded response
        pass
```

**Step 2: Register Tool**

Tool is automatically discovered by the registry if placed in `tools/` directory.

**Step 3: Add to Team**

Assign tool to relevant teams in `agentic_teams.py`:

```python
team_tools = {
    "fraud": ["my_new_tool", ...],
    "compliance": ["my_new_tool", ...],
}
```

### Best Practices

1. **Always Implement Fallbacks**: Every tool must work without external APIs
2. **Clear Error Messages**: Provide helpful error messages for debugging
3. **Type Hints**: Use type hints for all methods
4. **Async Support**: Make tools async for non-blocking execution
5. **Documentation**: Document all parameters and return values
6. **Testing**: Write unit tests for both success and fallback paths
7. **API Key Validation**: Check for API keys before attempting external calls
8. **Rate Limiting**: Implement rate limiting for external APIs
9. **Caching**: Cache responses where appropriate
10. **Logging**: Log all tool executions for debugging

### Testing Tools

```python
import pytest
from tools.my_new_tool import MyNewTool

@pytest.mark.asyncio
async def test_tool_execution():
    tool = MyNewTool()
    result = await tool.execute(
        arg1="value1",
        arg2="value2"
    )
    assert result["success"] == True
    assert "data" in result

@pytest.mark.asyncio
async def test_tool_fallback():
    tool = MyNewTool()
    # Simulate API failure
    result = await tool.execute(
        arg1="invalid",
        arg2="invalid"
    )
    assert result["success"] == True  # Should still succeed
    assert result.get("fallback") == True  # But via fallback
```

---

## Environment Variables

All tools support optional API keys with automatic fallbacks:

```bash
# Investment Research Tools
SERPER_API_KEY=your_serper_api_key           # Google search
SEC_API_KEY=your_sec_api_key                 # SEC EDGAR (optional)
BROWSERLESS_API_KEY=your_browserless_key     # Web scraping

# Fraud Detection Tools
IPGEOLOCATION_API_KEY=your_ipgeo_key         # IP geolocation
ABSTRACTAPI_EMAIL_KEY=your_email_key         # Email validation

# Compliance Tools
# OFAC screening is FREE - no API key required
```

**If API keys are not provided**, tools automatically use fallback mechanisms:
- **Search**: Mock search results
- **SEC**: Free EDGAR HTML parsing
- **Browser**: HTTP GET with BeautifulSoup
- **IP Geo**: Mock geolocation data
- **Email**: Format-based validation

---

## Performance & Optimization

### Execution Times

Typical tool execution times:

| Tool | Avg Time | Notes |
|------|----------|-------|
| Calculator | <10ms | Pure calculation |
| Entity Resolver | <50ms | Local fuzzy matching |
| OFAC Screening | 200-500ms | Free API, no cache |
| Sanctions Screening | 300-600ms | Multiple databases |
| AML Tools | 100-300ms | Rule-based |
| SEC Tools | 1-3 seconds | External API call |
| Search Tools | 500ms-2s | Serper API |
| Browser Tools | 2-5 seconds | Page rendering |
| Transaction Tools | 100-200ms | Pattern analysis |

### Caching Strategy

Recommended caching for:
- **OFAC Results**: 24 hours (sanctions lists don't change frequently)
- **SEC Filings**: 7 days (filings don't change)
- **Search Results**: 1 hour (for same query)
- **Entity Resolution**: Permanent (for known entities)

### Parallel Execution

Tools support parallel execution for better performance:

```python
import asyncio

results = await asyncio.gather(
    sec_tool.execute(company="ACME", filing_type="10-K"),
    search_tool.execute(query="ACME Corp revenue"),
    browser_tool.execute(url="https://acme.com/investors")
)
```

---

## Troubleshooting

### Common Issues

**Issue**: Tool returns fallback response
**Solution**: Check API key configuration, verify network connectivity

**Issue**: Tool execution timeout
**Solution**: Increase timeout settings, use caching, check API rate limits

**Issue**: Tool not found in registry
**Solution**: Verify tool file in `tools/` directory, check class inheritance from `BaseTool`

**Issue**: Missing required arguments
**Solution**: Check tool's `required_args` property, provide all required parameters

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

View tool execution logs:
```bash
./logs.sh backend | grep "Tool"
```

---

## Future Enhancements

Planned improvements:

1. **More Tools**:
   - Credit scoring APIs
   - Social media analysis
   - Document OCR
   - Email verification

2. **Advanced Features**:
   - Tool chaining (automatic multi-tool workflows)
   - Result caching layer
   - Rate limit management
   - API key rotation

3. **Performance**:
   - Connection pooling
   - Request batching
   - Intelligent caching
   - Parallel execution optimization

4. **Monitoring**:
   - Tool usage analytics
   - Performance metrics
   - Failure tracking
   - Cost monitoring

---

## Support

For questions or issues:
- **API Docs**: http://localhost:8000/docs
- **Tool Registry**: `GET /api/tools/registry`
- **Logs**: Use `./logs.sh backend` for debugging
- **Main README**: See root README.md
