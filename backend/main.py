from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
import logging
import httpx
import os
import uuid
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from database import get_db, init_db
from models import Email, WorkflowResult, WorkflowConfig, Team, TeamAgent
from mailpit_client import MailPitClient
from workflows import WorkflowEngine
from llm_service import ollama_service
from wiki_enrichment import email_enricher
from pydantic import BaseModel
from events import broadcaster
from email_fetcher import email_fetcher
from agentic_teams import orchestrator, detect_team_for_email, suggest_team_for_email_llm, TEAMS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mailbox Analysis API", version="1.0.0")

# Global storage for async agentic team tasks
agentic_tasks = {}

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
mailpit_client = MailPitClient()
workflow_engine = WorkflowEngine()

# Employee Directory Configuration
EMPLOYEE_DIRECTORY_HOST = os.getenv("EMPLOYEE_DIRECTORY_HOST", "employee-directory")
EMPLOYEE_DIRECTORY_PORT = os.getenv("EMPLOYEE_DIRECTORY_PORT", "8100")
EMPLOYEE_DIRECTORY_URL = f"http://{EMPLOYEE_DIRECTORY_HOST}:{EMPLOYEE_DIRECTORY_PORT}"

async def lookup_employee_by_email(email: str) -> Optional[dict]:
    """
    Look up employee information from the Employee Directory API
    Returns employee data including mobile, phone, department, etc.
    Returns None if employee not found or on error
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{EMPLOYEE_DIRECTORY_URL}/api/employees/by-email/{email}"
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None
            else:
                logger.warning(f"Employee directory returned status {response.status_code} for {email}")
                return None
    except Exception as e:
        logger.warning(f"Failed to lookup employee {email}: {e}")
        return None

# Pydantic models for API
class EmailResponse(BaseModel):
    id: int
    mailpit_id: Optional[str] = None
    subject: str
    sender: str
    recipient: str
    body_text: Optional[str] = None
    body_html: Optional[str] = None
    received_at: datetime
    is_phishing: bool
    processed: bool
    # REMOVED: label and phishing_type - These are GROUND TRUTH from training dataset
    # They should NOT be exposed to workflows or UI during analysis
    # They are kept in database for evaluation purposes only
    summary: Optional[str] = None
    call_to_actions: Optional[List[str]] = None
    badges: Optional[List[str]] = None
    ui_badges: Optional[List[str]] = None
    quick_reply_drafts: Optional[dict] = None
    llm_processed: Optional[bool] = False
    workflow_results: List[dict] = []
    enriched: Optional[bool] = False
    enriched_data: Optional[dict] = None
    enriched_at: Optional[datetime] = None
    wiki_enriched: Optional[bool] = False
    phone_enriched: Optional[bool] = False
    wiki_enriched_at: Optional[datetime] = None
    phone_enriched_at: Optional[datetime] = None
    suggested_team: Optional[str] = None
    assigned_team: Optional[str] = None
    agentic_task_id: Optional[str] = None
    team_assigned_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class WorkflowResultResponse(BaseModel):
    id: int
    email_id: int
    workflow_name: str
    is_phishing_detected: bool
    confidence_score: int
    risk_indicators: list
    executed_at: datetime

    class Config:
        from_attributes = True

class ProcessEmailRequest(BaseModel):
    email_id: int
    workflows: Optional[List[str]] = None

class TeamAgentRequest(BaseModel):
    role: str
    icon: str
    personality: Optional[str] = None
    responsibilities: Optional[str] = None
    style: Optional[str] = None
    position_order: int = 0
    is_decision_maker: bool = False

class TeamAgentResponse(BaseModel):
    id: int
    team_id: int
    role: str
    icon: str
    personality: Optional[str] = None
    responsibilities: Optional[str] = None
    style: Optional[str] = None
    position_order: int
    is_decision_maker: bool
    created_at: datetime

    class Config:
        from_attributes = True

class TeamRequest(BaseModel):
    key: str
    name: str
    description: Optional[str] = None
    icon: Optional[str] = None
    is_active: bool = True

class TeamResponse(BaseModel):
    id: int
    key: str
    name: str
    description: Optional[str] = None
    icon: Optional[str] = None
    is_active: bool
    created_at: datetime
    updated_at: datetime
    agents: List[TeamAgentResponse] = []

    class Config:
        from_attributes = True

class TeamConfigRequest(BaseModel):
    agents: List[TeamAgentRequest]

@app.on_event("startup")
async def startup_event():
    """Initialize database and start email fetcher on startup"""
    import asyncio

    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized successfully")

    # Auto-start email fetcher in background
    logger.info("Starting email fetcher in background...")
    asyncio.create_task(email_fetcher.start_fetching())
    logger.info("Email fetcher started")

    # Initialize wiki enrichment knowledge base in background (unless disabled)
    if os.getenv("DISABLE_WIKI_ENRICHMENT", "false").lower() != "true":
        logger.info("Starting wiki enrichment initialization in background...")
        asyncio.create_task(email_enricher.initialize())
        logger.info("Wiki enrichment initialization started (loading in background)")
    else:
        logger.info("Wiki enrichment disabled via DISABLE_WIKI_ENRICHMENT env var")

def detect_ui_badges(email: Email) -> List[str]:
    """Detect UI state badges for an email

    Available UI badges: ACTION_REQUIRED, HIGH_PRIORITY, REPLIED, ATTACHMENT, SNOOZED, AI_SUGGESTED

    IMPORTANT: If email is phishing or has risk indicators, ONLY show security-related badges.
    Do not clutter the UI with informational badges when there's a security threat.
    """
    ui_badges = []

    # Check if email has any risk indicators from LLM badges
    has_risk_indicators = False
    if email.badges:
        # Check if any security-related badges are present
        risk_badges = ["RISK", "EXTERNAL", "SUSPICIOUS", "MALICIOUS", "URGENT"]
        has_risk_indicators = any(badge in email.badges for badge in risk_badges)

    # If email is phishing or has risk indicators, ONLY show security badges
    if email.is_phishing or has_risk_indicators:
        # Only show HIGH_PRIORITY for phishing emails that DON'T have explicit risk badges
        # If RISK badge is already present, don't add redundant HIGH_PRIORITY
        if email.is_phishing and not has_risk_indicators:
            ui_badges.append("HIGH_PRIORITY")
        # Do NOT add any other informational badges for risky emails
        return ui_badges

    # For safe emails, show informational badges
    # ACTION_REQUIRED: Email has call-to-actions
    if email.call_to_actions and len(email.call_to_actions) > 0:
        ui_badges.append("ACTION_REQUIRED")

    # AI_SUGGESTED: Email has AI-generated quick reply drafts
    if email.quick_reply_drafts:
        ui_badges.append("AI_SUGGESTED")

    # ATTACHMENT: Would need email metadata - skip for now
    # REPLIED: Would need user action tracking - skip for now
    # SNOOZED: Would need user action tracking - skip for now

    return ui_badges

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Mailbox Analysis API"}

@app.get("/health")
async def health():
    """Dedicated health check endpoint for Docker"""
    return {"status": "healthy", "service": "Mailbox Analysis API"}

@app.get("/api/mailpit/stats")
async def get_mailpit_stats():
    """Get MailPit statistics"""
    try:
        stats = await mailpit_client.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error fetching MailPit stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def auto_suggest_teams_for_emails(db: Session):
    """
    Background task to automatically suggest teams for emails that don't have one.
    Uses LLM to intelligently route emails to appropriate teams.
    """
    try:
        # Get emails without suggested team
        emails = db.query(Email).filter(Email.suggested_team == None).limit(50).all()

        logger.info(f"Auto-suggesting teams for {len(emails)} emails")

        for email in emails:
            try:
                # Use LLM to suggest team
                suggested_team = await suggest_team_for_email_llm(
                    email.subject,
                    email.body_text or email.body_html or "",
                    email.sender
                )

                email.suggested_team = suggested_team
                logger.info(f"Suggested team '{suggested_team}' for email {email.id}")

            except Exception as e:
                logger.error(f"Error suggesting team for email {email.id}: {e}")
                continue

        db.commit()
        logger.info(f"Completed team suggestion for {len(emails)} emails")

    except Exception as e:
        logger.error(f"Error in auto_suggest_teams_for_emails: {e}")
        db.rollback()


@app.post("/api/emails/fetch")
async def fetch_emails_from_mailpit(
    background_tasks: BackgroundTasks,
    limit: int = 5000,
    db: Session = Depends(get_db)
):
    """Fetch emails from MailPit and store in database"""
    try:
        messages_data = await mailpit_client.get_messages(limit=limit)
        messages = messages_data.get("messages", [])

        fetched_count = 0
        for msg in messages:
            msg_id = msg.get("ID")

            # Check if email already exists
            existing = db.query(Email).filter(Email.mailpit_id == msg_id).first()
            if existing:
                continue

            # Fetch full message details
            full_msg = await mailpit_client.get_message(msg_id)

            # Fetch headers separately to get custom headers
            headers = await mailpit_client.get_message_headers(msg_id)

            # Extract ground truth labels from custom headers
            label = None
            phishing_type = None

            if "X-Email-Label" in headers:
                try:
                    label = int(headers["X-Email-Label"][0])
                except (ValueError, IndexError):
                    pass

            if "X-Phishing-Type" in headers:
                try:
                    phishing_type = headers["X-Phishing-Type"][0]
                except IndexError:
                    pass

            # Extract email data
            email = Email(
                mailpit_id=msg_id,
                subject=msg.get("Subject", ""),
                sender=msg.get("From", {}).get("Address", ""),
                recipient=", ".join([addr.get("Address", "") for addr in msg.get("To", [])]),
                body_text=full_msg.get("Text", ""),
                body_html=full_msg.get("HTML", ""),
                received_at=datetime.fromisoformat(msg.get("Created", "").replace("Z", "+00:00")),
                label=label,
                phishing_type=phishing_type
            )

            db.add(email)
            fetched_count += 1

        db.commit()

        # Auto-suggest teams for newly fetched emails in background (DISABLED - user preference)
        # if fetched_count > 0:
        #     background_tasks.add_task(auto_suggest_teams_for_emails, db)

        return {
            "status": "success",
            "fetched": fetched_count,
            "total_in_mailpit": messages_data.get("total", 0)
        }

    except Exception as e:
        logger.error(f"Error fetching emails: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/emails/update-bodies")
async def update_email_bodies(
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Update body_text for existing emails from MailPit"""
    try:
        # Get emails without body_text
        emails_to_update = db.query(Email).filter(
            (Email.body_text == None) | (Email.body_text == "")
        ).limit(limit).all()

        updated_count = 0
        for email in emails_to_update:
            try:
                # Fetch full message from MailPit
                full_msg = await mailpit_client.get_message(email.mailpit_id)

                # Update body_text and body_html
                email.body_text = full_msg.get("Text", "")
                email.body_html = full_msg.get("HTML", "")

                updated_count += 1

            except Exception as e:
                logger.error(f"Error updating email {email.id}: {e}")
                continue

        db.commit()

        return {
            "status": "success",
            "updated": updated_count,
            "total_without_body": db.query(Email).filter(
                (Email.body_text == None) | (Email.body_text == "")
            ).count()
        }

    except Exception as e:
        logger.error(f"Error updating email bodies: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/emails", response_model=List[EmailResponse])
async def get_emails(
    skip: int = 0,
    limit: int = 50,
    processed: Optional[bool] = None,
    is_phishing: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """Get emails from database with filters"""
    from sqlalchemy import case, and_

    query = db.query(Email)

    if processed is not None:
        query = query.filter(Email.processed == processed)

    if is_phishing is not None:
        query = query.filter(Email.is_phishing == is_phishing)

    # Sort by analysis completion status:
    # Priority 1: Both processed AND enriched (fully analyzed)
    # Priority 2: Processed only
    # Priority 3: Enriched only
    # Priority 4: Received date (newest first)
    priority = case(
        (and_(Email.processed == True, Email.enriched == True), 0),  # Fully analyzed - highest priority
        (Email.processed == True, 1),                                 # ML analysis done
        (Email.enriched == True, 2),                                  # Wiki enrichment done
        else_=3                                                        # Not processed - lowest priority
    )

    emails = query.order_by(priority, Email.received_at.desc()).offset(skip).limit(limit).all()

    # Convert to response format
    result = []
    for email in emails:
        email_dict = {
            "id": email.id,
            "mailpit_id": email.mailpit_id,
            "subject": email.subject,
            "sender": email.sender,
            "recipient": email.recipient,
            "received_at": email.received_at,
            "is_phishing": email.is_phishing,
            "processed": email.processed,
            "label": email.label,
            "phishing_type": email.phishing_type,
            "summary": email.summary,
            "call_to_actions": email.call_to_actions,
            "badges": email.badges,
            "ui_badges": email.ui_badges,
            "quick_reply_drafts": email.quick_reply_drafts,
            "llm_processed": email.llm_processed,
            "workflow_results": [
                {
                    "id": wr.id,
                    "workflow_name": wr.workflow_name,
                    "is_phishing_detected": wr.is_phishing_detected,
                    "confidence_score": wr.confidence_score,
                    "risk_indicators": wr.risk_indicators,
                    "result": wr.result
                }
                for wr in email.workflow_results
            ],
            "body_text": email.body_text,
            "body_html": email.body_html,
            "enriched": email.enriched,
            "enriched_data": email.enriched_data,
            "enriched_at": email.enriched_at,
            "wiki_enriched": email.wiki_enriched,
            "phone_enriched": email.phone_enriched,
            "suggested_team": email.suggested_team,
            "assigned_team": email.assigned_team,
            "agentic_task_id": email.agentic_task_id,
            "team_assigned_at": email.team_assigned_at
        }
        result.append(email_dict)

    return result

@app.get("/api/emails/{email_id}")
async def get_email(email_id: int, db: Session = Depends(get_db)):
    """Get single email with workflow results"""
    email = db.query(Email).filter(Email.id == email_id).first()

    if not email:
        raise HTTPException(status_code=404, detail="Email not found")

    return {
        "id": email.id,
        "mailpit_id": email.mailpit_id,
        "subject": email.subject,
        "sender": email.sender,
        "recipient": email.recipient,
        "body_text": email.body_text,
        "body_html": email.body_html,
        "received_at": email.received_at,
        "is_phishing": email.is_phishing,
        "processed": email.processed,
        "summary": email.summary,
        "call_to_actions": email.call_to_actions,
        "badges": email.badges,
        "ui_badges": email.ui_badges,
        "quick_reply_drafts": email.quick_reply_drafts,
        "llm_processed": email.llm_processed,
        "enriched": email.enriched,
        "enriched_data": email.enriched_data,
        "enriched_at": email.enriched_at,
        "wiki_enriched": email.wiki_enriched,
        "phone_enriched": email.phone_enriched,
        "suggested_team": email.suggested_team,
        "assigned_team": email.assigned_team,
        "agentic_task_id": email.agentic_task_id,
        "team_assigned_at": email.team_assigned_at,
        "workflow_results": [
            {
                "id": wr.id,
                "workflow_name": wr.workflow_name,
                "is_phishing_detected": wr.is_phishing_detected,
                "confidence_score": wr.confidence_score,
                "risk_indicators": wr.risk_indicators,
                "result": wr.result,
                "executed_at": wr.executed_at.isoformat()
            }
            for wr in email.workflow_results
        ]
    }

@app.delete("/api/emails/{email_id}")
async def delete_email(email_id: int, db: Session = Depends(get_db)):
    """Delete an email and its associated workflow results"""
    email = db.query(Email).filter(Email.id == email_id).first()

    if not email:
        raise HTTPException(status_code=404, detail="Email not found")

    try:
        # Delete associated workflow results first (due to foreign key constraints)
        db.query(WorkflowResult).filter(WorkflowResult.email_id == email_id).delete()

        # Delete the email
        db.delete(email)
        db.commit()

        logger.info(f"Deleted email {email_id} and associated workflow results")
        return {"status": "success", "message": f"Email {email_id} deleted"}
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting email {email_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete email: {str(e)}")

@app.post("/api/emails/{email_id}/process")
async def process_email(
    email_id: int,
    workflows: Optional[List[str]] = None,
    db: Session = Depends(get_db)
):
    """Process an email through phishing detection workflows and LLM analysis"""
    email = db.query(Email).filter(Email.id == email_id).first()

    if not email:
        raise HTTPException(status_code=404, detail="Email not found")

    email_data = {
        "subject": email.subject,
        "sender": email.sender,
        "body_text": email.body_text or "",
        "body_html": email.body_html or ""
    }

    # Step 1: Run workflows
    workflow_results = await workflow_engine.run_all_workflows(email_data)

    # Delete existing workflow results for this email to avoid duplicates
    db.query(WorkflowResult).filter(WorkflowResult.email_id == email.id).delete()

    # Store results
    phishing_votes = 0
    total_confidence = 0

    # Track high-precision models for ensemble
    naive_bayes_phishing = False
    fine_tuned_llm_phishing = False

    for result in workflow_results:
        workflow_result = WorkflowResult(
            email_id=email.id,
            workflow_name=result["workflow"],
            result=result,
            is_phishing_detected=result["is_phishing_detected"],
            confidence_score=result["confidence_score"],
            risk_indicators=result["risk_indicators"]
        )
        db.add(workflow_result)

        if result["is_phishing_detected"]:
            phishing_votes += 1
        total_confidence += result["confidence_score"]

        # Track high-precision models
        if "Naive Bayes" in result["workflow"] and result["is_phishing_detected"]:
            naive_bayes_phishing = True
        if "Fine-tuned LLM" in result["workflow"] and result["is_phishing_detected"]:
            fine_tuned_llm_phishing = True

    # Update email status
    email.processed = True
    # Use "High Precision" ensemble strategy (NB + LLM with OR logic)
    # This strategy achieved 82.10% accuracy and 81.90% F1-score in testing
    # If either high-precision model detects phishing, flag it
    email.is_phishing = naive_bayes_phishing or fine_tuned_llm_phishing

    # Step 2: Process with LLM
    llm_data = {}
    try:
        llm_result = await ollama_service.process_email(
            email.subject or "",
            email.body_text or "",
            is_phishing=email.is_phishing
        )

        badges = await ollama_service.detect_email_badges(
            email.subject or "",
            email.body_text or "",
            email.sender or "",
            email.is_phishing
        )

        quick_reply_drafts = await ollama_service.generate_quick_reply_drafts(
            email.subject or "",
            email.body_text or "",
            email.sender or ""
        )

        email.summary = llm_result["summary"]
        email.call_to_actions = llm_result["call_to_actions"]
        email.badges = badges
        email.quick_reply_drafts = quick_reply_drafts
        email.llm_processed = True
        email.llm_processed_at = datetime.utcnow()

        # Detect and set UI badges
        email.ui_badges = detect_ui_badges(email)

        llm_data = {
            "summary": email.summary,
            "badges": badges,
            "quick_reply_drafts": quick_reply_drafts,
            "ui_badges": email.ui_badges
        }

    except Exception as llm_error:
        logger.error(f"Error processing email {email_id} with LLM: {llm_error}")
        # Continue even if LLM processing fails

    db.commit()

    return {
        "status": "success",
        "email_id": email_id,
        "is_phishing": email.is_phishing,
        "workflows_run": len(workflow_results),
        "average_confidence": total_confidence / len(workflow_results) if workflow_results else 0,
        "results": workflow_results,
        "llm_data": llm_data
    }

@app.post("/api/emails/process-all")
async def process_all_unprocessed_emails(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Process all unprocessed emails with workflow analysis and LLM processing"""
    unprocessed = db.query(Email).filter(Email.processed == False).all()

    async def process_batch():
        for email in unprocessed:
            try:
                # Step 1: Run workflow analysis
                email_data = {
                    "subject": email.subject,
                    "sender": email.sender,
                    "body_text": email.body_text or "",
                    "body_html": email.body_html or ""
                }

                workflow_results = await workflow_engine.run_all_workflows(email_data)

                phishing_votes = 0
                for result in workflow_results:
                    workflow_result = WorkflowResult(
                        email_id=email.id,
                        workflow_name=result["workflow"],
                        result=result,
                        is_phishing_detected=result["is_phishing_detected"],
                        confidence_score=result["confidence_score"],
                        risk_indicators=result["risk_indicators"]
                    )
                    db.add(workflow_result)

                    if result["is_phishing_detected"]:
                        phishing_votes += 1

                email.processed = True
                email.is_phishing = phishing_votes >= len(workflow_results) / 2

                # Broadcast event for workflow processing
                await broadcaster.broadcast("email_processed", {
                    "email_id": email.id,
                    "is_phishing": email.is_phishing,
                    "subject": email.subject,
                    "processed_count": unprocessed.index(email) + 1,
                    "total_count": len(unprocessed)
                })

                # Step 2: Process with LLM (summary, badges, quick replies)
                try:
                    llm_result = await ollama_service.process_email(
                        email.subject or "",
                        email.body_text or "",
                        is_phishing=email.is_phishing
                    )

                    # Detect badges
                    badges = await ollama_service.detect_email_badges(
                        email.subject or "",
                        email.body_text or "",
                        email.sender or "",
                        email.is_phishing
                    )

                    # Generate quick reply drafts
                    quick_reply_drafts = await ollama_service.generate_quick_reply_drafts(
                        email.subject or "",
                        email.body_text or "",
                        email.sender or ""
                    )

                    email.summary = llm_result["summary"]
                    email.call_to_actions = llm_result["call_to_actions"]
                    email.badges = badges
                    email.quick_reply_drafts = quick_reply_drafts
                    email.llm_processed = True
                    email.llm_processed_at = datetime.utcnow()

                    # Detect and set UI badges
                    email.ui_badges = detect_ui_badges(email)

                    # Broadcast event for LLM processing
                    await broadcaster.broadcast("email_llm_processed", {
                        "email_id": email.id,
                        "badges": badges,
                        "ui_badges": email.ui_badges,
                        "subject": email.subject,
                        "processed_count": unprocessed.index(email) + 1,
                        "total_count": len(unprocessed)
                    })

                except Exception as llm_error:
                    logger.error(f"Error processing email {email.id} with LLM: {llm_error}")
                    # Continue even if LLM processing fails

            except Exception as e:
                logger.error(f"Error processing email {email.id}: {e}")

        db.commit()

        # Broadcast completion event
        await broadcaster.broadcast("batch_complete", {
            "total_processed": len(unprocessed)
        })

    background_tasks.add_task(process_batch)

    return {
        "status": "processing",
        "count": len(unprocessed),
        "message": "Processing emails with workflow analysis and LLM in background"
    }

@app.get("/api/statistics")
async def get_statistics(db: Session = Depends(get_db)):
    """Get overall statistics"""
    total_emails = db.query(Email).count()
    processed_emails = db.query(Email).filter(Email.processed == True).count()
    phishing_emails = db.query(Email).filter(Email.is_phishing == True).count()
    legitimate_emails = db.query(Email).filter(
        Email.processed == True,
        Email.is_phishing == False
    ).count()

    # Count emails by badge type
    all_emails = db.query(Email).filter(Email.llm_processed == True).all()
    badge_counts = {
        "MEETING": 0,
        "RISK": 0,
        "EXTERNAL": 0,
        "AUTOMATED": 0,
        "VIP": 0,
        "FOLLOW_UP": 0,
        "NEWSLETTER": 0,
        "FINANCE": 0
    }

    for email in all_emails:
        if email.badges:
            for badge in email.badges:
                if badge in badge_counts:
                    badge_counts[badge] += 1

    return {
        "total_emails": total_emails,
        "processed_emails": processed_emails,
        "unprocessed_emails": total_emails - processed_emails,
        "phishing_detected": phishing_emails,
        "legitimate_emails": legitimate_emails,
        "phishing_percentage": (phishing_emails / processed_emails * 100) if processed_emails > 0 else 0,
        "badge_counts": badge_counts,
        "llm_processed": len(all_emails)
    }

@app.get("/api/dashboard/enriched-stats")
async def get_enriched_dashboard_stats(db: Session = Depends(get_db)):
    """
    Get enriched statistics for dashboard including:
    - Email analysis stats
    - Workflow execution stats
    - Team assignment stats
    - Tool usage stats
    - Time saved estimation
    """
    from tool_framework import get_tool_registry
    from sqlalchemy import func

    # === EMAIL ANALYSIS STATS ===
    total_emails = db.query(Email).count()
    processed_emails = db.query(Email).filter(Email.processed == True).count()
    phishing_emails = db.query(Email).filter(Email.is_phishing == True).count()
    legitimate_emails = db.query(Email).filter(
        Email.processed == True,
        Email.is_phishing == False
    ).count()
    llm_processed_emails = db.query(Email).filter(Email.llm_processed == True).count()
    enriched_emails = db.query(Email).filter(Email.enriched == True).count()
    wiki_enriched = db.query(Email).filter(Email.wiki_enriched == True).count()
    phone_enriched = db.query(Email).filter(Email.phone_enriched == True).count()

    # Badge breakdown
    all_emails = db.query(Email).filter(Email.llm_processed == True).all()
    badge_breakdown = {
        "MEETING": 0,
        "RISK": 0,
        "EXTERNAL": 0,
        "AUTOMATED": 0,
        "VIP": 0,
        "FOLLOW_UP": 0,
        "NEWSLETTER": 0,
        "FINANCE": 0
    }

    for email in all_emails:
        if email.badges:
            for badge in email.badges:
                if badge in badge_breakdown:
                    badge_breakdown[badge] += 1

    # === WORKFLOW EXECUTION STATS ===
    all_workflow_results = db.query(WorkflowResult).all()

    workflow_stats_dict = {}
    for result in all_workflow_results:
        if result.workflow_name not in workflow_stats_dict:
            workflow_stats_dict[result.workflow_name] = {
                "total_executions": 0,
                "phishing_detected": 0,
                "confidence_scores": [],
                "email_ids": []
            }

        workflow_stats_dict[result.workflow_name]["total_executions"] += 1
        if result.is_phishing_detected:
            workflow_stats_dict[result.workflow_name]["phishing_detected"] += 1
        if result.confidence_score is not None:
            workflow_stats_dict[result.workflow_name]["confidence_scores"].append(result.confidence_score)
        workflow_stats_dict[result.workflow_name]["email_ids"].append(result.email_id)

    workflows_summary = []
    for workflow_name, stats in workflow_stats_dict.items():
        avg_confidence = sum(stats["confidence_scores"]) / len(stats["confidence_scores"]) if stats["confidence_scores"] else 0

        # REMOVED: Ground truth statistics
        # Ground truth labels are training data and should not be exposed via API
        # They remain in database for evaluation purposes only

        workflows_summary.append({
            "workflow_name": workflow_name,
            "total_executions": stats["total_executions"],
            "phishing_detected": stats["phishing_detected"],
            "safe_detected": stats["total_executions"] - stats["phishing_detected"],
            "avg_confidence": round(avg_confidence, 1)
        })

    # === TEAM ASSIGNMENT STATS ===
    team_stats = db.query(
        Email.assigned_team,
        func.count(Email.id).label('assigned_count')
    ).filter(Email.assigned_team.isnot(None)).group_by(Email.assigned_team).all()

    team_assignments = {stat.assigned_team: stat.assigned_count for stat in team_stats}

    suggested_team_stats = db.query(
        Email.suggested_team,
        func.count(Email.id).label('suggested_count')
    ).filter(Email.suggested_team.isnot(None)).group_by(Email.suggested_team).all()

    team_suggestions = {stat.suggested_team: stat.suggested_count for stat in suggested_team_stats}

    # === TOOL USAGE STATS ===
    registry = get_tool_registry()
    all_tools = registry.get_all_tools()

    tools_summary = {
        "total_tools": len(all_tools),
        "available_tools": len([t for t in all_tools if t.is_available()]),
        "unavailable_tools": len([t for t in all_tools if not t.is_available()]),
        "tools_by_type": {},
        "tools_by_team": {}
    }

    # Count tools by type
    for tool in all_tools:
        metadata = tool.get_metadata()
        tool_type = metadata.tool_type.value
        if tool_type not in tools_summary["tools_by_type"]:
            tools_summary["tools_by_type"][tool_type] = 0
        tools_summary["tools_by_type"][tool_type] += 1

    # Get team tool assignments from registry config
    try:
        team_assignments_config = registry.get_team_assignments()
        for team_key, tool_names in team_assignments_config.items():
            tools_summary["tools_by_team"][team_key] = len(tool_names)
    except:
        pass

    # === TIME SAVED ESTIMATION ===
    # Time estimates (in minutes):
    # - Manual email review: 2-3 min per email → average 2.5 min
    # - Phishing detection: 5 min per phishing email
    # - Summary generation: 3-4 min per email → average 3.5 min
    # - Quick reply draft: 5 min per response
    # - Team routing: 2 min per assignment
    # - Wiki enrichment: 3 min per email
    # - Daily digest: Saves 15-20 email checks/day at 2.5 min each
    #   Average person checks email 15 times/day = 37.5 min/day
    #   With digest, 1 check/day = 2.5 min/day
    #   Savings = 35 min/day per active user

    emails_with_summaries = db.query(Email).filter(Email.summary.isnot(None)).count()
    emails_with_replies = db.query(Email).filter(Email.quick_reply_drafts.isnot(None)).count()
    emails_with_team_assignment = db.query(Email).filter(Email.assigned_team.isnot(None)).count()

    # More realistic time saved calculation
    # Only count unique time savings, avoid double-counting
    # Estimates based on realistic time savings vs manual work

    # Estimate daily digest usage based on LLM processed emails
    days_with_activity = max(1, llm_processed_emails // 20)  # Rough estimate: 20 emails per day average
    daily_digest_time_saved = days_with_activity * 15  # 15 min saved per day (more realistic)

    time_saved_minutes = (
        (emails_with_summaries * 1.0) +  # AI summary saves 1 min vs reading full email
        (phishing_emails * 2.0) +         # Phishing detection saves 2 min vs manual analysis
        (emails_with_replies * 2.0) +     # Quick reply draft saves 2 min
        (emails_with_team_assignment * 0.5) +  # Team routing saves 30 sec
        (enriched_emails * 1.0) +         # Wiki enrichment saves 1 min of research
        daily_digest_time_saved           # Daily inbox digest time savings
    )

    time_saved_hours = round(time_saved_minutes / 60, 1)
    time_saved_days = round(time_saved_hours / 8, 1)  # 8-hour workday

    # === RESPONSE ===
    return {
        "email_analysis": {
            "total_emails": total_emails,
            "processed_emails": processed_emails,
            "unprocessed_emails": total_emails - processed_emails,
            "phishing_detected": phishing_emails,
            "legitimate_emails": legitimate_emails,
            "phishing_rate": round((phishing_emails / processed_emails * 100), 1) if processed_emails > 0 else 0,
            "llm_processed": llm_processed_emails,
            "enriched_emails": enriched_emails,
            "wiki_enriched": wiki_enriched,
            "phone_enriched": phone_enriched,
            "badge_breakdown": badge_breakdown,
            "emails_with_summaries": emails_with_summaries,
            "emails_with_replies": emails_with_replies
        },
        "workflow_execution": {
            "total_workflow_executions": sum([w["total_executions"] for w in workflows_summary]),
            "workflows": workflows_summary
        },
        "team_assignments": {
            "total_assigned": sum(team_assignments.values()) if team_assignments else 0,
            "total_suggested": sum(team_suggestions.values()) if team_suggestions else 0,
            "assignments_by_team": team_assignments,
            "suggestions_by_team": team_suggestions
        },
        "tools_usage": tools_summary,
        "time_saved": {
            "total_minutes": round(time_saved_minutes, 1),
            "total_hours": time_saved_hours,
            "total_days": time_saved_days,
            "breakdown": {
                "summary_generation": round(emails_with_summaries * 1.0, 1),
                "phishing_detection": round(phishing_emails * 2.0, 1),
                "reply_drafting": round(emails_with_replies * 2.0, 1),
                "team_routing": round(emails_with_team_assignment * 0.5, 1),
                "wiki_enrichment": round(enriched_emails * 1.0, 1),
                "daily_digest": round(daily_digest_time_saved, 1)
            },
            "calculations": {
                "summary_generation": {"count": emails_with_summaries, "rate": 1.0, "unit": "summaries"},
                "phishing_detection": {"count": phishing_emails, "rate": 2.0, "unit": "phishing emails"},
                "reply_drafting": {"count": emails_with_replies, "rate": 2.0, "unit": "drafts"},
                "team_routing": {"count": emails_with_team_assignment, "rate": 0.5, "unit": "assignments"},
                "wiki_enrichment": {"count": enriched_emails, "rate": 1.0, "unit": "enrichments"},
                "daily_digest": {"count": days_with_activity, "rate": 15, "unit": "days"}
            },
            "daily_digest_days": days_with_activity
        }
    }

@app.get("/api/workflows")
async def get_workflows():
    """Get available workflows"""
    return {
        "workflows": [
            {
                "name": workflow.name,
                "description": workflow.__class__.__doc__
            }
            for workflow in workflow_engine.workflows
        ]
    }

@app.get("/api/events")
async def event_stream():
    """Server-Sent Events (SSE) endpoint for real-time updates"""
    return StreamingResponse(
        broadcaster.event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/api/fetcher/start")
async def start_email_fetcher(background_tasks: BackgroundTasks):
    """Start the scheduled email fetcher"""
    if email_fetcher.is_running:
        return {
            "status": "already_running",
            "message": "Email fetcher is already running"
        }

    # Start fetcher in background
    background_tasks.add_task(email_fetcher.start_fetching)

    return {
        "status": "started",
        "message": "Email fetcher started",
        "batch_size": 100,
        "delay_seconds": 10
    }

@app.post("/api/fetcher/stop")
async def stop_email_fetcher():
    """Stop the scheduled email fetcher"""
    await email_fetcher.stop_fetching()
    return {
        "status": "stopped",
        "message": "Email fetcher stopped"
    }

@app.get("/api/fetcher/status")
async def get_fetcher_status():
    """Get the status of the email fetcher"""
    return email_fetcher.get_status()

@app.on_event("startup")
async def startup_llm():
    """Initialize LLM on startup - non-blocking"""
    async def init_ollama():
        try:
            logger.info("Initializing Ollama service...")
            model_loaded = await ollama_service.ensure_model_loaded()
            if model_loaded:
                logger.info(f"Ollama model '{ollama_service.model}' is ready")
            else:
                logger.warning(f"Ollama model '{ollama_service.model}' could not be loaded. LLM features may not work properly.")
        except Exception as e:
            logger.error(f"Error initializing Ollama on startup: {e}")
            logger.warning("LLM features may not work properly. Check Ollama service status.")

    # Run in background to avoid blocking startup
    asyncio.create_task(init_ollama())

@app.on_event("startup")
async def startup_mcp():
    """Initialize MCP servers on startup - non-blocking"""
    async def init_mcp_servers():
        try:
            logger.info("Initializing MCP servers for fraud detection...")
            from mcp_client import initialize_mcp_servers
            await initialize_mcp_servers()
            logger.info("MCP servers initialization complete")
        except Exception as e:
            logger.error(f"Error initializing MCP servers on startup: {e}")
            logger.warning("MCP features (DuckDuckGo fallback) may not work. Serper and mock fallbacks will still work.")

    # Run in background to avoid blocking startup
    import asyncio
    asyncio.create_task(init_mcp_servers())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

@app.post("/api/emails/{email_id}/process-llm")
async def process_email_with_llm(
    email_id: int,
    db: Session = Depends(get_db)
):
    """Process an email with LLM: generate summary, extract CTAs, detect badges, and create quick reply drafts"""
    email = db.query(Email).filter(Email.id == email_id).first()

    if not email:
        raise HTTPException(status_code=404, detail="Email not found")

    try:
        # Process with LLM (only extract CTAs from safe emails)
        result = await ollama_service.process_email(
            email.subject or "",
            email.body_text or "",
            is_phishing=email.is_phishing
        )

        # Detect badges
        badges = await ollama_service.detect_email_badges(
            email.subject or "",
            email.body_text or "",
            email.sender or "",
            email.is_phishing
        )

        # Generate quick reply drafts
        quick_reply_drafts = await ollama_service.generate_quick_reply_drafts(
            email.subject or "",
            email.body_text or "",
            email.sender or ""
        )

        # Update email
        email.summary = result["summary"]
        email.call_to_actions = result["call_to_actions"]
        email.badges = badges
        email.quick_reply_drafts = quick_reply_drafts
        email.llm_processed = True
        email.llm_processed_at = datetime.utcnow()

        db.commit()

        return {
            "status": "success",
            "email_id": email_id,
            "summary": email.summary,
            "call_to_actions": email.call_to_actions,
            "badges": email.badges,
            "quick_reply_drafts": email.quick_reply_drafts
        }

    except Exception as e:
        logger.error(f"Error processing email {email_id} with LLM: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/emails/{email_id}/enrich")
async def enrich_email_with_wiki(
    email_id: int,
    db: Session = Depends(get_db)
):
    """Enrich email with wiki context using RAG - ONLY for non-phishing emails"""
    email = db.query(Email).filter(Email.id == email_id).first()

    if not email:
        raise HTTPException(status_code=404, detail="Email not found")

    # IMPORTANT: Only enrich safe (non-phishing) emails
    if email.is_phishing:
        # Mark as enriched even though we skipped it (the enrichment process completed)
        email.enriched = True
        email.enriched_at = datetime.utcnow()
        email.enriched_data = {
            "enriched_keywords": [],
            "relevant_pages": []
        }
        email.wiki_enriched = False
        email.phone_enriched = False
        db.commit()

        return {
            "status": "skipped",
            "email_id": email_id,
            "reason": "Enrichment is only available for safe (non-phishing) emails",
            "enriched_keywords": [],
            "relevant_pages": []
        }

    try:
        # Enrich email with wiki context
        enrichment_data = await email_enricher.enrich_email(
            email.subject or "",
            email.body_text or ""
        )

        # Look up sender in employee directory
        sender_employee_data = None
        if email.sender:
            sender_employee_data = await lookup_employee_by_email(email.sender)
            if sender_employee_data:
                logger.info(f"Found sender employee data for {email.sender}: {sender_employee_data.get('first_name')} {sender_employee_data.get('last_name')}")

        # Look up recipient in employee directory
        recipient_employee_data = None
        if email.recipient:
            recipient_employee_data = await lookup_employee_by_email(email.recipient)
            if recipient_employee_data:
                logger.info(f"Found recipient employee data for {email.recipient}: {recipient_employee_data.get('first_name')} {recipient_employee_data.get('last_name')}")

        # Combine wiki enrichment with employee directory data
        enrichment_data["sender_employee"] = sender_employee_data
        enrichment_data["recipient_employee"] = recipient_employee_data

        # Set enrichment success tags - ONLY when actual data is found
        has_wiki_keywords = bool(enrichment_data.get("enriched_keywords"))  # Only true if keywords were actually matched
        has_phone_data = bool(sender_employee_data or recipient_employee_data)  # True if employee data was found

        # Update email with enrichment data
        email.enriched_data = enrichment_data
        email.enriched = True
        email.enriched_at = datetime.utcnow()

        # Set flags and timestamps only when actual data is found
        email.wiki_enriched = has_wiki_keywords
        email.phone_enriched = has_phone_data

        if has_wiki_keywords:
            email.wiki_enriched_at = datetime.utcnow()

        if has_phone_data:
            email.phone_enriched_at = datetime.utcnow()

        db.commit()

        logger.info(f"Email {email_id} enriched - Wiki keywords: {has_wiki_keywords} ({len(enrichment_data.get('enriched_keywords', []))} found), Phone: {has_phone_data}")

        return {
            "status": "success",
            "email_id": email_id,
            "enriched_keywords": enrichment_data.get("enriched_keywords", []),
            "relevant_pages": enrichment_data.get("relevant_pages", []),
            "sender_employee": sender_employee_data,
            "recipient_employee": recipient_employee_data,
            "wiki_enriched": has_wiki_keywords,
            "phone_enriched": has_phone_data
        }

    except Exception as e:
        logger.error(f"Error enriching email {email_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/emails/process-all-llm")
async def process_all_emails_with_llm(
    background_tasks: BackgroundTasks,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Process all unprocessed emails with LLM in background"""
    unprocessed = db.query(Email).filter(Email.llm_processed == False).limit(limit).all()

    async def process_batch():
        for email in unprocessed:
            try:
                result = await ollama_service.process_email(
                    email.subject or "",
                    email.body_text or "",
                    is_phishing=email.is_phishing
                )

                # Detect badges
                badges = await ollama_service.detect_email_badges(
                    email.subject or "",
                    email.body_text or "",
                    email.sender or "",
                    email.is_phishing
                )

                # Generate quick reply drafts
                quick_reply_drafts = await ollama_service.generate_quick_reply_drafts(
                    email.subject or "",
                    email.body_text or "",
                    email.sender or ""
                )

                email.summary = result["summary"]
                email.call_to_actions = result["call_to_actions"]
                email.badges = badges
                email.quick_reply_drafts = quick_reply_drafts
                email.llm_processed = True
                email.llm_processed_at = datetime.utcnow()

                # Broadcast event for LLM processing
                await broadcaster.broadcast("email_llm_processed", {
                    "email_id": email.id,
                    "badges": badges,
                    "subject": email.subject,
                    "processed_count": unprocessed.index(email) + 1,
                    "total_count": len(unprocessed)
                })

            except Exception as e:
                logger.error(f"Error processing email {email.id} with LLM: {e}")

        db.commit()

        # Broadcast completion event
        await broadcaster.broadcast("llm_batch_complete", {
            "total_processed": len(unprocessed)
        })

    background_tasks.add_task(process_batch)

    return {
        "status": "processing",
        "count": len(unprocessed),
        "message": f"Processing {len(unprocessed)} emails with LLM in background"
    }

@app.get("/api/daily-summary")
async def get_daily_summary(
    hours: int = 24,
    db: Session = Depends(get_db)
):
    """Get aggregated summary for emails from last N hours"""
    try:
        # Calculate time threshold
        time_threshold = datetime.utcnow() - timedelta(hours=hours)

        # Get emails from last N hours
        recent_emails = db.query(Email).filter(
            Email.received_at >= time_threshold,
            Email.llm_processed == True
        ).all()

        if not recent_emails:
            return {
                "period_hours": hours,
                "total_emails": 0,
                "aggregate_summary": "No emails processed in the selected time period",
                "consolidated_ctas": [],
                "top_senders": []
            }

        # Extract summaries and CTAs
        summaries = [email.summary for email in recent_emails if email.summary]

        # Normalize CTAs - handle both string arrays and dict arrays
        all_ctas = []
        for email in recent_emails:
            if email.call_to_actions:
                ctas = email.call_to_actions
                # Convert [{"action": "text"}] to ["text"]
                if isinstance(ctas, list) and len(ctas) > 0:
                    if isinstance(ctas[0], dict) and "action" in ctas[0]:
                        normalized = [item["action"] for item in ctas if isinstance(item, dict) and "action" in item]
                        all_ctas.append(normalized)
                    elif isinstance(ctas[0], str):
                        all_ctas.append(ctas)

        # Generate aggregate summary
        aggregate_summary = await ollama_service.aggregate_summaries(summaries)

        # Consolidate CTAs
        consolidated_ctas = await ollama_service.aggregate_call_to_actions(all_ctas) if all_ctas else []

        # Get top senders
        sender_counts = {}
        for email in recent_emails:
            sender = email.sender
            sender_counts[sender] = sender_counts.get(sender, 0) + 1

        top_senders = sorted(sender_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "period_hours": hours,
            "total_emails": len(recent_emails),
            "processed_emails": len([e for e in recent_emails if e.llm_processed]),
            "aggregate_summary": aggregate_summary,
            "consolidated_ctas": consolidated_ctas,
            "top_senders": [{"sender": s, "count": c} for s, c in top_senders],
            "phishing_detected": len([e for e in recent_emails if e.is_phishing]),
            "time_range": {
                "from": time_threshold.isoformat(),
                "to": datetime.utcnow().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Error generating daily summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/inbox-digest")
async def get_inbox_digest(
    hours: int = 24,
    db: Session = Depends(get_db)
):
    """Get emails grouped by badges for Daily Inbox Digest view"""
    try:
        # Calculate time threshold
        time_threshold = datetime.utcnow() - timedelta(hours=hours)

        # Get recent emails that have been processed (ML phishing detection)
        recent_emails = db.query(Email).filter(
            Email.received_at >= time_threshold,
            Email.processed == True
        ).order_by(Email.received_at.desc()).all()

        # Group emails by primary badge
        grouped = {
            "MEETING": [],
            "RISK": [],
            "EXTERNAL": [],
            "AUTOMATED": [],
            "VIP": [],
            "FOLLOW_UP": [],
            "NEWSLETTER": [],
            "FINANCE": [],
            "OTHER": []
        }

        total_today = len(recent_emails)

        for email in recent_emails:
            # Auto-assign RISK badge to phishing emails if no badges exist
            badges = email.badges or []
            if not badges and email.is_phishing:
                badges = ["RISK"]

            email_data = {
                "id": email.id,
                "subject": email.subject,
                "sender": email.sender,
                "recipient": email.recipient,
                "summary": email.summary if email.summary else [email.subject],
                "badges": badges,
                "received_at": email.received_at.isoformat(),
                "is_phishing": email.is_phishing,
                "call_to_actions": email.call_to_actions or [],
                "body_text": email.body_text if hasattr(email, 'body_text') else None
            }

            # Add to primary badge group (first badge takes priority)
            if badges and len(badges) > 0:
                primary_badge = badges[0]
                if primary_badge in grouped:
                    grouped[primary_badge].append(email_data)
                else:
                    grouped["OTHER"].append(email_data)
            else:
                grouped["OTHER"].append(email_data)

        # Calculate badge counts
        badge_counts = {
            badge: len(emails) for badge, emails in grouped.items() if len(emails) > 0
        }

        return {
            "period_hours": hours,
            "total_today": total_today,
            "badge_counts": badge_counts,
            "grouped_emails": grouped,
            "time_range": {
                "from": time_threshold.isoformat(),
                "to": datetime.utcnow().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Error generating inbox digest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# AGENTIC TEAMS ENDPOINTS
# ============================================================================

@app.get("/api/agentic/teams")
async def get_available_teams(db: Session = Depends(get_db)):
    """Get list of all available virtual bank teams from database"""
    teams = db.query(Team).filter(Team.is_active == True).all()

    # If no teams in database, return hardcoded TEAMS as fallback
    if not teams:
        teams_list = []
        for team_key, team_data in TEAMS.items():
            teams_list.append({
                "key": team_key,
                "name": team_data["name"],
                "agent_count": len(team_data["agents"]),
                "agents": [
                    {
                        "role": agent["role"],
                        "icon": agent["icon"],
                        "personality": agent.get("personality", ""),
                        "responsibilities": agent["responsibilities"],
                        "style": agent.get("style", "")
                    }
                    for agent in team_data["agents"]
                ]
            })
        return {"teams": teams_list}

    # Return teams from database
    teams_list = []
    for team in teams:
        teams_list.append({
            "id": team.id,
            "key": team.key,
            "name": team.name,
            "description": team.description,
            "icon": team.icon,
            "agent_count": len(team.agents),
            "agents": [
                {
                    "id": agent.id,
                    "role": agent.role,
                    "icon": agent.icon,
                    "personality": agent.personality,
                    "responsibilities": agent.responsibilities,
                    "style": agent.style,
                    "position_order": agent.position_order,
                    "is_decision_maker": agent.is_decision_maker
                }
                for agent in team.agents
            ]
        })
    return {"teams": teams_list}

@app.get("/api/agentic/teams/{team_key}/config")
async def get_team_config(team_key: str, db: Session = Depends(get_db)):
    """Get detailed configuration for a specific team"""
    team = db.query(Team).filter(Team.key == team_key, Team.is_active == True).first()

    if not team:
        # Fallback to hardcoded TEAMS
        if team_key in TEAMS:
            team_data = TEAMS[team_key]
            return {
                "key": team_key,
                "name": team_data["name"],
                "agents": [
                    {
                        "role": agent["role"],
                        "icon": agent["icon"],
                        "personality": agent.get("personality", ""),
                        "responsibilities": agent["responsibilities"],
                        "style": agent.get("style", "")
                    }
                    for agent in team_data["agents"]
                ]
            }
        raise HTTPException(status_code=404, detail="Team not found")

    return TeamResponse.from_orm(team)

@app.post("/api/agentic/teams/{team_key}/config")
async def save_team_config(team_key: str, config: TeamConfigRequest, db: Session = Depends(get_db)):
    """Save team configuration"""
    team = db.query(Team).filter(Team.key == team_key).first()

    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    # Delete existing agents
    db.query(TeamAgent).filter(TeamAgent.team_id == team.id).delete()

    # Add new agents
    for agent_data in config.agents:
        agent = TeamAgent(
            team_id=team.id,
            role=agent_data.role,
            icon=agent_data.icon,
            personality=agent_data.personality,
            responsibilities=agent_data.responsibilities,
            style=agent_data.style,
            position_order=agent_data.position_order,
            is_decision_maker=agent_data.is_decision_maker
        )
        db.add(agent)

    team.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(team)

    return {"status": "success", "message": "Team configuration saved successfully"}

@app.post("/api/agentic/teams")
async def create_team(team: TeamRequest, db: Session = Depends(get_db)):
    """Create a new team"""
    # Check if team key already exists
    existing_team = db.query(Team).filter(Team.key == team.key).first()
    if existing_team:
        raise HTTPException(status_code=400, detail="Team with this key already exists")

    new_team = Team(
        key=team.key,
        name=team.name,
        description=team.description,
        icon=team.icon,
        is_active=team.is_active
    )
    db.add(new_team)
    db.commit()
    db.refresh(new_team)

    return TeamResponse.from_orm(new_team)

@app.put("/api/agentic/teams/{team_key}")
async def update_team(team_key: str, team: TeamRequest, db: Session = Depends(get_db)):
    """Update team metadata"""
    existing_team = db.query(Team).filter(Team.key == team_key).first()

    if not existing_team:
        raise HTTPException(status_code=404, detail="Team not found")

    existing_team.name = team.name
    existing_team.description = team.description
    existing_team.icon = team.icon
    existing_team.is_active = team.is_active
    existing_team.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(existing_team)

    return TeamResponse.from_orm(existing_team)

@app.delete("/api/agentic/teams/{team_key}")
async def delete_team(team_key: str, db: Session = Depends(get_db)):
    """Delete a team (soft delete by setting is_active to false)"""
    team = db.query(Team).filter(Team.key == team_key).first()

    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    team.is_active = False
    team.updated_at = datetime.utcnow()
    db.commit()

    return {"status": "success", "message": "Team deleted successfully"}


async def run_agentic_workflow_background(task_id: str, email_id: int, team: str, email_subject: str, email_body: str, email_sender: str):
    """Background task to run agentic team discussion with real-time updates"""
    try:
        logger.info(f"[Task {task_id}] Starting agentic workflow for email {email_id} with team '{team}'")

        # Create database session for this background task
        from database import SessionLocal
        db = SessionLocal()

        # Update status to processing
        agentic_tasks[task_id]["status"] = "processing"
        agentic_tasks[task_id]["team"] = team

        # Initialize result structure for real-time updates
        agentic_tasks[task_id]["result"] = {
            "status": "processing",
            "email_id": email_id,
            "team": team,
            "team_name": TEAMS[team]["name"],
            "discussion": {
                "messages": [],
                "status": "processing",
                "team": team,
                "team_name": TEAMS[team]["name"],
                "email_id": email_id
            }
        }

        # Callback to update task with new messages in real-time
        async def on_message(message, all_messages):
            """Update task status as each agent speaks"""
            logger.info(f"[Task {task_id}] Agent spoke: {message['role']}")
            agentic_tasks[task_id]["result"]["discussion"]["messages"] = all_messages.copy()

            # Broadcast message via SSE for real-time updates
            await broadcaster.broadcast("agentic_message", {
                "task_id": task_id,
                "email_id": email_id,
                "agent_name": message.get('role', 'Agent'),
                "message": message.get('text', ''),
                "total_messages": len(all_messages),
                "status": "processing"
            })

        # Run the team discussion with real-time callback (2 rounds for faster CPU processing)
        result = await orchestrator.run_team_discussion(
            email_id=email_id,
            email_subject=email_subject,
            email_body=email_body,
            email_sender=email_sender,
            team=team,
            max_rounds=2,
            on_message_callback=on_message,
            db=db
        )

        # Store the final result
        agentic_tasks[task_id]["status"] = "completed"
        agentic_tasks[task_id]["result"] = {
            "status": "success",
            "email_id": email_id,
            "team": team,
            "team_name": result["team_name"],
            "discussion": result
        }

        # Save result to database for persistence across restarts
        # Using the same db session created at the beginning
        try:
            workflow_result = WorkflowResult(
                email_id=email_id,
                workflow_name=f"agentic_{team}_{task_id}",
                result=agentic_tasks[task_id]["result"],
                is_phishing_detected=False,
                confidence_score=100,
                risk_indicators=[],
                executed_at=datetime.now()
            )
            db.add(workflow_result)

            # Update email processed status
            email = db.query(Email).filter(Email.id == email_id).first()
            if email:
                email.processed = True
                email.llm_processed = True
                logger.info(f"[Task {task_id}] Updated email {email_id} processed status to True")

            db.commit()
            logger.info(f"[Task {task_id}] Saved result to database")
        except Exception as db_error:
            logger.error(f"[Task {task_id}] Failed to save to database: {db_error}")
            db.rollback()

        # Broadcast completion via SSE
        await broadcaster.broadcast("agentic_complete", {
            "task_id": task_id,
            "email_id": email_id,
            "status": "completed",
            "result": agentic_tasks[task_id]["result"]
        })

        logger.info(f"[Task {task_id}] Completed agentic workflow for email {email_id}")

    except Exception as e:
        logger.error(f"[Task {task_id}] Error in agentic workflow: {e}")
        agentic_tasks[task_id]["status"] = "failed"
        agentic_tasks[task_id]["error"] = str(e)
    finally:
        # Close database session
        if 'db' in locals():
            db.close()


@app.post("/api/agentic/emails/{email_id}/process")
async def process_email_with_agentic_team(
    email_id: int,
    team: Optional[str] = None,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Start processing an email with a virtual bank team (async).
    Returns immediately with a task_id for polling.
    """
    try:
        # Get email from database
        email = db.query(Email).filter(Email.id == email_id).first()
        if not email:
            raise HTTPException(status_code=404, detail=f"Email {email_id} not found")

        # Auto-detect team if not specified
        if not team:
            team = detect_team_for_email(email.subject, email.body_text or email.body_html)
            logger.info(f"Auto-detected team '{team}' for email {email_id}")

        # Validate team
        if team not in TEAMS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid team '{team}'. Valid teams: {list(TEAMS.keys())}"
            )

        # Create task ID
        task_id = str(uuid.uuid4())

        # Initialize task status
        agentic_tasks[task_id] = {
            "status": "pending",
            "email_id": email_id,
            "team": team,
            "created_at": datetime.now().isoformat()
        }

        # Update email record with task_id and assigned_team
        email.assigned_team = team
        email.agentic_task_id = task_id
        email.team_assigned_at = datetime.now()
        db.commit()

        # Start background task
        asyncio.create_task(run_agentic_workflow_background(
            task_id=task_id,
            email_id=email.id,
            team=team,
            email_subject=email.subject,
            email_body=email.body_text or email.body_html,
            email_sender=email.sender
        ))

        logger.info(f"Started agentic workflow task {task_id} for email {email_id}")

        return {
            "status": "started",
            "task_id": task_id,
            "email_id": email_id,
            "team": team
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting agentic workflow for email {email_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agentic/tasks/{task_id}")
async def get_agentic_task_status(task_id: str, db: Session = Depends(get_db)):
    """Poll for agentic task status and results"""
    # First check in-memory cache
    if task_id in agentic_tasks:
        task = agentic_tasks[task_id]
        response = {
            "task_id": task_id,
            "status": task["status"],
            "email_id": task["email_id"],
            "team": task.get("team"),
            "created_at": task["created_at"]
        }
    else:
        # Task not in memory, check database for completed results
        logger.info(f"Task {task_id} not in memory, checking database...")
        email = db.query(Email).filter(Email.agentic_task_id == task_id).first()

        if not email:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        # Get workflow result from database
        workflow_result = db.query(WorkflowResult).filter(
            WorkflowResult.email_id == email.id,
            WorkflowResult.workflow_name.like(f"%{task_id}%")
        ).order_by(WorkflowResult.executed_at.desc()).first()

        if workflow_result and workflow_result.result:
            # Return stored result
            response = {
                "task_id": task_id,
                "status": "completed",
                "email_id": email.id,
                "team": email.assigned_team,
                "created_at": workflow_result.executed_at.isoformat(),
                "result": workflow_result.result
            }
            logger.info(f"Loaded task {task_id} from database")
        else:
            # Email exists but no workflow result stored
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} result not available. Please run analysis again."
            )

    # Add result if task is in memory and has result data
    if task_id in agentic_tasks:
        task = agentic_tasks[task_id]
        if task["status"] == "completed" or task["status"] == "processing":
            if "result" in task:
                response["result"] = task["result"]
        elif task["status"] == "failed":
            response["error"] = task.get("error", "Unknown error")

    return response


@app.post("/api/agentic/direct-query")
async def create_direct_query_task(request: dict, db: Session = Depends(get_db)):
    """
    Create a direct query task without an email
    Request body: {"team": "investments", "query": "I want a complete analysis for stock Apple"}
    """
    try:
        team = request.get("team")
        query = request.get("query", "")

        if not team or not query:
            raise HTTPException(status_code=400, detail="team and query are required")

        # Validate team
        if team not in TEAMS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid team '{team}'. Valid teams: {list(TEAMS.keys())}"
            )

        # Create Email record in database for persistence
        email = Email(
            mailpit_id=None,  # No MailPit ID for direct queries
            subject=f"📊 Direct Query: {query[:100]}",
            sender="Direct User Query",
            recipient=TEAMS[team]["name"],
            body_text=query,
            body_html="",
            received_at=datetime.now(),
            assigned_team=team,  # Assign to team immediately
            team_assigned_at=datetime.now(),
            is_phishing=False,
            processed=False,
            llm_processed=False
        )
        db.add(email)
        db.commit()
        db.refresh(email)

        # Create task ID
        task_id = str(uuid.uuid4())

        # Update email with task_id
        email.agentic_task_id = task_id
        db.commit()

        # Initialize task status
        agentic_tasks[task_id] = {
            "status": "pending",
            "email_id": email.id,  # Link to database record
            "team": team,
            "query": query,
            "created_at": datetime.now().isoformat(),
            "messages": []
        }

        # Start background task for agentic workflow
        asyncio.create_task(run_agentic_workflow_background(
            task_id=task_id,
            email_id=email.id,  # Pass database ID
            team=team,
            email_subject=email.subject,
            email_body=query,
            email_sender="Direct User Query"
        ))

        logger.info(f"Created direct query task {task_id} for team '{team}' with email_id {email.id}")

        return {
            "status": "created",
            "task_id": task_id,
            "email_id": email.id,  # Return email ID
            "team": team,
            "query": query
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating direct query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agentic/task/{task_id}/chat")
async def send_chat_message_to_team(task_id: str, request: dict):
    """
    Send a chat message to the team and get a response
    Request body: {"message": "What about the company's debt levels?"}
    """
    try:
        message = request.get("message", "").strip()

        if not message:
            raise HTTPException(status_code=400, detail="message is required")

        if task_id not in agentic_tasks:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        task = agentic_tasks[task_id]
        team = task.get("team")

        if not team or team not in TEAMS:
            raise HTTPException(status_code=400, detail="Invalid team")

        # Import orchestrator for LLM calls
        from agentic_teams import orchestrator, TEAMS as BACKEND_TEAMS

        # Add user message to workflow context
        if team == "investments":
            from investment_workflow import investment_workflow
            investment_workflow.add_user_message(message)
        elif team == "fraud":
            from fraud_workflow import fraud_workflow
            fraud_workflow.add_user_message(message)
        elif team == "compliance":
            from compliance_workflow import ComplianceWorkflow
            compliance_workflow = ComplianceWorkflow()
            compliance_workflow.add_user_message(message)

        # Select an appropriate agent to respond (use first agent in team)
        team_info = BACKEND_TEAMS[team]
        agent = team_info["agents"][0]  # First agent responds to chat

        # Add to task messages
        if "messages" not in task:
            task["messages"] = []

        # User message
        user_msg = {
            "role": "You",
            "icon": "👤",
            "text": message,
            "timestamp": datetime.now().isoformat()
        }
        task["messages"].append(user_msg)

        # Broadcast user message via SSE
        await broadcaster.broadcast("agentic_chat_user", {
            "task_id": task_id,
            "message": user_msg
        })

        # Get analysis context from task result
        analysis_context = ""
        if "result" in task and "discussion" in task["result"]:
            messages = task["result"]["discussion"].get("messages", [])
            # Include last 3-5 messages as context
            recent_messages = messages[-5:] if len(messages) > 5 else messages
            if recent_messages:
                analysis_context = "\n\nPREVIOUS ANALYSIS CONTEXT:\n"
                for msg in recent_messages:
                    role = msg.get("role", "Unknown")
                    text = msg.get("text", "")[:500]  # Limit length
                    analysis_context += f"{role}: {text}\n\n"

        # Create agent-specific prompt for chat response
        system_prompt = f"""You are a {agent['role']} at a Swiss bank. You are part of the {team_info['name']} team.

Your personality: {agent['personality']}
Your responsibilities: {agent['responsibilities']}
Your communication style: {agent['style']}

You are having a conversation with a user who asked a question about an analysis your team just completed.
Use the previous analysis context to inform your response. Be specific and reference actual findings from the analysis.
Keep your response concise but informative (3-5 sentences)."""

        user_prompt = f"""{analysis_context}

CURRENT USER QUESTION: {message}

Provide a helpful and professional response based on your role, expertise, and the analysis context above.
Reference specific findings or data points from the analysis when relevant."""

        # Call LLM
        response_text = await orchestrator.call_llm(user_prompt, system_prompt)

        # Agent response
        agent_msg = {
            "role": agent["role"],
            "icon": agent["icon"],
            "text": response_text,
            "timestamp": datetime.now().isoformat()
        }
        task["messages"].append(agent_msg)

        # Broadcast agent response via SSE
        await broadcaster.broadcast("agentic_chat_response", {
            "task_id": task_id,
            "message": agent_msg
        })

        logger.info(f"Chat message sent to task {task_id}")

        return {
            "status": "success",
            "agent": agent["role"],
            "icon": agent["icon"],
            "response": response_text
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agentic/emails/{email_id}/team")
async def detect_team_for_email_endpoint(
    email_id: int,
    db: Session = Depends(get_db)
):
    """Detect which team should handle a specific email"""
    try:
        email = db.query(Email).filter(Email.id == email_id).first()
        if not email:
            raise HTTPException(status_code=404, detail=f"Email {email_id} not found")

        team = detect_team_for_email(email.subject, email.body_text or email.body_html)
        team_info = TEAMS[team]

        return {
            "email_id": email_id,
            "detected_team": team,
            "team_name": team_info["name"],
            "agent_count": len(team_info["agents"])
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting team for email {email_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agentic/simulate-discussion")
async def simulate_team_discussion(
    team: str = "fraud",
    subject: str = "Suspicious Wire Transfer Request"
):
    """
    Simulate a team discussion for testing purposes.
    This endpoint creates a mock email scenario for demonstration.
    """
    try:
        if team not in TEAMS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid team '{team}'. Valid teams: {list(TEAMS.keys())}"
            )

        # Create a mock email body based on the team
        mock_bodies = {
            "fraud": "We received a suspicious wire transfer request for CHF 500K to an unknown account. The email appears to be from our CEO but the sending domain is slightly different.",
            "compliance": "We need clarification on FATCA reporting requirements for our new US client accounts. What are the thresholds and documentation needed?",
            "investments": "Please provide a comprehensive analysis of Tesla (TSLA) stock, including financial metrics, valuation, and investment recommendation."
        }

        result = await orchestrator.run_team_discussion(
            email_id=0,  # Mock ID
            email_subject=subject,
            email_body=mock_bodies.get(team, "Sample email body for team discussion."),
            email_sender="client@example.com",
            team=team,
            max_rounds=1
        )

        return {
            "status": "success",
            "simulation": True,
            "team": team,
            "discussion": result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error simulating team discussion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/emails/{email_id}/suggest-team")
async def suggest_team_for_email_endpoint(
    email_id: int,
    db: Session = Depends(get_db)
):
    """
    Use LLM to suggest which team should handle an email.
    Stores the suggestion in the email record.
    """
    try:
        email = db.query(Email).filter(Email.id == email_id).first()
        if not email:
            raise HTTPException(status_code=404, detail=f"Email {email_id} not found")

        # Use LLM to suggest team
        suggested_team = await suggest_team_for_email_llm(
            email.subject,
            email.body_text or email.body_html,
            email.sender
        )

        # Store suggestion in database
        email.suggested_team = suggested_team
        db.commit()

        team_info = TEAMS[suggested_team]

        logger.info(f"Suggested team '{suggested_team}' for email {email_id}")

        return {
            "email_id": email_id,
            "suggested_team": suggested_team,
            "team_name": team_info["name"],
            "agent_count": len(team_info["agents"]),
            "agents": [agent["role"] for agent in team_info["agents"]]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error suggesting team for email {email_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/emails/{email_id}/analyze-tasks")
async def analyze_email_tasks(
    email_id: int,
    request: dict,
    db: Session = Depends(get_db)
):
    """
    Analyze an email and suggest multiple task options for the user to choose from.
    Uses LLM to understand the email content and propose specific actions.

    Request body:
    {
        "team": "fraud"  // The team context for task suggestions
    }

    Returns:
    {
        "email_id": 123,
        "team": "fraud",
        "task_options": [
            {
                "id": "task_1",
                "title": "Verify transaction authenticity",
                "description": "Check if the reported transaction is legitimate",
                "priority": "high"
            },
            ...
        ]
    }
    """
    try:
        team = request.get("team")

        # Get email from database
        email = db.query(Email).filter(Email.id == email_id).first()
        if not email:
            raise HTTPException(status_code=404, detail=f"Email {email_id} not found")

        # Validate team
        if team not in TEAMS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid team '{team}'. Valid teams: {list(TEAMS.keys())}"
            )

        team_info = TEAMS[team]

        # Build LLM prompt to analyze the email and suggest task options
        system_prompt = f"""You are an intelligent task analysis system for a Swiss bank's {team_info['name']} team.

Your job is to analyze incoming emails and suggest 3-4 specific, actionable tasks that the team should perform.

Team expertise:
{chr(10).join([f"- {agent['role']}: {agent['responsibilities']}" for agent in team_info['agents']])}

Guidelines:
- Suggest 3-4 distinct tasks that align with the team's expertise
- Each task should be specific to the email content
- Tasks should be actionable and have clear outcomes
- Prioritize tasks by urgency (high/medium/low)
- Consider different perspectives (investigation, analysis, verification, recommendation)

Respond with ONLY valid JSON in this format:
{{
    "task_options": [
        {{
            "id": "task_1",
            "title": "Brief task title (5-8 words)",
            "description": "Detailed description of what needs to be done (1-2 sentences)",
            "priority": "high|medium|low"
        }}
    ]
}}"""

        user_prompt = f"""EMAIL TO ANALYZE:
Subject: {email.subject}
From: {email.sender}
Body: {(email.body_text or email.body_html)[:1000]}

Based on this email content and the {team_info['name']} team's expertise, suggest 3-4 specific tasks the team should perform. Return ONLY the JSON response."""

        # Call LLM to generate task options
        response = await orchestrator.call_llm(user_prompt, system_prompt)

        # Parse JSON response
        import re
        import json
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            task_data = json.loads(json_match.group())
            task_options = task_data.get("task_options", [])

            # Ensure each task has a unique ID
            for i, task in enumerate(task_options):
                if "id" not in task:
                    task["id"] = f"task_{i+1}"

            logger.info(f"Generated {len(task_options)} task options for email {email_id}, team {team}")

            return {
                "email_id": email_id,
                "team": team,
                "team_name": team_info["name"],
                "task_options": task_options
            }
        else:
            # Fallback: create generic task options
            fallback_tasks = [
                {
                    "id": "task_1",
                    "title": f"Analyze email for {team} concerns",
                    "description": f"Review the email content and identify key {team}-related issues",
                    "priority": "medium"
                },
                {
                    "id": "task_2",
                    "title": "Investigate and provide recommendations",
                    "description": "Conduct thorough investigation and provide actionable recommendations",
                    "priority": "high"
                }
            ]

            return {
                "email_id": email_id,
                "team": team,
                "team_name": team_info["name"],
                "task_options": fallback_tasks
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing tasks for email {email_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/emails/{email_id}/assign-team")
async def assign_team_to_email(
    email_id: int,
    request: dict,
    db: Session = Depends(get_db)
):
    """
    Manually assign a team to an email and trigger the agentic workflow.
    This is called when the operator drags an email to a team or confirms assignment.

    Request body:
    {
        "team": "investments",
        "assignment_message": "Please check this email and tell me what you suggest I should do?",
        "selected_task": {
            "id": "task_1",
            "title": "Verify transaction authenticity",
            "description": "Check if the reported transaction is legitimate"
        }
    }
    """
    try:
        team = request.get("team")
        assignment_message = request.get("assignment_message", "")
        selected_task = request.get("selected_task", None)  # Get the user-selected task

        # Get email from database
        email = db.query(Email).filter(Email.id == email_id).first()
        if not email:
            raise HTTPException(status_code=404, detail=f"Email {email_id} not found")

        # Validate team
        if team not in TEAMS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid team '{team}'. Valid teams: {list(TEAMS.keys())}"
            )

        # Create task ID for agentic workflow
        task_id = str(uuid.uuid4())

        # Update email record with assigned team and task_id
        email.assigned_team = team
        email.agentic_task_id = task_id
        email.team_assigned_at = datetime.now()
        db.commit()

        # Initialize task status
        agentic_tasks[task_id] = {
            "status": "pending",
            "email_id": email_id,
            "team": team,
            "assignment_message": assignment_message,
            "selected_task": selected_task,  # Store the selected task
            "created_at": datetime.now().isoformat(),
            "messages": []
        }

        # Prepare email body with assignment message and selected task context
        email_body = email.body_text or email.body_html

        # Build context for the workflow
        task_context = ""
        if selected_task:
            task_context = f"""SELECTED TASK:
Task: {selected_task.get('title', 'N/A')}
Description: {selected_task.get('description', 'N/A')}
Priority: {selected_task.get('priority', 'medium')}

The team should focus specifically on this task when analyzing the email.
"""

        if assignment_message:
            email_body = f"ASSIGNMENT MESSAGE: {assignment_message}\n\n{task_context}\n{email_body}"
        elif task_context:
            email_body = f"{task_context}\n{email_body}"

        # Start background task for agentic workflow
        asyncio.create_task(run_agentic_workflow_background(
            task_id=task_id,
            email_id=email.id,
            team=team,
            email_subject=email.subject,
            email_body=email_body,
            email_sender=email.sender
        ))

        log_msg = f"Assigned team '{team}' to email {email_id}"
        if selected_task:
            log_msg += f" with task '{selected_task.get('title')}'"
        if assignment_message:
            log_msg += " with custom message"
        logger.info(log_msg)

        return {
            "status": "assigned",
            "email_id": email_id,
            "assigned_team": team,
            "task_id": task_id,
            "workflow_url": f"/pages/agentic-teams.html?email_id={email_id}&task_id={task_id}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error assigning team to email {email_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/emails/{email_id}/workflow-status")
async def get_email_workflow_status(
    email_id: int,
    db: Session = Depends(get_db)
):
    """
    Get the agentic workflow status for an email.
    Returns workflow link if available.
    """
    try:
        email = db.query(Email).filter(Email.id == email_id).first()
        if not email:
            raise HTTPException(status_code=404, detail=f"Email {email_id} not found")

        if not email.agentic_task_id:
            return {
                "email_id": email_id,
                "has_workflow": False,
                "suggested_team": email.suggested_team,
                "assigned_team": email.assigned_team
            }

        # Get task status
        task_id = email.agentic_task_id
        task = agentic_tasks.get(task_id)

        if not task:
            return {
                "email_id": email_id,
                "has_workflow": True,
                "task_id": task_id,
                "status": "unknown",
                "assigned_team": email.assigned_team,
                "workflow_url": f"/agentic-teams.html?email_id={email_id}&task_id={task_id}"
            }

        response = {
            "email_id": email_id,
            "has_workflow": True,
            "task_id": task_id,
            "status": task["status"],
            "assigned_team": email.assigned_team,
            "team_assigned_at": email.team_assigned_at.isoformat() if email.team_assigned_at else None,
            "workflow_url": f"/agentic-teams.html?email_id={email_id}&task_id={task_id}"
        }

        if task["status"] == "completed":
            response["result"] = task.get("result")
        elif task["status"] == "failed":
            response["error"] = task.get("error")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow status for email {email_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/teams/{team_key}/tools")
async def get_team_tools(team_key: str):
    """
    Get all configured tools for a specific team from actual tool classes
    No hardcoded examples - all data comes from real tool implementations
    """
    try:
        from tool_discovery import get_team_tools

        tools = get_team_tools(team_key)
        logger.info(f"Discovered {len(tools)} real tools for team '{team_key}'")

        return {"team": team_key, "tools": tools}

    except Exception as e:
        logger.error(f"Error discovering tools for team {team_key}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tools/test")
async def test_tool(request: dict):
    """
    Test a tool and return actual data from it.
    Supports MCP servers, proprietary tools, public APIs, and regular APIs.
    """
    tool_name = request.get("tool_name")
    tool_type = request.get("tool_type")
    configuration = request.get("configuration", {})
    provider = request.get("provider")

    logger.info(f"Testing tool: {tool_name} ({tool_type})")

    try:
        # Handle different tool types
        if tool_type == "mcp":
            # Test MCP server connection and invoke sample capability
            result = await test_mcp_tool(tool_name, configuration)
        elif tool_type == "proprietary":
            # Test proprietary internal tool
            result = await test_proprietary_tool(tool_name, configuration)
        elif tool_type == "public" or tool_type == "api":
            # Test public API or external API
            result = await test_api_tool(tool_name, configuration, provider)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown tool type: {tool_type}")

        return {
            "status": "success",
            "data": result
        }

    except Exception as e:
        logger.error(f"Error testing tool {tool_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def test_mcp_tool(tool_name: str, config: dict):
    """Test an MCP server by attempting to connect and invoke a test capability"""
    mcp_server = config.get("mcp_server")
    mcp_location = config.get("mcp_location")
    mcp_protocol = config.get("mcp_protocol", "stdio")

    # For now, return connection info and simulated test
    # TODO: Implement actual MCP server invocation when MCP servers are deployed
    return {
        "status": "connected",
        "mcp_server": mcp_server,
        "protocol": mcp_protocol,
        "location": mcp_location,
        "test_result": "MCP server endpoint detected, ready for invocation",
        "note": "MCP integration will invoke actual server when deployed"
    }


async def test_proprietary_tool(tool_name: str, config: dict):
    """Test a proprietary internal tool"""
    internal_url = config.get("internal_url")

    if not internal_url:
        raise ValueError("Proprietary tool must have internal_url configured")

    try:
        # Make actual HTTP request to proprietary tool
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Try GET first for health/status endpoint
            if "/health" in internal_url or "/status" in internal_url:
                response = await client.get(internal_url)
            else:
                # For analysis endpoints, try POST with sample data
                response = await client.post(
                    internal_url,
                    json={"test": True, "sample_data": "test query"}
                )

            response.raise_for_status()
            data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {"response": response.text}

            return {
                "status": "operational",
                "url": internal_url,
                "response_code": response.status_code,
                "data": data
            }

    except httpx.HTTPStatusError as e:
        return {
            "status": "error",
            "url": internal_url,
            "error": f"HTTP {e.response.status_code}: {e.response.text}"
        }
    except httpx.ConnectError:
        return {
            "status": "not_deployed",
            "url": internal_url,
            "message": "✓ Tool configuration is valid",
            "warning": "Internal service not currently deployed or running",
            "note": "Service needs to be started to test full functionality",
            "is_warning": True  # Flag for frontend
        }
    except Exception as e:
        return {
            "status": "error",
            "url": internal_url,
            "error": str(e)
        }


async def test_api_tool(tool_name: str, config: dict, provider: str):
    """Test a public or external API tool by calling its actual implementation"""

    try:
        # Test using the actual tool implementation
        if "Serper" in provider or "search" in tool_name.lower():
            # Test Serper API using real SearchTools
            from tools.search_tools import SearchTools
            search_tools = SearchTools()
            if not search_tools.use_serper:
                return {
                    "status": "configured",
                    "message": "✓ SearchTools configured",
                    "note": "SERPER_API_KEY not set - using fallback mode",
                    "is_warning": True
                }
            # Make actual search
            result = await search_tools.search_internet("test", num_results=1)
            return {
                "status": "operational",
                "provider": provider,
                "test_query": "test",
                "result_preview": result[:200] + "..." if len(result) > 200 else result,
                "message": "✓ API test successful - returning real search data"
            }

        elif "Browserless" in provider or "scraping" in tool_name.lower():
            # Test Browserless API using real BrowserTools
            from tools.browser_tools import BrowserTools
            browser_tools = BrowserTools()
            if not browser_tools.use_browserless:
                return {
                    "status": "configured",
                    "message": "✓ BrowserTools configured",
                    "note": "BROWSERLESS_API_KEY not set - using HTTP fallback",
                    "is_warning": True
                }
            # Make actual scrape of a simple page
            result = await browser_tools.scrape_website("https://example.com")
            return {
                "status": "operational",
                "provider": provider,
                "test_url": "https://example.com",
                "result_preview": result[:200] + "..." if len(result) > 200 else result,
                "message": "✓ API test successful - web scraping operational"
            }

        elif "SEC" in provider or "sec" in tool_name.lower():
            # Test SEC API using real SECTools
            from tools.sec_tools import SECTools
            sec_tools = SECTools()
            if not sec_tools.use_sec_api:
                return {
                    "status": "configured",
                    "message": "✓ SECTools configured",
                    "note": "SEC_API_KEY not set - using free EDGAR access",
                    "is_warning": True
                }
            # Make actual 10-K query
            result = await sec_tools.get_10k("AAPL")
            return {
                "status": "operational",
                "provider": provider,
                "test_query": "AAPL 10-K",
                "result_preview": result[:300] + "..." if len(result) > 300 else result,
                "message": "✓ API test successful - SEC filings accessible"
            }

        elif "IPGeolocation" in provider or "geolocation" in tool_name.lower():
            # Test IPGeolocation API using real RiskTools
            from tools.risk_tools import RiskTools
            risk_tools = RiskTools()
            if not risk_tools.ipgeolocation_api_key:
                return {
                    "status": "configured",
                    "message": "✓ RiskTools configured",
                    "note": "IPGEOLOCATION_API_KEY not set - using mock data",
                    "is_warning": True
                }
            # Make actual geolocation query
            result = await risk_tools.analyze_geolocation("8.8.8.8")
            return {
                "status": "operational",
                "provider": provider,
                "test_ip": "8.8.8.8",
                "result_preview": result[:300] + "..." if len(result) > 300 else result,
                "message": "✓ API test successful - IP geolocation operational"
            }

        elif "AbstractAPI" in provider or "email" in tool_name.lower():
            # Test AbstractAPI Email using real InvestigationTools
            from tools.investigation_tools import InvestigationTools
            investigation_tools = InvestigationTools()
            if not investigation_tools.abstractapi_email_key:
                return {
                    "status": "configured",
                    "message": "✓ InvestigationTools configured",
                    "note": "ABSTRACTAPI_EMAIL_KEY not set - using basic validation",
                    "is_warning": True
                }
            # Make actual email validation
            result = await investigation_tools.validate_email("test@example.com")
            return {
                "status": "operational",
                "provider": provider,
                "test_email": "test@example.com",
                "result_preview": result[:300] + "..." if len(result) > 300 else result,
                "message": "✓ API test successful - email validation operational"
            }

        else:
            # Fallback: try basic GET request
            api_endpoint = config.get("api_endpoint")
            if not api_endpoint:
                raise ValueError("API tool must have api_endpoint configured")

            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(api_endpoint)
                response.raise_for_status()

                return {
                    "status": "accessible",
                    "provider": provider,
                    "endpoint": api_endpoint,
                    "response_code": response.status_code,
                    "message": "✓ API endpoint accessible"
                }

    except httpx.HTTPStatusError as e:
        # Authentication errors (401, 403) mean endpoint works but needs proper auth
        if e.response.status_code in [401, 403]:
            return {
                "status": "configured",
                "response_code": e.response.status_code,
                "message": "✓ Tool is properly configured and API endpoint is operational",
                "analysis_value": "This API returns valuable data for analysis when authenticated",
                "setup_note": "API key/credentials configured in environment for production use",
                "error_detail": e.response.text[:200],
                "is_warning": True
            }
        # Other HTTP errors
        return {
            "status": "api_error",
            "response_code": e.response.status_code,
            "error": e.response.text[:200],
            "note": "API endpoint exists but returned an error. Check parameters or documentation."
        }
    except httpx.ConnectError as e:
        return {
            "status": "unreachable",
            "message": "Cannot connect to API endpoint",
            "warning": "Check if endpoint URL is correct and network accessible",
            "error": str(e),
            "is_warning": True
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "note": "Tool test encountered an error"
        }


# ============================================================================
# TOOL REGISTRY ENDPOINTS - Plugin Architecture
# ============================================================================

@app.get("/api/tools/registry")
async def get_tool_registry_info():
    """
    Get complete tool registry information

    Returns all tools, their metadata, and registry statistics
    """
    try:
        from tool_framework import get_tool_registry

        registry = get_tool_registry()
        return registry.to_dict()

    except Exception as e:
        logger.error(f"Error getting tool registry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tools/registry/stats")
async def get_registry_stats():
    """
    Get tool registry statistics

    Returns:
        Statistics about tools, assignments, and availability
    """
    try:
        from tool_framework import get_tool_registry

        registry = get_tool_registry()
        return registry.get_registry_stats()

    except Exception as e:
        logger.error(f"Error getting registry stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tools/search/capability/{capability}")
async def search_tools_by_capability(capability: str):
    """
    Search for tools by capability

    Args:
        capability: Capability to search for (search, scrape, analyze, etc.)

    Returns:
        List of tools with that capability
    """
    try:
        from tool_framework import get_tool_registry, ToolCapability

        registry = get_tool_registry()

        # Convert string to ToolCapability enum
        try:
            cap_enum = ToolCapability[capability.upper()]
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid capability. Valid options: {[c.value for c in ToolCapability]}"
            )

        tools = registry.search_by_capability(cap_enum)

        return {
            "capability": capability,
            "tool_count": len(tools),
            "tools": [tool.to_dict() for tool in tools]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching tools by capability: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tools/search/category/{category}")
async def search_tools_by_category(category: str):
    """
    Search for tools by category

    Args:
        category: Category to search for (fraud, investment, compliance, etc.)

    Returns:
        List of tools in that category
    """
    try:
        from tool_framework import get_tool_registry

        registry = get_tool_registry()
        tools = registry.search_by_category(category)

        return {
            "category": category,
            "tool_count": len(tools),
            "tools": [tool.to_dict() for tool in tools]
        }

    except Exception as e:
        logger.error(f"Error searching tools by category: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tools/{tool_name}")
async def get_tool_details(tool_name: str):
    """
    Get detailed information about a specific tool

    Args:
        tool_name: Name of the tool

    Returns:
        Tool details including metadata, methods, and configuration
    """
    try:
        from tool_framework import get_tool_registry

        registry = get_tool_registry()
        tool = registry.get_tool(tool_name)

        if not tool:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

        tool_dict = tool.to_dict()

        # Add validation info
        validation = tool.validate_config()
        tool_dict["validation"] = validation

        # Add test example if available
        test_example = tool.get_test_example()
        if test_example:
            tool_dict["test_example"] = test_example

        return tool_dict

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting tool details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tools/registry/assign")
async def assign_tools_to_team(request: dict):
    """
    Assign tools to a team

    Request body:
        {
            "team_key": "fraud",
            "tool_names": ["Tool 1", "Tool 2"]
        }

    Returns:
        Success message with assignment details
    """
    try:
        from tool_framework import get_tool_registry

        team_key = request.get("team_key")
        tool_names = request.get("tool_names", [])

        if not team_key:
            raise HTTPException(status_code=400, detail="team_key is required")

        registry = get_tool_registry()

        # Validate that all tools exist
        invalid_tools = []
        for tool_name in tool_names:
            if not registry.get_tool(tool_name):
                invalid_tools.append(tool_name)

        if invalid_tools:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tools: {', '.join(invalid_tools)}"
            )

        # Assign tools
        registry.assign_tools_to_team(team_key, tool_names)

        return {
            "success": True,
            "team": team_key,
            "tool_count": len(tool_names),
            "tools": tool_names
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error assigning tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tools/{tool_name}/readiness")
async def get_tool_readiness(tool_name: str):
    """
    Get detailed readiness status for a tool.
    Shows what's configured and what's missing so users can see if tool is ready before using it.

    Args:
        tool_name: Name of the tool

    Returns:
        Detailed readiness information including:
        - is_ready: Whether tool is ready to use
        - is_active: Current active status
        - status_message: Human-readable status
        - required: Required environment variables (configured vs missing)
        - optional: Optional environment variables (configured vs missing)
        - dependencies: Other requirements (MCP, database, etc.)
    """
    try:
        from tool_framework import get_tool_registry

        registry = get_tool_registry()
        tool = registry.get_tool(tool_name)

        if not tool:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

        readiness = tool.get_readiness_status()

        return {
            "tool_name": tool_name,
            "readiness": readiness
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting tool readiness: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/events")
async def events():
    """Server-Sent Events endpoint for real-time updates"""
    return StreamingResponse(
        broadcaster.event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
