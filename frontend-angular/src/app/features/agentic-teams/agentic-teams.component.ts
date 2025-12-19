import { Component, OnInit, inject, OnDestroy, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute } from '@angular/router';
import { Store } from '@ngrx/store';
import { Observable, Subject, combineLatest, BehaviorSubject } from 'rxjs';
import { takeUntil, map, filter, take } from 'rxjs/operators';
import { Email } from '../../core/models/email.model';
import { selectAllEmails } from '../../store';
import * as EmailsActions from '../../store/emails/emails.actions';
import { EmailService } from '../../core/services/email.service';
import { SseService } from '../../core/services/sse.service';
import { ApiService } from '../../core/services/api.service';
import { marked } from 'marked';

interface TeamMember {
  name: string;
  role: string;
  icon: string;
  memberType: string;
  personality: string;
  responsibilities: string;
  communicationStyle: string;
}

interface DiscussionMessage {
  agentName: string;
  agentIcon: string;
  content: string;
  timestamp: string;
  isToolUsage?: boolean;
  isDecision?: boolean;
}

interface Tool {
  name: string;
  type: 'mcp' | 'proprietary' | 'public' | 'api';
  description?: string;
  provider?: string;
  isActive: boolean;
  configuration: {
    [key: string]: string | number | boolean;
  };
}

interface ToolTestResult {
  success: boolean;
  isWarning?: boolean;
  data?: any;
  error?: string;
  warning?: string;
  message?: string;
  timestamp: string;
  responseTime?: number;
}

interface TeamInfo {
  name: string;
  key: string;
  badge: string;
  members: TeamMember[];
  tools: Tool[];
}

@Component({
  selector: 'app-agentic-teams',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './agentic-teams.component.html',
  styleUrl: './agentic-teams.component.scss'
})
export class AgenticTeamsComponent implements OnInit, OnDestroy {
  private route = inject(ActivatedRoute);
  private store = inject(Store);
  private emailService = inject(EmailService);
  private apiService = inject(ApiService);
  private sseService = inject(SseService);
  private cdr = inject(ChangeDetectorRef);
  private destroy$ = new Subject<void>();
  private selectedTeam$ = new BehaviorSubject<string | null>(null);

  selectedTeam: string | null = null;
  teamEmails$: Observable<Email[]>;
  selectedEmail: Email | null = null;
  chatMessage: string = '';
  directQuery: string = '';
  isSubmittingQuery: boolean = false;

  showDirectInteraction: boolean = false;
  showTeamPresentation: boolean = false;
  showWorkflowModal: boolean = false;
  isEditMode: boolean = false;
  expandedToolIndex: number | null = null;
  editingToolIndex: number | null = null;
  testingToolIndex: number | null = null;
  toolTestResults: { [key: number]: ToolTestResult } = {};

  // Tools Editor
  isEditingTools: boolean = false;
  registryTools: any[] = [];
  loadingRegistry: boolean = false;

  teams: { [key: string]: TeamInfo } = {
    'fraud': {
      name: 'Fraud Investigation Unit',
      key: 'fraud',
      badge: 'team-fraud',
      tools: [],  // Will be loaded from backend
      members: [
        {
          name: 'Fraud Detection Specialist',
          role: 'Identify suspicious patterns, transaction anomalies, and fraud indicators',
          icon: '🔍',
          memberType: 'Team Member',
          personality: 'Suspicious and investigative. Looks for red flags. Says \'I notice that...\' and \'This pattern suggests...\'',
          responsibilities: 'Identify suspicious patterns, transaction anomalies, and fraud indicators',
          communicationStyle: 'Skeptical, detail-focused, investigative'
        },
        {
          name: 'Forensic Analyst',
          role: 'Conduct technical analysis, trace transactions, analyze digital evidence',
          icon: '🧪',
          memberType: 'Team Member',
          personality: 'Technical and methodical. Deep dives into evidence. Uses phrases like \'The technical analysis shows...\' and \'Examining the metadata...\'',
          responsibilities: 'Conduct technical analysis, trace transactions, analyze digital evidence',
          communicationStyle: 'Technical, precise, methodical'
        },
        {
          name: 'Legal Advisor',
          role: 'Assess legal implications, regulatory requirements, evidence admissibility',
          icon: '⚖️',
          memberType: 'Team Member',
          personality: 'Cautious and procedural. Ensures compliance. Says \'From a legal standpoint...\' and \'We must ensure...\'',
          responsibilities: 'Assess legal implications, regulatory requirements, evidence admissibility',
          communicationStyle: 'Procedural, cautious, compliance-focused'
        },
        {
          name: 'Security Director',
          role: 'Decide on containment actions, client contact, law enforcement involvement',
          icon: '🛡️',
          memberType: 'Decision Maker',
          personality: 'Decisive and action-oriented. Makes containment decisions. Uses phrases like \'We need to immediately...\' and \'The priority is...\'',
          responsibilities: 'Decide on containment actions, client contact, law enforcement involvement',
          communicationStyle: 'Decisive, action-oriented, protective'
        }
      ]
    },
    'compliance': {
      name: 'Compliance & Regulatory Affairs',
      key: 'compliance',
      badge: 'team-compliance',
      tools: [],  // Will be loaded from backend
      members: [
        {
          name: 'Compliance Officer',
          role: 'Verify regulatory compliance, policy adherence, documentation requirements',
          icon: '📋',
          memberType: 'Team Member',
          personality: 'Rule-oriented and systematic. Checks regulations. Says \'According to regulation...\' and \'We must comply with...\'',
          responsibilities: 'Verify regulatory compliance, policy adherence, documentation requirements',
          communicationStyle: 'Systematic, rule-bound, thorough'
        },
        {
          name: 'Legal Counsel',
          role: 'Interpret regulations, assess legal risks, provide legal opinions',
          icon: '⚖️',
          memberType: 'Team Member',
          personality: 'Analytical and interpretive. Explains legal nuances. Uses phrases like \'The legal interpretation is...\' and \'From a liability perspective...\'',
          responsibilities: 'Interpret regulations, assess legal risks, provide legal opinions',
          communicationStyle: 'Analytical, interpretive, cautious'
        },
        {
          name: 'Auditor',
          role: 'Audit compliance processes, verify documentation, check audit trails',
          icon: '📊',
          memberType: 'Team Member',
          personality: 'Meticulous and verification-focused. Double-checks everything. Says \'Let me verify...\' and \'The audit trail shows...\'',
          responsibilities: 'Audit compliance processes, verify documentation, check audit trails',
          communicationStyle: 'Meticulous, verification-focused, detail-oriented'
        },
        {
          name: 'Regulatory Liaison',
          role: 'Determine reporting obligations, draft regulator communications, manage relationships',
          icon: '🏛️',
          memberType: 'Decision Maker',
          personality: 'Strategic and communicative. Manages regulator relationships. Uses phrases like \'Based on regulator expectations...\' and \'We should report...\'',
          responsibilities: 'Determine reporting obligations, draft regulator communications, manage relationships',
          communicationStyle: 'Strategic, communicative, proactive'
        }
      ]
    },
    'investments': {
      name: 'Investment Research Team',
      key: 'investments',
      badge: 'team-investments',
      tools: [],  // Will be loaded from backend
      members: [
        {
          name: 'Financial Analyst',
          role: 'Impress customers with financial data and market trends analysis',
          icon: '📊',
          memberType: 'Team Member',
          personality: 'Seasoned expert in stock market analysis. The Best Financial Analyst. Says \'The financial data shows...\' and \'Market trends indicate...\'',
          responsibilities: 'Impress customers with financial data and market trends analysis. Evaluate P/E ratio, EPS growth, revenue trends, and debt-to-equity metrics. Compare performance against industry peers.',
          communicationStyle: 'Expert, analytical, confident'
        },
        {
          name: 'Research Analyst',
          role: 'Excel at data gathering and interpretation',
          icon: '🔍',
          memberType: 'Team Member',
          personality: 'Known as the BEST research analyst. Skilled in sifting through news, company announcements, and market sentiments. Says \'The research shows...\' and \'Looking at recent developments...\'',
          responsibilities: 'Excel at data gathering and interpretation. Compile recent news, press releases, and market analyses. Highlight significant events and analyst perspectives.',
          communicationStyle: 'Thorough, investigative, detail-oriented'
        },
        {
          name: 'Filings Analyst',
          role: 'Review latest 10-Q and 10-K EDGAR filings',
          icon: '📋',
          memberType: 'Team Member',
          personality: 'Expert in analyzing SEC filings and regulatory documents. Says \'The filings reveal...\' and \'According to the 10-K...\'',
          responsibilities: 'Review latest 10-Q and 10-K EDGAR filings. Extract insights from Management Discussion & Analysis, financial statements, and risk factors.',
          communicationStyle: 'Meticulous, regulatory-focused, analytical'
        },
        {
          name: 'Investment Advisor',
          role: 'Deliver comprehensive stock analyses and strategic investment recommendations',
          icon: '💼',
          memberType: 'Decision Maker',
          personality: 'Experienced advisor combining analytical insights. Says \'Based on our comprehensive analysis...\' and \'My recommendation is...\'',
          responsibilities: 'Deliver comprehensive stock analyses and strategic investment recommendations. Synthesize all analyses into unified investment guidance.',
          communicationStyle: 'Authoritative, strategic, actionable'
        }
      ]
    }
  };

  private discussionMessages: { [emailId: number]: DiscussionMessage[] } = {};

  // Progress tracker state
  investmentProgress = {
    currentStep: -1,
    steps: [
      { agent: 'Financial Analyst', icon: '📊', label: 'Financial Analysis', status: 'pending' },
      { agent: 'Research Analyst', icon: '🔍', label: 'Market Research', status: 'pending' },
      { agent: 'Filings Analyst', icon: '📄', label: 'SEC Filings Review', status: 'pending' },
      { agent: 'Investment Advisor', icon: '💼', label: 'Final Recommendation', status: 'pending' }
    ]
  };

  fraudProgress = {
    currentStep: -1,
    steps: [
      { agent: 'Transaction Analyst', icon: '👤', label: 'Transaction Pattern Analysis', status: 'pending' },
      { agent: 'Risk Analyst', icon: '📊', label: 'Fraud Risk Assessment', status: 'pending' },
      { agent: 'Investigation Specialist', icon: '🔍', label: 'Evidence Review', status: 'pending' },
      { agent: 'Fraud Decision Agent', icon: '⚖️', label: 'Final Decision', status: 'pending' }
    ]
  };

  complianceProgress = {
    currentStep: -1,
    steps: [
      { agent: 'Compliance Officer', icon: '📋', label: 'Policy Compliance Check', status: 'pending' },
      { agent: 'Sanctions Analyst', icon: '🔍', label: 'Sanctions Screening', status: 'pending' },
      { agent: 'AML/KYC Specialist', icon: '💰', label: 'AML/KYC Verification', status: 'pending' },
      { agent: 'Regulatory Liaison', icon: '🏛️', label: 'Final Determination', status: 'pending' }
    ]
  };

  constructor() {
    // Configure marked for markdown rendering
    marked.setOptions({
      breaks: true,
      gfm: true
    });

    // Combine selected team and emails observables
    this.teamEmails$ = combineLatest([
      this.store.select(selectAllEmails),
      this.selectedTeam$
    ]).pipe(
      map(([emails, team]) => this.filterEmailsByTeam(emails, team))
    );
  }

  ngOnInit(): void {
    // Load emails from store
    this.store.dispatch(EmailsActions.loadEmails({ limit: 100, offset: 0, append: false }));

    // Connect to SSE for real-time updates
    this.sseService.connect().pipe(
      takeUntil(this.destroy$)
    ).subscribe();

    // Listen for agentic workflow messages
    this.sseService.onEvent('agentic_message').pipe(
      takeUntil(this.destroy$)
    ).subscribe(data => {
      console.log('[SSE] Agentic message:', data);
      this.handleAgenticMessage(data);
    });

    // Listen for workflow completion
    this.sseService.onEvent('agentic_complete').pipe(
      takeUntil(this.destroy$)
    ).subscribe(data => {
      console.log('[SSE] Agentic complete:', data);
      this.handleAgenticComplete(data);
    });

    // Subscribe to route query parameters
    this.route.queryParams.pipe(
      takeUntil(this.destroy$)
    ).subscribe(params => {
      this.selectedTeam = params['team'] || null;
      this.selectedTeam$.next(this.selectedTeam);
      console.log('Selected team:', this.selectedTeam);

      // Check if email_id is provided in query params
      const emailId = params['email_id'];
      if (emailId) {
        // Wait for emails to load, then select the email
        this.store.select(selectAllEmails)
          .pipe(takeUntil(this.destroy$))
          .subscribe(emails => {
            const email = emails.find(e => e.id === parseInt(emailId, 10));
            if (email && this.selectedEmail?.id !== email.id) {
              console.log('[Query Params] Selecting email:', emailId);
              this.selectEmail(email);
            }
          });
      }

      // Show direct interaction and team presentation when team is selected
      this.updateViewState();
    });

    // Subscribe to team changes to load tools dynamically
    this.selectedTeam$.pipe(
      takeUntil(this.destroy$)
    ).subscribe(teamKey => {
      if (teamKey && teamKey !== 'all') {
        console.log(`[selectedTeam$] Team changed to: ${teamKey}, loading tools...`);
        this.loadTeamTools(teamKey);
      }
    });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    this.selectedTeam$.complete();
  }

  private filterEmailsByTeam(emails: Email[], team: string | null): Email[] {
    console.log('Filtering emails:', emails.length, 'Team:', team);

    if (!team || team === 'all') {
      // Show all emails with any team assignment
      const filtered = emails.filter(e => e.assigned_team).slice(0, 10);
      console.log('All teams - showing', filtered.length, 'emails');
      return filtered;
    }

    // Filter by specific team
    const filtered = emails.filter(e => e.assigned_team === team).slice(0, 10);
    console.log(`Team ${team} - showing`, filtered.length, 'emails');
    return filtered;
  }

  getCurrentTeamInfo(): TeamInfo | null {
    if (!this.selectedTeam || this.selectedTeam === 'all') {
      return null;
    }
    return this.teams[this.selectedTeam] || null;
  }

  getTeamName(teamKey: string): string {
    return this.teams[teamKey]?.name || teamKey;
  }

  selectEmail(email: Email): void {
    console.log('[selectEmail] START - Email ID:', email.id, 'Subject:', email.subject);
    console.log('[selectEmail] Email has workflow_results:', email.workflow_results?.length || 0);

    this.selectedEmail = email;

    // Auto-select team from email if not already selected
    if (email.assigned_team && !this.selectedTeam) {
      console.log(`[selectEmail] Auto-selecting team '${email.assigned_team}' from email ${email.id}`);
      this.selectedTeam = email.assigned_team;
      this.selectedTeam$.next(this.selectedTeam);
      // Load tools for the auto-selected team
      this.loadTeamTools(email.assigned_team);
    }

    console.log('[selectEmail] Before updateViewState - showDirectInteraction:', this.showDirectInteraction, 'showTeamPresentation:', this.showTeamPresentation);
    this.updateViewState();
    console.log('[selectEmail] After updateViewState - showDirectInteraction:', this.showDirectInteraction, 'showTeamPresentation:', this.showTeamPresentation);

    // Reset progress trackers for new email
    this.resetProgressTrackers();

    // Load historical messages from workflow_results if available
    if (email && email.workflow_results && email.workflow_results.length > 0) {
      console.log(`[selectEmail] Email ${email.id} has ${email.workflow_results.length} workflow results`);

      // Find the agentic workflow result (not ML results)
      const agenticWorkflow = email.workflow_results.find(wr =>
        wr.workflow_name && wr.workflow_name.startsWith('agentic_')
      );

      if (!agenticWorkflow) {
        console.log(`[selectEmail] No agentic workflow found for email ${email.id}`);
        // Don't return - just skip loading messages
      } else if (!agenticWorkflow.result) {
        console.warn(`[selectEmail] No result field in workflow_result for email ${email.id}`);
        // Don't return - just skip loading messages
      } else if (!agenticWorkflow.result.discussion) {
        console.warn(`[selectEmail] No discussion field in result for email ${email.id}`);
        // Don't return - just skip loading messages
      } else if (!agenticWorkflow.result.discussion.messages) {
        console.warn(`[selectEmail] No messages array in discussion for email ${email.id}`);
        // Don't return - just skip loading messages
      } else {
        // Convert workflow messages to discussion messages format
        console.log('[selectEmail] Converting messages - Count:', agenticWorkflow.result.discussion.messages.length);

        this.discussionMessages[email.id] = agenticWorkflow.result.discussion.messages.map((msg: any) => ({
          agentName: msg.role,
          agentIcon: msg.icon,
          content: msg.text,
          timestamp: msg.timestamp || 'Earlier',
          isToolUsage: msg.is_tool_usage || false,
          isDecision: msg.is_decision || false
        }));

        console.log(`[selectEmail] Successfully loaded ${this.discussionMessages[email.id].length} historical messages for email ${email.id}`);
        console.log('[selectEmail] Sample message:', this.discussionMessages[email.id][0]);

        // Mark all progress steps as completed for completed workflows
        if (email.assigned_team === 'investments') {
          this.investmentProgress.steps.forEach(step => step.status = 'completed');
          this.investmentProgress.currentStep = this.investmentProgress.steps.length - 1;
        } else if (email.assigned_team === 'fraud') {
          this.fraudProgress.steps.forEach(step => step.status = 'completed');
          this.fraudProgress.currentStep = this.fraudProgress.steps.length - 1;
        } else if (email.assigned_team === 'compliance') {
          this.complianceProgress.steps.forEach(step => step.status = 'completed');
          this.complianceProgress.currentStep = this.complianceProgress.steps.length - 1;
        }
      }
    } else {
      console.log(`[selectEmail] Email ${email.id} has no workflow results`);
    }
  }

  private updateViewState(): void {
    // Show direct interaction and team presentation when:
    // 1. A specific team is selected (not 'all')
    // 2. No email is currently selected
    const hasSpecificTeam = !!(this.selectedTeam && this.selectedTeam !== 'all');
    const noEmailSelected = !this.selectedEmail;

    this.showDirectInteraction = hasSpecificTeam && noEmailSelected;
    this.showTeamPresentation = hasSpecificTeam && noEmailSelected;
  }

  formatDate(dateString: string): string {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 60) {
      return `${diffMins} min ago`;
    } else if (diffMins < 1440) {
      return `${Math.floor(diffMins / 60)} hr ago`;
    } else {
      return date.toLocaleDateString();
    }
  }

  getTeamColor(team?: string): string {
    const colors: { [key: string]: string } = {
      'fraud': '#ffebee',
      'compliance': '#f3e5f5',
      'investments': '#e8f5e9',
      'default': '#f5f5f5'
    };
    return colors[team || 'default'] || colors['default'];
  }

  getProgressPercent(email: Email): number {
    if (!email.processed) return 0;
    if (email.workflow_results && email.workflow_results.length > 0) return 100;
    return 50;
  }

  // Markdown parsing
  parseMarkdown(text: string): string {
    try {
      return marked.parse(text) as string;
    } catch (error) {
      console.error('[Markdown] Parse error:', error);
      return text.replace(/\n/g, '<br>');
    }
  }

  // Progress tracker methods
  resetProgressTrackers(): void {
    this.investmentProgress.currentStep = -1;
    this.investmentProgress.steps.forEach(step => step.status = 'pending');

    this.fraudProgress.currentStep = -1;
    this.fraudProgress.steps.forEach(step => step.status = 'pending');

    this.complianceProgress.currentStep = -1;
    this.complianceProgress.steps.forEach(step => step.status = 'pending');
  }

  updateInvestmentProgress(role: string): void {
    const agentIndex = this.investmentProgress.steps.findIndex(step => step.agent === role);
    if (agentIndex === -1) return;

    this.investmentProgress.currentStep = agentIndex;

    // Update step statuses
    this.investmentProgress.steps.forEach((step, index) => {
      if (index < agentIndex) {
        step.status = 'completed';
      } else if (index === agentIndex) {
        step.status = 'active';
      } else {
        step.status = 'pending';
      }
    });
  }

  updateFraudProgress(role: string): void {
    const agentIndex = this.fraudProgress.steps.findIndex(step => step.agent === role);
    if (agentIndex === -1) return;

    this.fraudProgress.currentStep = agentIndex;

    // Update step statuses
    this.fraudProgress.steps.forEach((step, index) => {
      if (index < agentIndex) {
        step.status = 'completed';
      } else if (index === agentIndex) {
        step.status = 'active';
      } else {
        step.status = 'pending';
      }
    });
  }

  getInvestmentProgressPercent(): number {
    if (this.investmentProgress.currentStep === -1) return 0;
    return ((this.investmentProgress.currentStep + 1) / this.investmentProgress.steps.length) * 100;
  }

  getFraudProgressPercent(): number {
    if (this.fraudProgress.currentStep === -1) return 0;
    return ((this.fraudProgress.currentStep + 1) / this.fraudProgress.steps.length) * 100;
  }

  updateComplianceProgress(role: string): void {
    const agentIndex = this.complianceProgress.steps.findIndex(step => step.agent === role);
    if (agentIndex === -1) return;

    this.complianceProgress.currentStep = agentIndex;

    // Update step statuses
    this.complianceProgress.steps.forEach((step, index) => {
      if (index < agentIndex) {
        step.status = 'completed';
      } else if (index === agentIndex) {
        step.status = 'active';
      } else {
        step.status = 'pending';
      }
    });
  }

  getComplianceProgressPercent(): number {
    if (this.complianceProgress.currentStep === -1) return 0;
    return ((this.complianceProgress.currentStep + 1) / this.complianceProgress.steps.length) * 100;
  }

  getEmailDiscussion(emailId: number): DiscussionMessage[] {
    const messages = this.discussionMessages[emailId] || [];
    console.log(`[getEmailDiscussion] Email ${emailId} has ${messages.length} messages`);
    console.log('[getEmailDiscussion] All email IDs with messages:', Object.keys(this.discussionMessages));
    return messages;
  }

  sendChatMessage(): void {
    if (!this.chatMessage.trim() || !this.selectedEmail) return;

    // Check if email has an agentic task
    if (!this.selectedEmail.agentic_task_id) {
      console.error('[Chat] No agentic task found for this email');
      return;
    }

    const message = this.chatMessage.trim();
    const taskId = this.selectedEmail.agentic_task_id;
    const emailId = this.selectedEmail.id;

    // Add user message to discussion
    const userMessage: DiscussionMessage = {
      agentName: 'You',
      agentIcon: '👤',
      content: message,
      timestamp: 'Just now'
    };

    if (!this.discussionMessages[emailId]) {
      this.discussionMessages[emailId] = [];
    }

    this.discussionMessages[emailId].push(userMessage);

    // Clear input immediately
    this.chatMessage = '';

    // Show loading message
    const loadingMessage: DiscussionMessage = {
      agentName: 'AI Coordinator',
      agentIcon: '🤖',
      content: 'Thank you for your question. The team is analyzing this...',
      timestamp: 'Just now'
    };
    this.discussionMessages[emailId].push(loadingMessage);

    // Send to backend
    this.apiService.sendChatMessage(taskId, message)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          console.log('[Chat] Response received:', response);

          // Remove loading message
          const messages = this.discussionMessages[emailId];
          const loadingIndex = messages.indexOf(loadingMessage);
          if (loadingIndex > -1) {
            messages.splice(loadingIndex, 1);
          }

          // Add actual response
          if (response.response) {
            const aiResponse: DiscussionMessage = {
              agentName: response.agent || 'AI Coordinator',
              agentIcon: response.icon || this.getAgentIcon(response.agent || 'AI Coordinator'),
              content: response.response,
              timestamp: 'Just now'
            };
            this.discussionMessages[emailId].push(aiResponse);
          }

          this.cdr.detectChanges();
        },
        error: (error) => {
          console.error('[Chat] Error sending message:', error);

          // Remove loading message
          const messages = this.discussionMessages[emailId];
          const loadingIndex = messages.indexOf(loadingMessage);
          if (loadingIndex > -1) {
            messages.splice(loadingIndex, 1);
          }

          // Show error message
          const errorMessage: DiscussionMessage = {
            agentName: 'System',
            agentIcon: '⚠️',
            content: 'Failed to send message. Please try again.',
            timestamp: 'Just now'
          };
          this.discussionMessages[emailId].push(errorMessage);

          this.cdr.detectChanges();
        }
      });
  }

  handleChatKeypress(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.sendChatMessage();
    }
  }

  submitDirectQuery(): void {
    const query = this.directQuery.trim();

    console.log('=====================================');
    console.log('[submitDirectQuery] *** FUNCTION CALLED ***');
    console.log('[submitDirectQuery] START - Query:', query, 'Team:', this.selectedTeam);
    console.log('[submitDirectQuery] Current showWorkflowModal:', this.showWorkflowModal);
    console.log('=====================================');

    if (!query) {
      console.log('[submitDirectQuery] Empty query, aborting');
      return;
    }

    if (!this.selectedTeam || this.selectedTeam === 'all') {
      console.log('[submitDirectQuery] No team selected, aborting');
      return;
    }

    // Show loading state
    this.isSubmittingQuery = true;
    console.log('[submitDirectQuery] Submitting query...');

    // Call API to create direct query task
    this.emailService.submitDirectQuery(this.selectedTeam, query)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (result) => {
          console.log('[Direct Query] Task created:', result.task_id, 'Email:', result.email_id);

          // Clear query input
          this.directQuery = '';
          this.isSubmittingQuery = false;

          // FIX: Directly fetch the email by ID instead of waiting for store
          // This is more reliable than waiting for pagination/filtering
          this.emailService.getEmailById(result.email_id)
            .pipe(takeUntil(this.destroy$))
            .subscribe({
              next: (email) => {
                console.log('[Direct Query] Fetched email directly:', email.id);
                console.log('[Direct Query] Selecting email:', email.id);

                // Select the email
                this.selectEmail(email);

                // Force view update - hide direct interaction, show email details
                this.showDirectInteraction = false;
                this.showTeamPresentation = false;

                // Open the workflow modal AFTER email is selected
                this.showWorkflowModal = true;

                // Manually trigger change detection to ensure modal displays
                this.cdr.detectChanges();

                console.log('[Direct Query] View state updated');
                console.log('[Direct Query] Modal opened - showWorkflowModal:', this.showWorkflowModal);
                console.log('[Direct Query] Selected email team:', email.assigned_team);
                console.log('[Direct Query] Selected email ID:', this.selectedEmail?.id);

                // Also reload emails in background to update the list
                this.store.dispatch(EmailsActions.loadEmails({ limit: 100, offset: 0, append: false }));
              },
              error: (error) => {
                console.error('[Direct Query] Error fetching email:', error);
              }
            });
        },
        error: (error) => {
          console.error('[Direct Query] Error:', error);
          this.isSubmittingQuery = false;
        }
      });
  }

  clearDirectQuery(): void {
    this.directQuery = '';
  }

  closeWorkflowModal(): void {
    this.showWorkflowModal = false;
  }

  enterEditMode(): void {
    if (!this.selectedTeam || this.selectedTeam === 'all') {
      return;
    }

    const teamInfo = this.getCurrentTeamInfo();
    if (!teamInfo) {
      return;
    }

    // Switch to edit mode
    this.isEditMode = true;
    this.showTeamPresentation = false;
  }

  exitEditMode(): void {
    // Switch back to presentation mode
    this.isEditMode = false;
    this.showTeamPresentation = true;
  }

  // Tool management methods
  toggleToolExpansion(index: number): void {
    if (this.expandedToolIndex === index) {
      this.expandedToolIndex = null;
    } else {
      this.expandedToolIndex = index;
      this.editingToolIndex = null; // Close editing when expanding another
    }
  }

  isToolExpanded(index: number): boolean {
    return this.expandedToolIndex === index;
  }

  toggleToolEdit(index: number): void {
    if (this.editingToolIndex === index) {
      this.editingToolIndex = null;
    } else {
      this.editingToolIndex = index;
    }
  }

  isToolEditing(index: number): boolean {
    return this.editingToolIndex === index;
  }

  getToolTypeBadgeClass(type: string): string {
    const badges = {
      'mcp': 'tool-type-mcp',
      'proprietary': 'tool-type-proprietary',
      'public': 'tool-type-public',
      'api': 'tool-type-api'
    };
    return badges[type as keyof typeof badges] || 'tool-type-default';
  }

  getToolTypeIcon(type: string): string {
    const icons = {
      'mcp': 'extension',
      'proprietary': 'lock',
      'public': 'public',
      'api': 'api'
    };
    return icons[type as keyof typeof icons] || 'settings';
  }

  getConfigKeys(config: { [key: string]: string | number | boolean }): string[] {
    return Object.keys(config);
  }

  getConfigValue(value: string | number | boolean): string {
    if (typeof value === 'boolean') {
      return value ? 'enabled' : 'disabled';
    }
    return String(value);
  }

  async testTool(tool: Tool, index: number): Promise<void> {
    console.log(`[Tool Test] Testing tool: ${tool.name} (${tool.type})`);

    this.testingToolIndex = index;
    const startTime = Date.now();

    try {
      // Call backend API to actually test the tool
      const response = await fetch('http://localhost:8000/api/tools/test', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          tool_name: tool.name,
          tool_type: tool.type,
          configuration: tool.configuration,
          provider: tool.provider
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      const responseTime = Date.now() - startTime;

      // Check if this is a warning (e.g., configured but needs auth, not deployed)
      const isWarning = result.data?.is_warning || result.data?.status === 'configured' || result.data?.status === 'authentication_required' || result.data?.status === 'not_deployed';

      this.toolTestResults[index] = {
        success: true,
        isWarning: isWarning,
        data: result.data || result,
        message: result.data?.message,
        warning: result.data?.warning,
        timestamp: new Date().toLocaleTimeString(),
        responseTime
      };

      console.log(`[Tool Test] ${isWarning ? 'Warning' : 'Success'} for ${tool.name}:`, result);
    } catch (error: any) {
      const responseTime = Date.now() - startTime;
      this.toolTestResults[index] = {
        success: false,
        error: error.message || 'Tool test failed',
        timestamp: new Date().toLocaleTimeString(),
        responseTime
      };
      console.error(`[Tool Test] Error for ${tool.name}:`, error);
    } finally {
      this.testingToolIndex = null;
    }
  }

  isToolTesting(index: number): boolean {
    return this.testingToolIndex === index;
  }

  getToolTestResult(index: number): ToolTestResult | null {
    return this.toolTestResults[index] || null;
  }

  async loadTeamTools(teamKey: string): Promise<void> {
    if (!teamKey || teamKey === 'all') {
      return;
    }

    console.log(`[loadTeamTools] Loading tools for team: ${teamKey}`);

    try {
      const response = await fetch(`http://localhost:8000/api/teams/${teamKey}/tools`);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();

      // Update the team's tools with the loaded data
      if (this.teams[teamKey]) {
        this.teams[teamKey].tools = result.tools || [];
        console.log(`[loadTeamTools] Loaded ${result.tools?.length || 0} tools for team ${teamKey}:`, result.tools);
      }
    } catch (error: any) {
      console.error(`[loadTeamTools] Error loading tools for team ${teamKey}:`, error);
      // Keep empty array on error
      if (this.teams[teamKey]) {
        this.teams[teamKey].tools = [];
      }
    }
  }

  formatConfigKey(key: string): string {
    return key.split('_').map(word =>
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  }

  getEnvVariableForTool(tool: Tool): string {
    // Map tool providers/names to their environment variable names
    const provider = tool.provider || '';
    const toolName = tool.name || '';

    if (provider.includes('Serper') || toolName.includes('Serper')) {
      return 'SERPER_API_KEY';
    } else if (provider.includes('Browserless') || toolName.includes('Browserless')) {
      return 'BROWSERLESS_API_KEY';
    } else if (provider.includes('SEC-API') || toolName.includes('SEC')) {
      return 'SEC_API_KEY';
    } else if (provider.includes('IPGeolocation') || toolName.includes('IP Geolocation')) {
      return 'IPGEOLOCATION_API_KEY';
    } else if (provider.includes('AbstractAPI') || toolName.includes('Email Validation')) {
      return 'ABSTRACTAPI_EMAIL_KEY';
    } else if (tool.type === 'mcp') {
      return 'MCP_ENABLED';
    } else {
      // Generic fallback - try to extract from configuration
      const apiKeyField = tool.configuration?.['api_key'];
      if (typeof apiKeyField === 'string' && apiKeyField.includes('not_configured')) {
        // Try to infer from provider name
        return `${provider.toUpperCase().replace(/[^A-Z0-9]/g, '_')}_API_KEY`;
      }
      return 'API_KEY';
    }
  }

  getStatusText(email: Email): string {
    if (!email.processed) return 'Pending';
    if (email.workflow_results && email.workflow_results.length > 0) return 'Completed';
    return 'Processing';
  }

  getStatusClass(email: Email): string {
    if (!email.processed) return 'status-pending';
    if (email.workflow_results && email.workflow_results.length > 0) return 'status-completed';
    return 'status-processing';
  }

  getQueryHint(): string {
    if (this.selectedTeam === 'investments') {
      return 'For Investment Team: Request stock analysis by company name or ticker symbol (e.g., "Analyze Apple stock" or "Complete analysis for TSLA")';
    } else if (this.selectedTeam === 'fraud') {
      return 'For Fraud Team: Report suspicious activities (e.g., "Investigate unusual transaction pattern")';
    } else if (this.selectedTeam === 'compliance') {
      return 'For Compliance Team: Submit regulatory queries (e.g., "Review FATCA compliance requirements")';
    }
    return 'Submit your request to the team';
  }

  private handleAgenticMessage(data: any): void {
    const emailId = data.email_id;
    const agentName = data.agent_name || 'Agent';
    const message = data.message || '';

    if (!emailId) {
      console.warn('[SSE] Received message without email_id:', data);
      return;
    }

    console.log('[SSE] Received message from agent:', agentName, 'for email:', emailId, 'Message:', message.substring(0, 100));

    // Initialize messages array if needed
    if (!this.discussionMessages[emailId]) {
      this.discussionMessages[emailId] = [];
      console.log('[SSE] Initialized messages array for email:', emailId);
    }

    // Create and add the message
    const newMessage: DiscussionMessage = {
      agentName: agentName,
      agentIcon: this.getAgentIcon(agentName),
      content: message,
      timestamp: 'Just now',
      isToolUsage: data.tool_usage || false,
      isDecision: data.is_decision || false
    };

    this.discussionMessages[emailId].push(newMessage);
    console.log('[SSE] Added message to email:', emailId, 'Total messages:', this.discussionMessages[emailId].length);

    // Only update progress if this message is for the currently selected email
    if (!this.selectedEmail || this.selectedEmail.id !== emailId) {
      console.log('[SSE] Message is not for currently selected email (selected:', this.selectedEmail?.id, 'vs message:', emailId, ')');
      // Message stored for when/if this email is selected
      return;
    }

    console.log('[Progress] Updating progress for currently selected email:', emailId);

    // Update progress trackers based on team and agent
    if (this.selectedEmail.assigned_team === 'investments') {
      console.log('[Progress] Updating investment progress for:', agentName);
      this.updateInvestmentProgress(agentName);
      console.log('[Progress] Current step:', this.investmentProgress.currentStep, 'Steps:', this.investmentProgress.steps.map(s => s.agent + ':' + s.status));
    } else if (this.selectedEmail.assigned_team === 'fraud') {
      console.log('[Progress] Updating fraud progress for:', agentName);
      this.updateFraudProgress(agentName);
      console.log('[Progress] Current step:', this.fraudProgress.currentStep, 'Steps:', this.fraudProgress.steps.map(s => s.agent + ':' + s.status));
    } else if (this.selectedEmail?.assigned_team === 'compliance') {
      console.log('[Progress] Updating compliance progress for:', agentName);
      this.updateComplianceProgress(agentName);
      console.log('[Progress] Current step:', this.complianceProgress.currentStep, 'Steps:', this.complianceProgress.steps.map(s => s.agent + ':' + s.status));
    }

    // Manually trigger change detection to update the UI
    this.cdr.detectChanges();
    console.log('[SSE] Triggered change detection after message update');
  }

  private handleAgenticComplete(data: any): void {
    const emailId = data.email_id;
    console.log('[Workflow Complete] Email:', emailId);

    // Mark all steps as completed when workflow finishes
    if (this.selectedEmail?.assigned_team === 'investments') {
      this.investmentProgress.steps.forEach(step => step.status = 'completed');
      this.investmentProgress.currentStep = this.investmentProgress.steps.length - 1;
      console.log('[Workflow Complete] Investment analysis completed');
    } else if (this.selectedEmail?.assigned_team === 'fraud') {
      this.fraudProgress.steps.forEach(step => step.status = 'completed');
      this.fraudProgress.currentStep = this.fraudProgress.steps.length - 1;
      console.log('[Workflow Complete] Fraud investigation completed');
    } else if (this.selectedEmail?.assigned_team === 'compliance') {
      this.complianceProgress.steps.forEach(step => step.status = 'completed');
      this.complianceProgress.currentStep = this.complianceProgress.steps.length - 1;
      console.log('[Workflow Complete] Compliance review completed');
    }

    // Manually trigger change detection
    this.cdr.detectChanges();

    // Reload emails to get updated status
    this.store.dispatch(EmailsActions.loadEmails({ limit: 100, offset: 0, append: false }));
  }

  private getAgentIcon(agentName: string): string {
    const iconMap: { [key: string]: string } = {
      'Financial Analyst': '📊',
      'Research Analyst': '🔍',
      'Filings Analyst': '📋',
      'Investment Advisor': '💼',
      'Fraud Detection Specialist': '🔍',
      'Forensic Analyst': '🧪',
      'Legal Advisor': '⚖️',
      'Security Director': '🛡️',
      'Compliance Officer': '📋',
      'Legal Counsel': '⚖️',
      'Auditor': '📊',
      'Regulatory Liaison': '🏛️'
    };
    return iconMap[agentName] || '🤖';
  }

  // Tools Editor Methods
  async toggleToolsEditor() {
    this.isEditingTools = !this.isEditingTools;

    if (this.isEditingTools) {
      // Load all tools from registry when opening
      this.loadingRegistry = true;
      try {
        const response = await fetch('http://localhost:8000/api/tools/registry');
        const data = await response.json();
        // Filter out example/demo tools
        this.registryTools = (data.tools || []).filter((tool: any) =>
          !tool.name.toLowerCase().includes('example') &&
          !tool.name.toLowerCase().includes('demo')
        );
      } catch (error) {
        console.error('Error loading tools registry:', error);
        this.registryTools = [];
      } finally {
        this.loadingRegistry = false;
      }
    }
  }

  getAssignedToolNames(): string[] {
    const teamInfo = this.getCurrentTeamInfo();
    return teamInfo?.tools.map(t => t.name) || [];
  }

  getAvailableToolNames(): string[] {
    const assignedNames = this.getAssignedToolNames();
    return this.registryTools
      .filter(tool => !assignedNames.includes(tool.name))
      .map(tool => tool.name);
  }

  isToolAssigned(toolName: string): boolean {
    const assignedNames = this.getAssignedToolNames();

    // Direct name match
    if (assignedNames.includes(toolName)) {
      return true;
    }

    // Check if a similar tool is already assigned (by provider)
    const registryTool = this.registryTools.find(t => t.name === toolName);
    if (registryTool && registryTool.provider) {
      const teamInfo = this.getCurrentTeamInfo();
      if (teamInfo) {
        // Check if any assigned tool has the same provider
        return teamInfo.tools.some(tool =>
          tool.provider && tool.provider.toLowerCase() === registryTool.provider.toLowerCase()
        );
      }
    }

    return false;
  }

  async addToolToTeam(toolName: string) {
    if (!this.selectedTeam) return;

    try {
      // Add tool to team assignment
      const currentTools = this.getAssignedToolNames();
      const newTools = [...currentTools, toolName];

      const response = await fetch('http://localhost:8000/api/tools/registry/assign', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          team_key: this.selectedTeam,
          tool_names: newTools
        })
      });

      if (response.ok) {
        // Reload team tools
        this.loadTeamTools(this.selectedTeam);
        console.log(`Added ${toolName} to ${this.selectedTeam} team`);
      } else {
        console.error('Failed to add tool to team');
      }
    } catch (error) {
      console.error('Error adding tool to team:', error);
    }
  }

  async removeToolFromTeam(toolName: string) {
    if (!this.selectedTeam) return;

    try {
      // Remove tool from team assignment
      const currentTools = this.getAssignedToolNames();
      const newTools = currentTools.filter(t => t !== toolName);

      const response = await fetch('http://localhost:8000/api/tools/registry/assign', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          team_key: this.selectedTeam,
          tool_names: newTools
        })
      });

      if (response.ok) {
        // Reload team tools
        this.loadTeamTools(this.selectedTeam);
        console.log(`Removed ${toolName} from ${this.selectedTeam} team`);
      } else {
        console.error('Failed to remove tool from team');
      }
    } catch (error) {
      console.error('Error removing tool from team:', error);
    }
  }
}
