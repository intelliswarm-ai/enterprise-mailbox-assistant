import { Component, Input, Output, EventEmitter, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Email, TeamAssignment, TaskOption } from '../../../../core/models/email.model';
import { Router } from '@angular/router';
import { Store } from '@ngrx/store';
import { SseService } from '../../../../core/services/sse.service';
import { EmailService } from '../../../../core/services/email.service';
import * as EmailsActions from '../../../../store/emails/emails.actions';
import { Subject, takeUntil } from 'rxjs';

interface ProgressStep {
  icon: string;
  label: string;
  status: 'pending' | 'active' | 'completed';
  agent: string;
}

interface AnalysisMessage {
  icon: string;
  agentName: string;
  timestamp: string;
  content: string;
}

@Component({
  selector: 'app-email-detail-modal',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './email-detail-modal.component.html',
  styleUrl: './email-detail-modal.component.scss'
})
export class EmailDetailModalComponent implements OnInit, OnDestroy {
  @Input() email: Email | null = null;
  @Input() isOpen = false;
  @Output() closeModal = new EventEmitter<void>();
  @Output() processEmail = new EventEmitter<number>();
  @Output() assignToTeam = new EventEmitter<TeamAssignment>();

  private destroy$ = new Subject<void>();

  // Task selection state
  showTaskOptions = false;
  loadingTaskOptions = false;
  taskOptions: TaskOption[] = [];
  selectedTeam: string | null = null;

  // Unified progress tracking for all workflows
  showWorkflowProgress = false;
  currentWorkflowType: 'fraud' | 'investment' | 'compliance' | null = null;

  // Fraud progress tracking
  fraudProgressSteps: ProgressStep[] = [
    { icon: '🔍', label: 'Fraud Type Detection', status: 'pending', agent: 'Fraud Investigation Unit' },
    { icon: '🎣', label: 'Deep Investigation', status: 'pending', agent: 'Phishing Analysis' },
    { icon: '💾', label: 'Historical Analysis', status: 'pending', agent: 'Database Investigation' },
    { icon: '⚖️', label: 'Risk Assessment', status: 'pending', agent: 'Final Decision' }
  ];
  fraudProgressPercent = 0;
  fraudMessages: AnalysisMessage[] = [];

  // Investment progress tracking
  investmentProgressSteps: ProgressStep[] = [
    { icon: '📊', label: 'Financial Analysis', status: 'pending', agent: 'Financial Analyst' },
    { icon: '🔍', label: 'Market Research', status: 'pending', agent: 'Research Analyst' },
    { icon: '📋', label: 'SEC Filings Review', status: 'pending', agent: 'Filings Analyst' },
    { icon: '💼', label: 'Investment Recommendation', status: 'pending', agent: 'Investment Advisor' }
  ];
  investmentProgressPercent = 0;
  investmentMessages: AnalysisMessage[] = [];

  // Compliance progress tracking
  complianceProgressSteps: ProgressStep[] = [
    { icon: '📋', label: 'Policy Review', status: 'pending', agent: 'Compliance Officer' },
    { icon: '⚖️', label: 'Legal Assessment', status: 'pending', agent: 'Legal Counsel' },
    { icon: '🔍', label: 'AML/KYC Screening', status: 'pending', agent: 'Auditor' },
    { icon: '🏛️', label: 'Final Determination', status: 'pending', agent: 'Regulatory Liaison' }
  ];
  complianceProgressPercent = 0;
  complianceMessages: AnalysisMessage[] = [];

  constructor(
    private router: Router,
    private store: Store,
    private sseService: SseService,
    private emailService: EmailService
  ) {}

  ngOnInit(): void {
    // Ensure SSE is connected (idempotent - won't reconnect if already connected)
    if (!this.sseService.connected) {
      this.sseService.connect().pipe(
        takeUntil(this.destroy$)
      ).subscribe();
    }

    // Subscribe to SSE events for real-time updates
    this.subscribeToSSE();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  private subscribeToSSE(): void {
    // Listen for agentic progress events
    this.sseService.onEvent('agentic_progress').pipe(
      takeUntil(this.destroy$)
    ).subscribe((data: any) => {
      console.log('[EmailDetailModal] Received agentic_progress:', data);
      if (this.email && data.email_id === this.email.id) {
        this.updateWorkflowProgress(data);
      }
    });

    // Listen for agentic messages
    this.sseService.onEvent('agentic_message').pipe(
      takeUntil(this.destroy$)
    ).subscribe((data: any) => {
      console.log('[EmailDetailModal] Received agentic_message:', data);
      if (this.email && data.email_id === this.email.id) {
        this.addWorkflowMessage(data);
      }
    });

    // Listen for workflow completion
    this.sseService.onEvent('agentic_complete').pipe(
      takeUntil(this.destroy$)
    ).subscribe((data: any) => {
      console.log('[EmailDetailModal] Received agentic_complete:', data);
      if (this.email && data.email_id === this.email.id) {
        console.log('[EmailDetailModal] Workflow completed for current email, reloading email data');

        // Show completion message
        if (this.showWorkflowProgress) {
          this.addWorkflowMessage({
            agent: 'System',
            role: 'System',
            team: this.currentWorkflowType,
            message: '✅ Workflow completed successfully! Analysis has been saved and is now available in your past analyses.',
            text: '✅ Workflow completed successfully! Analysis has been saved and is now available in your past analyses.'
          });
        }

        // Reload the specific email to get updated assigned_team and workflow results
        this.reloadCurrentEmail();
      }
    });
  }

  private updateWorkflowProgress(data: any): void {
    const { agent, status, step, team } = data;

    // Determine which workflow to update
    let progressSteps: ProgressStep[];
    if (team === 'fraud') {
      progressSteps = this.fraudProgressSteps;
    } else if (team === 'investments') {
      progressSteps = this.investmentProgressSteps;
    } else if (team === 'compliance') {
      progressSteps = this.complianceProgressSteps;
    } else {
      return;
    }

    // Update step status based on agent
    progressSteps.forEach((progressStep, index) => {
      if (progressStep.agent === agent || index === step) {
        progressStep.status = status;
      }
    });

    // Calculate progress percentage
    const completedSteps = progressSteps.filter(s => s.status === 'completed').length;
    const totalSteps = progressSteps.length;
    const progressPercent = (completedSteps / totalSteps) * 100;

    // Update the appropriate progress percentage
    if (team === 'fraud') {
      this.fraudProgressPercent = progressPercent;
    } else if (team === 'investments') {
      this.investmentProgressPercent = progressPercent;
    } else if (team === 'compliance') {
      this.complianceProgressPercent = progressPercent;
    }
  }

  private addWorkflowMessage(data: any): void {
    const message: AnalysisMessage = {
      icon: this.getAgentIcon(data.agent || data.role),
      agentName: data.agent || data.role || 'System',
      timestamp: new Date().toLocaleTimeString(),
      content: data.message || data.text || ''
    };

    // Add to appropriate message array based on team
    const team = data.team || this.currentWorkflowType;
    if (team === 'fraud') {
      this.fraudMessages.push(message);
    } else if (team === 'investments') {
      this.investmentMessages.push(message);
    } else if (team === 'compliance') {
      this.complianceMessages.push(message);
    }
  }

  private reloadCurrentEmail(): void {
    if (!this.email) return;

    console.log('[EmailDetailModal] Reloading email', this.email.id, 'to get updated workflow results');

    // Fetch the updated email from the backend
    this.emailService.getEmailById(this.email.id).pipe(
      takeUntil(this.destroy$)
    ).subscribe({
      next: (updatedEmail) => {
        console.log('[EmailDetailModal] Email reloaded successfully:', updatedEmail);
        console.log('[EmailDetailModal] Assigned team:', updatedEmail.assigned_team);
        console.log('[EmailDetailModal] Workflow results:', updatedEmail.workflow_results?.length || 0);

        // Update local email reference
        this.email = updatedEmail;

        // Dispatch action to update the email in the store
        // This will refresh the email in the mailbox list and agentic teams sidebar
        this.store.dispatch(EmailsActions.loadEmails({
          limit: 100,
          offset: 0,
          append: false
        }));

        console.log('[EmailDetailModal] Dispatched loadEmails action to refresh store');
      },
      error: (error) => {
        console.error('[EmailDetailModal] Error reloading email:', error);
      }
    });
  }

  private getAgentIcon(agentName: string): string {
    const icons: { [key: string]: string } = {
      'Database Investigation Agent': '💾',
      'Phishing Analysis Specialist': '🔨',
      'Fraud Investigation Unit': '🔍',
      'Risk Assessment Agent': '⚖️'
    };
    return icons[agentName] || '🤖';
  }

  close(): void {
    this.closeModal.emit();
  }

  getStatusBadgeClass(): string {
    if (!this.email) return '';
    if (!this.email.processed) return 'bg-gradient-warning';
    return this.email.is_phishing ? 'bg-gradient-danger' : 'bg-gradient-success';
  }

  getStatusText(): string {
    if (!this.email) return '';
    if (!this.email.processed) return 'Pending Analysis';
    return this.email.is_phishing ? 'Phishing' : 'Legitimate';
  }

  formatDate(dateString: string): string {
    return new Date(dateString).toLocaleString();
  }

  getBadgeIcon(badgeType: string): string {
    const icons: { [key: string]: string } = {
      'MEETING': 'event',
      'RISK': 'shield',
      'EXTERNAL': 'close',
      'AUTOMATED': 'settings',
      'VIP': 'star',
      'FOLLOW_UP': 'refresh',
      'NEWSLETTER': 'mail',
      'FINANCE': 'attach_money'
    };
    return icons[badgeType] || 'label';
  }

  getTeamDisplayName(teamKey: string): string {
    const teamNames: { [key: string]: string } = {
      'fraud': 'Fraud Investigation',
      'compliance': 'Compliance',
      'investments': 'Investment Team'
    };
    return teamNames[teamKey] || teamKey;
  }

  runAnalysis(): void {
    if (this.email) {
      this.processEmail.emit(this.email.id);
    }
  }

  viewTeamDiscussion(): void {
    if (this.email?.agentic_task_id) {
      // Navigate to agentic-teams page with task_id and email_id
      this.router.navigate(['/agentic-teams'], {
        queryParams: {
          email_id: this.email.id,
          task_id: this.email.agentic_task_id
        }
      });
      this.close();
    }
  }

  assignTeam(team: string): void {
    if (!this.email) return;

    console.log(`[EmailDetailModal] Loading task options for team: ${team}`);
    this.selectedTeam = team;
    this.loadingTaskOptions = true;
    this.showTaskOptions = false;
    this.taskOptions = [];

    // Call backend to analyze email and get task options
    this.emailService.analyzeEmailTasks(this.email.id, team).pipe(
      takeUntil(this.destroy$)
    ).subscribe({
      next: (response) => {
        console.log(`[EmailDetailModal] Received ${response.task_options.length} task options`);
        this.taskOptions = response.task_options;
        this.loadingTaskOptions = false;
        this.showTaskOptions = true;
      },
      error: (error) => {
        console.error('[EmailDetailModal] Error loading task options:', error);
        this.loadingTaskOptions = false;
        // Fallback: assign without task selection
        this.confirmTeamAssignment(team, undefined);
      }
    });
  }

  selectTask(task: TaskOption): void {
    if (!this.email || !this.selectedTeam) return;

    console.log(`[EmailDetailModal] Task selected:`, task);
    this.confirmTeamAssignment(this.selectedTeam, task);
  }

  cancelTaskSelection(): void {
    this.showTaskOptions = false;
    this.taskOptions = [];
    this.selectedTeam = null;
  }

  private confirmTeamAssignment(team: string, selectedTask?: TaskOption): void {
    if (!this.email) return;

    console.log(`[EmailDetailModal] Assigning email ${this.email.id} to team: ${team}`, selectedTask);

    // Show progress tracker for the selected workflow
    this.showWorkflowProgress = true;
    this.currentWorkflowType = team as 'fraud' | 'investment' | 'compliance';

    // Reset progress based on team type
    if (team === 'fraud') {
      console.log('[EmailDetailModal] Showing fraud progress tracker');
      this.fraudMessages = [];
      this.fraudProgressSteps.forEach(step => step.status = 'pending');
      this.fraudProgressPercent = 0;
    } else if (team === 'investments') {
      console.log('[EmailDetailModal] Showing investment progress tracker');
      this.investmentMessages = [];
      this.investmentProgressSteps.forEach(step => step.status = 'pending');
      this.investmentProgressPercent = 0;
    } else if (team === 'compliance') {
      console.log('[EmailDetailModal] Showing compliance progress tracker');
      this.complianceMessages = [];
      this.complianceProgressSteps.forEach(step => step.status = 'pending');
      this.complianceProgressPercent = 0;
    }

    // Hide task options
    this.showTaskOptions = false;

    // Assign to team with selected task
    this.assignToTeam.emit({
      emailId: this.email.id,
      team,
      message: undefined,
      selectedTask
    });

    // Don't close modal - let user watch the analysis for all workflows
    console.log('[EmailDetailModal] Modal will stay open to show progress');
  }

  getPriorityBadgeClass(priority: string): string {
    const classes: { [key: string]: string } = {
      'high': 'bg-danger',
      'medium': 'bg-warning',
      'low': 'bg-info'
    };
    return classes[priority] || 'bg-secondary';
  }
}
