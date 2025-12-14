import { Component, OnInit, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';
import { environment } from '../../../environments/environment';
import { EmailDetailModalComponent } from '../mailbox/components/email-detail-modal/email-detail-modal.component';
import { Email } from '../../core/models/email.model';

type BadgeType = 'MEETING' | 'RISK' | 'EXTERNAL' | 'AUTOMATED' | 'VIP' | 'FOLLOW_UP' | 'NEWSLETTER' | 'FINANCE';

interface EmailSummary {
  id: number;
  subject: string;
  sender: string;
  recipient?: string;
  received_at: string;
  summary: string[];
  body_text?: string;
  has_cta: boolean;
  cta_text?: string;
  cta_type?: string;
  badges: BadgeType[];
  processed?: boolean;
  is_phishing?: boolean;
}

interface GroupedEmails {
  [category: string]: EmailSummary[];
}

interface DigestStats {
  total_today: number;
  badge_counts: { [key in BadgeType]?: number };
}

interface CallToAction {
  email_id: number;
  subject: string;
  cta_text: string;
  cta_type: string;
  category: BadgeType;
}

@Component({
  selector: 'app-daily-inbox-digest',
  standalone: true,
  imports: [CommonModule, FormsModule, EmailDetailModalComponent],
  templateUrl: './daily-inbox-digest.component.html',
  styleUrl: './daily-inbox-digest.component.scss'
})
export class DailyInboxDigestComponent implements OnInit {
  private http = inject(HttpClient);
  private router = inject(Router);

  groupedEmails: GroupedEmails = {};
  otherEmails: EmailSummary[] = []; // Store OTHER category emails for CTA lookup
  availableCategories: BadgeType[] = [];
  selectedCategories: Set<BadgeType> = new Set();

  stats: DigestStats = {
    total_today: 0,
    badge_counts: {}
  };

  allCallToActions: CallToAction[] = [];

  selectedEmail: Email | null = null;
  showEmailModal: boolean = false;
  isLoading: boolean = false;

  badgeIcons: { [key in BadgeType]: string } = {
    'MEETING': 'event',
    'RISK': 'shield',
    'EXTERNAL': 'close',
    'AUTOMATED': 'settings',
    'VIP': 'star',
    'FOLLOW_UP': 'cached',
    'NEWSLETTER': 'email',
    'FINANCE': 'attach_money'
  };

  badgeClasses: { [key in BadgeType]: string } = {
    'MEETING': 'badge-meeting',
    'RISK': 'badge-risk',
    'EXTERNAL': 'badge-external',
    'AUTOMATED': 'badge-automated',
    'VIP': 'badge-vip',
    'FOLLOW_UP': 'badge-follow-up',
    'NEWSLETTER': 'badge-newsletter',
    'FINANCE': 'badge-finance'
  };

  ngOnInit(): void {
    this.loadDigestData();
  }

  loadDigestData(): void {
    this.isLoading = true;
    const apiUrl = `${environment.apiUrl}/inbox-digest?hours=24`;

    this.http.get<any>(apiUrl).subscribe({
      next: (response) => {
        // Map the API response to our component format
        this.groupedEmails = {};

        // Initialize all categories
        const allCategories: BadgeType[] = ['MEETING', 'RISK', 'EXTERNAL', 'AUTOMATED', 'VIP', 'FOLLOW_UP', 'NEWSLETTER', 'FINANCE'];
        allCategories.forEach(cat => {
          this.groupedEmails[cat] = [];
        });

        // Reset CTAs and other emails before extracting
        this.allCallToActions = [];
        this.otherEmails = [];

        // Fill with API data - including OTHER category for CTA extraction
        Object.keys(response.grouped_emails || {}).forEach(category => {
          const emails = (response.grouped_emails[category] || []).map((email: any) => {
            // Handle call_to_actions - can be array of strings or array of objects
            const ctas = email.call_to_actions || [];
            let ctaText = '';
            let ctaType = '';

            if (ctas.length > 0) {
              if (typeof ctas[0] === 'string') {
                ctaText = ctas[0];
                ctaType = 'action';
              } else {
                ctaText = ctas[0]?.action || ctas[0]?.text || '';
                ctaType = ctas[0]?.type || 'action';
              }
            }

            return {
              id: email.id,
              subject: email.subject,
              sender: email.sender,
              recipient: email.recipient,
              received_at: email.received_at,
              summary: Array.isArray(email.summary) ? email.summary : (email.summary ? [email.summary] : []),
              body_text: email.body_text,
              has_cta: ctas.length > 0,
              cta_text: ctaText,
              cta_type: ctaType,
              badges: email.badges || [],
              processed: true,
              is_phishing: email.is_phishing
            };
          });

          // Store in appropriate category
          if (category !== 'OTHER') {
            this.groupedEmails[category as BadgeType] = emails;
          } else {
            // Store OTHER category emails for CTA lookup
            this.otherEmails = emails;
          }

          // Extract CTAs from ALL categories (including OTHER)
          emails.forEach((email: EmailSummary) => {
            if (email.has_cta && email.cta_text) {
              this.allCallToActions.push({
                email_id: email.id,
                subject: email.subject,
                cta_text: email.cta_text,
                cta_type: email.cta_type || 'action',
                category: (category !== 'OTHER' ? category : 'AUTOMATED') as BadgeType
              });
            }
          });
        });

        // Update stats
        this.stats = {
          total_today: response.total_today || 0,
          badge_counts: response.badge_counts || {}
        };

        // Get available categories (only those with emails)
        this.availableCategories = Object.keys(this.groupedEmails).filter(
          cat => this.groupedEmails[cat].length > 0
        ) as BadgeType[];

        // Select all by default
        this.selectedCategories = new Set(this.availableCategories);

        this.isLoading = false;
      },
      error: (error) => {
        console.error('Error loading digest data:', error);
        this.isLoading = false;
        // Keep empty state on error
      }
    });
  }

  extractCallToActions(): void {
    this.allCallToActions = [];
    Object.entries(this.groupedEmails).forEach(([category, emails]) => {
      emails.forEach(email => {
        if (email.has_cta && email.cta_text) {
          this.allCallToActions.push({
            email_id: email.id,
            subject: email.subject,
            cta_text: email.cta_text,
            cta_type: email.cta_type || 'action',
            category: category as BadgeType
          });
        }
      });
    });
  }

  selectAllCategories(): void {
    this.selectedCategories = new Set(this.availableCategories);
  }

  deselectAllCategories(): void {
    this.selectedCategories.clear();
  }

  toggleCategory(category: BadgeType): void {
    if (this.selectedCategories.has(category)) {
      this.selectedCategories.delete(category);
    } else {
      this.selectedCategories.add(category);
    }
  }

  isCategorySelected(category: BadgeType): boolean {
    return this.selectedCategories.has(category);
  }

  getFilteredCategories(): BadgeType[] {
    return this.availableCategories.filter(cat => this.selectedCategories.has(cat));
  }

  getEmailsForCategory(category: BadgeType): EmailSummary[] {
    return this.groupedEmails[category] || [];
  }

  formatDate(dateString: string): string {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 60) {
      return `${diffMins}m ago`;
    } else if (diffMins < 1440) {
      return `${Math.floor(diffMins / 60)}h ago`;
    } else {
      return date.toLocaleDateString();
    }
  }

  viewEmail(emailSummary: EmailSummary): void {
    // Fetch the full email details from the mailbox API
    this.isLoading = true;
    const apiUrl = `${environment.apiUrl}/emails/${emailSummary.id}`;

    this.http.get<Email>(apiUrl).subscribe({
      next: (fullEmail) => {
        this.selectedEmail = fullEmail;
        this.showEmailModal = true;
        this.isLoading = false;
      },
      error: (error) => {
        console.error('Error loading full email details:', error);
        // Fallback to basic email data if API fails
        this.selectedEmail = {
          id: emailSummary.id,
          subject: emailSummary.subject,
          sender: emailSummary.sender,
          recipient: emailSummary.recipient || '',
          body_text: emailSummary.body_text,
          received_at: emailSummary.received_at,
          processed: emailSummary.processed || false,
          is_phishing: emailSummary.is_phishing || false,
          llm_processed: false,
          enriched: false,
          wiki_enriched: false,
          phone_enriched: false,
          // REMOVED: label and phishing_type - Ground truth fields removed
          suggested_team: null,
          assigned_team: null,
          badges: emailSummary.badges,
          ui_badges: emailSummary.badges,
          workflow_results: [],
          summary: Array.isArray(emailSummary.summary) ? emailSummary.summary.join(' ') : (emailSummary.summary ? String(emailSummary.summary) : null),
          call_to_actions: emailSummary.has_cta && emailSummary.cta_text ? [emailSummary.cta_text] : []
        };
        this.showEmailModal = true;
        this.isLoading = false;
      }
    });
  }

  closeModal(): void {
    this.showEmailModal = false;
    this.selectedEmail = null;
  }

  handleCTA(cta: CallToAction, event?: Event): void {
    if (event) {
      event.stopPropagation();
    }
    console.log('CTA clicked:', cta);

    // Find the email associated with this CTA in both grouped and other emails
    let email = Object.values(this.groupedEmails)
      .flat()
      .find(e => e.id === cta.email_id);

    // If not found in grouped emails, search in other emails
    if (!email) {
      email = this.otherEmails.find(e => e.id === cta.email_id);
    }

    if (email) {
      // Show the email details modal
      this.viewEmail(email);
    } else {
      // Log error if email not found
      console.error('Email not found for CTA:', cta);
    }
  }
}
