import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment';
import { Email, TeamAssignment } from '../models/email.model';

@Injectable({
  providedIn: 'root'
})
export class EmailService {
  private http = inject(HttpClient);
  private apiUrl = `${environment.apiUrl}/emails`;

  getEmails(limit: number = 50, offset: number = 0): Observable<Email[]> {
    const params = new HttpParams()
      .set('limit', limit.toString())
      .set('offset', offset.toString());
    return this.http.get<Email[]>(this.apiUrl, { params });
  }

  getEmailById(id: number): Observable<Email> {
    return this.http.get<Email>(`${this.apiUrl}/${id}`);
  }

  fetchEmailsFromMailpit(): Observable<{ fetched: number; total_in_mailpit: number }> {
    return this.http.post<{ fetched: number; total_in_mailpit: number }>(`${this.apiUrl}/fetch`, {});
  }

  processAllEmails(): Observable<{ count: number }> {
    return this.http.post<{ count: number }>(`${this.apiUrl}/process-all`, {});
  }

  analyzeEmailTasks(emailId: number, team: string): Observable<{ email_id: number; team: string; team_name: string; task_options: any[] }> {
    return this.http.post<{ email_id: number; team: string; team_name: string; task_options: any[] }>(
      `${this.apiUrl}/${emailId}/analyze-tasks`,
      { team }
    );
  }

  assignTeamToEmail(assignment: TeamAssignment): Observable<{ status: string; email_id: number; assigned_team: string; task_id: string; workflow_url: string }> {
    return this.http.post<{ status: string; email_id: number; assigned_team: string; task_id: string; workflow_url: string }>(
      `${this.apiUrl}/${assignment.emailId}/assign-team`,
      { team: assignment.team, assignment_message: assignment.message, selected_task: assignment.selectedTask }
    );
  }

  startFraudDetection(emailId: number, message?: string): Observable<{ status: string; session_id: string }> {
    return this.http.post<{ status: string; session_id: string }>(
      `${this.apiUrl}/${emailId}/fraud-detect`,
      { message }
    );
  }

  submitDirectQuery(team: string, query: string): Observable<{ task_id: string; email_id: number }> {
    return this.http.post<{ task_id: string; email_id: number }>(
      `${environment.apiUrl}/agentic/direct-query`,
      { team, query }
    );
  }

  // Fetcher control
  startFetcher(): Observable<{ status: string; message: string }> {
    return this.http.post<{ status: string; message: string }>(
      `${environment.apiUrl}/fetcher/start`,
      {}
    );
  }

  stopFetcher(): Observable<{ status: string; message: string }> {
    return this.http.post<{ status: string; message: string }>(
      `${environment.apiUrl}/fetcher/stop`,
      {}
    );
  }

  getFetcherStatus(): Observable<{ running: boolean; current_batch?: number; total_fetched?: number }> {
    return this.http.get<{ running: boolean; current_batch?: number; total_fetched?: number }>(
      `${environment.apiUrl}/fetcher/status`
    );
  }

  // Get emails for workflow stats (limited to 100 most recent)
  getRecentEmailsForWorkflowStats(): Observable<Email[]> {
    const params = new HttpParams().set('limit', '100').set('offset', '0');
    return this.http.get<Email[]>(this.apiUrl, { params });
  }
}
