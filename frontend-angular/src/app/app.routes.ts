import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    redirectTo: '/dashboard',
    pathMatch: 'full'
  },
  {
    path: 'dashboard',
    loadComponent: () => import('./features/dashboard/dashboard.component')
      .then(m => m.DashboardComponent)
  },
  {
    path: 'mailbox',
    loadComponent: () => import('./features/mailbox/mailbox.component')
      .then(m => m.MailboxComponent)
  },
  {
    path: 'daily-inbox-digest',
    loadComponent: () => import('./features/daily-inbox-digest/daily-inbox-digest.component')
      .then(m => m.DailyInboxDigestComponent)
  },
  {
    path: 'auto-respond',
    loadComponent: () => import('./features/auto-respond/auto-respond.component')
      .then(m => m.AutoRespondComponent)
  },
  {
    path: 'agentic-teams',
    loadComponent: () => import('./features/agentic-teams/agentic-teams.component')
      .then(m => m.AgenticTeamsComponent)
  }
  // Future routes (to be added during migration):
  // Will be replaced with actual components once migrated
];
