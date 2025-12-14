import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterModule, IsActiveMatchOptions } from '@angular/router';

interface NavItem {
  label: string;
  route: string;
  icon: string;
  queryParams?: any;
}

interface NavSection {
  header?: string;
  items: NavItem[];
}

@Component({
  selector: 'app-sidebar',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './sidebar.component.html',
  styleUrl: './sidebar.component.scss'
})
export class SidebarComponent {
  constructor(private router: Router) {}

  isLinkActive(item: NavItem): boolean {
    const urlTree = this.router.createUrlTree([item.route], {
      queryParams: item.queryParams || {}
    });

    const matchOptions: IsActiveMatchOptions = {
      paths: 'exact',
      queryParams: 'exact',
      fragment: 'ignored',
      matrixParams: 'ignored'
    };

    return this.router.isActive(urlTree, matchOptions);
  }
  navSections: NavSection[] = [
    {
      items: [
        { label: 'Dashboard', route: '/dashboard', icon: 'dashboard' },
        { label: 'Mailbox', route: '/mailbox', icon: 'email' },
        { label: 'Daily Inbox Digest', route: '/daily-inbox-digest', icon: 'summarize' },
        { label: 'Auto-Respond', route: '/auto-respond', icon: 'reply' }
      ]
    },
    {
      header: 'Virtual Teams',
      items: [
        { label: 'Fraud Unit', route: '/agentic-teams', icon: 'security', queryParams: { team: 'fraud' } },
        { label: 'Compliance', route: '/agentic-teams', icon: 'gavel', queryParams: { team: 'compliance' } },
        { label: 'Investments', route: '/agentic-teams', icon: 'query_stats', queryParams: { team: 'investments' } }
      ]
    }
  ];
}
