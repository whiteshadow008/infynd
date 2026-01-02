"""
AI Hackathon Minds 2025 - Complete Digital Twin Management System
Flask Backend + Cyberpunk Frontend in Single File

Run: python digitaltwin_complete.py
Access: http://localhost:5000
"""

from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
import datetime
import random
import threading
import time
import secrets
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from functools import wraps

app = Flask(__name__)
CORS(app)

# ============================================================================
# HTML TEMPLATE (CYBERPUNK FRONTEND)
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Digital Twin Dashboard - AI Hackathon 2025</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
        *{margin:0;padding:0;box-sizing:border-box}
        :root{--primary:#00ff88;--secondary:#0066ff;--dark:#0a0e1a;--card:#151925;--text:#e0e6f0;--text-dim:#8b95a8}
        body{font-family:'Space Grotesk',sans-serif;background:var(--dark);color:var(--text);overflow-x:hidden}
        .container{max-width:1800px;margin:0 auto;padding:2rem}
        header{display:flex;justify-content:space-between;align-items:center;padding:2rem 0;border-bottom:1px solid rgba(0,255,136,0.1);margin-bottom:3rem;flex-wrap:wrap;gap:1rem}
        .logo{display:flex;align-items:center;gap:1rem}
        .logo-icon{width:50px;height:50px;background:linear-gradient(135deg,var(--primary),var(--secondary));border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1.5rem;font-weight:700}
        .logo h1{font-size:1.8rem;background:linear-gradient(135deg,var(--primary),var(--secondary));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
        .header-actions{display:flex;gap:1rem;align-items:center}
        .api-status{display:flex;align-items:center;gap:0.5rem;padding:0.5rem 1rem;background:rgba(0,255,136,0.1);border-radius:10px;font-size:0.85rem}
        .status-dot{width:8px;height:8px;background:var(--primary);border-radius:50%;animation:pulse 2s infinite}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.5}}
        .btn{padding:0.75rem 1.5rem;border:none;border-radius:10px;font-weight:600;cursor:pointer;font-family:'Space Grotesk',sans-serif;transition:all 0.3s ease}
        .btn-primary{background:linear-gradient(135deg,var(--primary),var(--secondary));color:var(--dark)}
        .btn-primary:hover{transform:translateY(-2px);box-shadow:0 10px 30px rgba(0,255,136,0.3)}
        .btn-ghost{background:transparent;color:var(--text);border:1px solid rgba(255,255,255,0.1)}
        .btn-ghost:hover{background:rgba(255,255,255,0.05)}
        .btn-sm{padding:0.5rem 1rem;font-size:0.85rem}
        .card{background:var(--card);border:1px solid rgba(0,255,136,0.1);border-radius:20px;padding:2rem;transition:all 0.3s ease}
        .card:hover{border-color:var(--primary);transform:translateY(-2px);box-shadow:0 10px 30px rgba(0,255,136,0.1)}
        .metrics-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:1.5rem;margin-bottom:3rem}
        .metric-label{font-size:0.85rem;color:var(--text-dim);text-transform:uppercase;margin-bottom:0.5rem;letter-spacing:1px}
        .metric-value{font-size:2.5rem;font-weight:700;background:linear-gradient(135deg,var(--primary),var(--secondary));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
        .metric-subtitle{font-size:0.9rem;color:var(--text-dim);margin-top:0.5rem}
        .tabs{display:flex;gap:1rem;margin-bottom:2rem;border-bottom:2px solid rgba(255,255,255,0.05);overflow-x:auto}
        .tab{padding:1rem 2rem;cursor:pointer;border-bottom:3px solid transparent;transition:all 0.3s;font-weight:600;white-space:nowrap}
        .tab.active{border-bottom-color:var(--primary);color:var(--primary)}
        .tab:hover{color:var(--primary)}
        .tab-content{display:none}
        .tab-content.active{display:block;animation:fadeIn 0.3s ease}
        @keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
        .projects-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(400px,1fr));gap:1.5rem;margin-bottom:3rem}
        .project-card{background:var(--card);border:1px solid rgba(0,255,136,0.1);border-radius:20px;padding:2rem;transition:all 0.3s ease}
        .project-card:hover{border-color:var(--primary);transform:translateY(-2px);box-shadow:0 10px 30px rgba(0,255,136,0.1)}
        .project-header{display:flex;justify-content:space-between;align-items:start;margin-bottom:1rem}
        .project-title{font-size:1.3rem;font-weight:700;color:var(--text)}
        .priority-badge{padding:0.4rem 0.8rem;border-radius:8px;font-size:0.75rem;font-weight:700;text-transform:uppercase}
        .priority-HIGH{background:rgba(255,59,48,0.2);color:#ff3b30;border:1px solid #ff3b30}
        .priority-MEDIUM{background:rgba(0,102,255,0.2);color:#0066ff;border:1px solid #0066ff}
        .priority-CRITICAL{background:rgba(255,0,0,0.3);color:#ff0000;border:1px solid #ff0000}
        .project-team{color:var(--text-dim);font-size:0.9rem;margin-bottom:1rem}
        .progress-container{margin:1.5rem 0}
        .progress-label{display:flex;justify-content:space-between;margin-bottom:0.5rem;font-size:0.9rem}
        .progress-bar{width:100%;height:8px;background:rgba(0,255,136,0.1);border-radius:10px;overflow:hidden}
        .progress-fill{height:100%;background:linear-gradient(90deg,var(--primary),var(--secondary));transition:width 0.5s ease;border-radius:10px}
        .project-info{display:grid;grid-template-columns:1fr 1fr;gap:1rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.05)}
        .info-item{font-size:0.85rem}
        .info-label{color:var(--text-dim);margin-bottom:0.25rem}
        .info-value{color:var(--text);font-weight:600}
        .alerts-section{margin-bottom:3rem}
        .section-title{font-size:1.5rem;font-weight:700;margin-bottom:1.5rem;display:flex;align-items:center;gap:1rem}
        .alert-card{background:var(--card);border-left:4px solid;border-radius:15px;padding:1.5rem;margin-bottom:1rem;display:flex;align-items:start;gap:1rem}
        .alert-HIGH{border-left-color:#ff3b30}
        .alert-MEDIUM{border-left-color:#ff9500}
        .alert-icon{font-size:1.5rem}
        .alert-content{flex:1}
        .alert-type{font-size:0.85rem;color:var(--text-dim);text-transform:uppercase;margin-bottom:0.25rem}
        .alert-message{font-size:1rem;color:var(--text)}
        .events-section{margin-bottom:3rem}
        .event-card{background:var(--card);border:1px solid rgba(0,255,136,0.1);border-radius:15px;padding:1.5rem;margin-bottom:1rem;transition:all 0.3s}
        .event-card:hover{border-color:var(--primary);transform:translateX(5px)}
        .event-time{font-size:0.85rem;color:var(--text-dim);margin-bottom:0.5rem}
        .event-description{color:var(--text);line-height:1.6}
        .modal{position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.8);display:none;align-items:center;justify-content:center;z-index:1000;backdrop-filter:blur(10px)}
        .modal.active{display:flex;animation:fadeIn 0.3s ease}
        .modal-content{background:var(--card);border:1px solid rgba(0,255,136,0.1);border-radius:20px;padding:2rem;max-width:600px;width:90%;max-height:80vh;overflow-y:auto}
        .modal-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:1.5rem}
        .modal-title{font-size:1.5rem;font-weight:700}
        .close-btn{background:none;border:none;color:var(--text);font-size:1.5rem;cursor:pointer}
        .form-group{margin-bottom:1.5rem}
        .form-label{display:block;margin-bottom:0.5rem;color:var(--text-dim);font-size:0.9rem;font-weight:600}
        .form-input{width:100%;padding:0.75rem 1rem;background:rgba(0,0,0,0.3);border:1px solid rgba(255,255,255,0.1);border-radius:10px;color:var(--text);font-family:'Space Grotesk',sans-serif;font-size:1rem}
        .form-input:focus{outline:none;border-color:var(--primary);box-shadow:0 0 0 3px rgba(0,255,136,0.1)}
        .loading{text-align:center;padding:3rem;color:var(--text-dim)}
        .loading-spinner{width:50px;height:50px;border:3px solid rgba(0,255,136,0.1);border-top-color:var(--primary);border-radius:50%;animation:spin 1s linear infinite;margin:0 auto 1rem}
        @keyframes spin{to{transform:rotate(360deg)}}
        .error-message{background:rgba(255,59,48,0.1);border:1px solid #ff3b30;border-radius:15px;padding:1.5rem;color:#ff3b30;margin-bottom:2rem}
        @media(max-width:768px){
            .projects-grid{grid-template-columns:1fr}
            .tabs{overflow-x:scroll}
            header{flex-direction:column;align-items:stretch}
            .header-actions{width:100%}
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">
                <div class="logo-icon">⚡</div>
                <div>
                    <h1>Digital Twin</h1>
                    <p style="color:var(--text-dim);font-size:0.9rem">AI Hackathon Minds 2025</p>
                </div>
            </div>
            <div class="header-actions">
                <div class="api-status">
                    <div class="status-dot"></div>
                    <span>Connected</span>
                </div>
                <button class="btn btn-ghost btn-sm" onclick="openApiKeyModal()">⚙️ API Key</button>
                <button class="btn btn-primary btn-sm" onclick="loadDashboard()">🔄 Refresh</button>
            </div>
        </header>

        <div id="loading" class="loading">
            <div class="loading-spinner"></div>
            <p>Connecting to Digital Twin System...</p>
        </div>

        <div id="error" style="display:none;" class="error-message"></div>

        <div id="dashboard" style="display:none;">
            <!-- Metrics -->
            <div class="metrics-grid" id="metricsGrid"></div>

            <!-- Tabs -->
            <div class="tabs">
                <div class="tab active" onclick="switchTab('projects')">📊 Projects</div>
                <div class="tab" onclick="switchTab('alerts')">⚠️ Alerts</div>
                <div class="tab" onclick="switchTab('events')">📝 Events</div>
            </div>

            <!-- Tab Content -->
            <div id="tab-projects" class="tab-content active">
                <h2 class="section-title">
                    <span>Active Projects</span>
                </h2>
                <div class="projects-grid" id="projectsGrid"></div>
            </div>

            <div id="tab-alerts" class="tab-content">
                <h2 class="section-title">
                    <span class="status-dot"></span>
                    <span>Critical Alerts</span>
                </h2>
                <div id="alertsContainer"></div>
            </div>

            <div id="tab-events" class="tab-content">
                <h2 class="section-title">
                    <span class="status-dot"></span>
                    <span>Recent Events</span>
                </h2>
                <div id="eventsContainer"></div>
            </div>
        </div>
    </div>

    <!-- API Key Modal -->
    <div id="apiKeyModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2 class="modal-title">API Key Configuration</h2>
                <button class="close-btn" onclick="closeApiKeyModal()">×</button>
            </div>
            <div class="form-group">
                <label class="form-label">API Key</label>
                <input type="text" id="apiKeyInput" class="form-input" 
                       placeholder="Enter your API key" value="admin_key_12345">
                <p style="color:var(--text-dim);font-size:0.85rem;margin-top:0.5rem">
                    Default keys: admin_key_12345 (admin) | readonly_key_67890 (read-only)
                </p>
            </div>
            <button class="btn btn-primary" style="width:100%" onclick="saveApiKey()">Connect</button>
        </div>
    </div>

    <script>
        let API_KEY = localStorage.getItem('digital_twin_api_key') || 'admin_key_12345';
        const API_BASE = '/api';
        let refreshInterval;
        let currentTab = 'projects';

        function switchTab(tab) {
            currentTab = tab;
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('tab-' + tab).classList.add('active');
        }

        function openApiKeyModal() {
            document.getElementById('apiKeyModal').classList.add('active');
            document.getElementById('apiKeyInput').value = API_KEY;
        }

        function closeApiKeyModal() {
            document.getElementById('apiKeyModal').classList.remove('active');
        }

        function saveApiKey() {
            API_KEY = document.getElementById('apiKeyInput').value.trim();
            localStorage.setItem('digital_twin_api_key', API_KEY);
            closeApiKeyModal();
            loadDashboard();
        }

        async function fetchWithAuth(endpoint) {
            const response = await fetch(API_BASE + endpoint, {
                headers: { 'X-API-Key': API_KEY }
            });
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || 'API request failed');
            }
            return await response.json();
        }

        async function loadDashboard() {
            try {
                document.getElementById('loading').style.display = 'block';
                document.getElementById('dashboard').style.display = 'none';
                document.getElementById('error').style.display = 'none';

                const [analytics, projects, alerts, events] = await Promise.all([
                    fetchWithAuth('/analytics'),
                    fetchWithAuth('/projects'),
                    fetchWithAuth('/alerts'),
                    fetchWithAuth('/events?limit=15')
                ]);

                renderMetrics(analytics);
                renderProjects(projects);
                renderAlerts(alerts);
                renderEvents(events);

                document.getElementById('loading').style.display = 'none';
                document.getElementById('dashboard').style.display = 'block';

                if (!refreshInterval) {
                    refreshInterval = setInterval(loadDashboard, 5000);
                }
            } catch (error) {
                document.getElementById('error').textContent = '❌ Error: ' + error.message;
                document.getElementById('error').style.display = 'block';
                document.getElementById('loading').style.display = 'none';
                clearInterval(refreshInterval);
            }
        }

        function renderMetrics(analytics) {
            document.getElementById('metricsGrid').innerHTML = `
                <div class="card">
                    <div class="metric-label">Total Projects</div>
                    <div class="metric-value">${analytics.total_projects}</div>
                    <div class="metric-subtitle">Active hackathon projects</div>
                </div>
                <div class="card">
                    <div class="metric-label">Average Progress</div>
                    <div class="metric-value">${analytics.avg_progress}%</div>
                    <div class="metric-subtitle">Across all teams</div>
                </div>
                <div class="card">
                    <div class="metric-label">Completion Rate</div>
                    <div class="metric-value">${analytics.completion_rate}%</div>
                    <div class="metric-subtitle">${analytics.completed_steps}/${analytics.total_steps} steps done</div>
                </div>
                <div class="card">
                    <div class="metric-label">Open Issues</div>
                    <div class="metric-value">${analytics.open_issues}</div>
                    <div class="metric-subtitle">${analytics.resolved_issues} resolved</div>
                </div>
            `;
        }

        function renderProjects(projects) {
            document.getElementById('projectsGrid').innerHTML = projects.map(p => `
                <div class="project-card">
                    <div class="project-header">
                        <div class="project-title">${p.name}</div>
                        <div class="priority-badge priority-${p.priority}">${p.priority}</div>
                    </div>
                    <div class="project-team">${p.team}</div>
                    <div class="progress-container">
                        <div class="progress-label">
                            <span>Progress</span>
                            <span style="font-weight:700">${p.progress}%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width:${p.progress}%"></div>
                        </div>
                    </div>
                    <div class="project-info">
                        <div class="info-item">
                            <div class="info-label">Deadline</div>
                            <div class="info-value">${p.deadline}</div>
                            <div style="font-size:0.75rem;color:var(--text-dim)">(${p.days_until_deadline} days)</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Budget</div>
                            <div class="info-value">$${p.budget_used.toLocaleString()}</div>
                            <div style="font-size:0.75rem;color:var(--text-dim)">of $${p.budget_allocated.toLocaleString()}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Steps</div>
                            <div class="info-value">${p.steps.filter(s => s.completed).length}/${p.steps.length}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Issues</div>
                            <div class="info-value">${p.issues.length}</div>
                        </div>
                    </div>
                </div>
            `).join('');
        }

        function renderAlerts(alerts) {
            const container = document.getElementById('alertsContainer');
            if (alerts.length === 0) {
                container.innerHTML = '<div class="card"><p style="color:var(--text-dim)">No alerts at this time</p></div>';
                return;
            }
            container.innerHTML = alerts.map(a => `
                <div class="alert-card alert-${a.severity}">
                    <div class="alert-icon">⚠️</div>
                    <div class="alert-content">
                        <div class="alert-type">${a.type.replace(/_/g, ' ')}</div>
                        <div class="alert-message">${a.message}</div>
                    </div>
                </div>
            `).join('');
        }

        function renderEvents(events) {
            document.getElementById('eventsContainer').innerHTML = events.reverse().map(e => `
                <div class="event-card">
                    <div class="event-time">🕐 ${e.timestamp}</div>
                    <div class="event-description">${e.description}</div>
                </div>
            `).join('');
        }

        // Initialize
        window.onload = () => {
            document.getElementById('apiKeyInput').value = API_KEY;
            loadDashboard();
        };

        // Close modal on escape
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeApiKeyModal();
        });

        // Close modal on background click
        document.getElementById('apiKeyModal').addEventListener('click', (e) => {
            if (e.target.id === 'apiKeyModal') closeApiKeyModal();
        });
    </script>
</body>
</html>
"""

# ============================================================================
# API KEY MANAGEMENT
# ============================================================================

class APIKeyManager:
    def __init__(self):
        self.api_keys = {}
        self._initialize_keys()
    
    def _initialize_keys(self):
        self.api_keys["admin_key_12345"] = {
            "name": "Admin",
            "permissions": ["read", "write", "delete"],
            "created_at": datetime.datetime.now().isoformat(),
            "usage_count": 0
        }
        self.api_keys["readonly_key_67890"] = {
            "name": "Read Only",
            "permissions": ["read"],
            "created_at": datetime.datetime.now().isoformat(),
            "usage_count": 0
        }
    
    def validate_key(self, api_key: str) -> bool:
        return api_key in self.api_keys
    
    def has_permission(self, api_key: str, permission: str) -> bool:
        if api_key not in self.api_keys:
            return False
        return permission in self.api_keys[api_key]["permissions"]
    
    def record_usage(self, api_key: str):
        if api_key in self.api_keys:
            self.api_keys[api_key]["usage_count"] += 1

api_key_manager = APIKeyManager()

# ============================================================================
# AUTHENTICATION DECORATOR
# ============================================================================

def require_api_key(permission: str = "read"):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
            
            if not api_key:
                return jsonify({"error": "API key required", "message": "Please provide an API key"}), 401
            
            if not api_key_manager.validate_key(api_key):
                return jsonify({"error": "Invalid API key", "message": "The provided API key is not valid"}), 403
            
            if not api_key_manager.has_permission(api_key, permission):
                return jsonify({"error": "Insufficient permissions", "message": f"This API key does not have '{permission}' permission"}), 403
            
            api_key_manager.record_usage(api_key)
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ProjectStep:
    step_number: int
    description: str
    completed: bool
    started: bool = False
    completion_time: Optional[str] = None
    start_time: Optional[str] = None
    notes: str = ""
    assigned_to: str = ""

@dataclass
class Issue:
    id: str
    description: str
    severity: str
    status: str
    created_at: str
    resolved_at: Optional[str] = None
    assigned_to: str = ""
    resolution_notes: str = ""

@dataclass
class TeamMember:
    name: str
    role: str
    tasks_assigned: int
    tasks_completed: int
    availability: str

@dataclass
class HackathonProject:
    id: int
    name: str
    team: str
    status: str
    priority: str
    deadline: str
    steps: List[ProjectStep]
    issues: List[Issue]
    team_members: List[TeamMember]
    budget_allocated: float
    budget_used: float
    
    @property
    def progress(self) -> float:
        if not self.steps:
            return 0.0
        completed = sum(1 for step in self.steps if step.completed)
        return round((completed / len(self.steps)) * 100, 2)
    
    @property
    def days_until_deadline(self) -> int:
        deadline = datetime.datetime.strptime(self.deadline, "%Y-%m-%d")
        delta = deadline - datetime.datetime.now()
        return delta.days
    
    @property
    def budget_remaining(self) -> float:
        return self.budget_allocated - self.budget_used

# ============================================================================
# DIGITAL TWIN SYSTEM
# ============================================================================

class DigitalTwinHackathon:
    def __init__(self):
        self.projects: Dict[int, HackathonProject] = {}
        self.events_log: List[Dict] = []
        self.simulation_running = True
        self._initialize_projects()
        self._start_simulation()
    
    def _initialize_projects(self):
        self.projects[1] = HackathonProject(
            id=1, name="Personalized Campaign Generator", team="Team Alpha",
            status="In Progress", priority="HIGH", deadline="2025-10-20",
            steps=[
                ProjectStep(1, "Setup Platform", True, True, "2025-10-12 09:00", "2025-10-12 08:00", "Done", "Alice"),
                ProjectStep(2, "Data Upload", True, True, "2025-10-12 11:30", "2025-10-12 09:30", "Done", "Bob"),
                ProjectStep(3, "Segmentation Logic", True, True, "2025-10-12 14:00", "2025-10-12 12:00", "Done", "Alice"),
                ProjectStep(4, "Content Generation", False, True, None, "2025-10-12 14:30", "In progress", "Charlie"),
                ProjectStep(5, "Multi-channel Support", False, False, None, None, "Pending", "Diana")
            ],
            issues=[Issue("I1-001", "API rate limiting", "Medium", "Open", "2025-10-12 15:00", None, "Charlie", "")],
            team_members=[
                TeamMember("Alice", "Lead Dev", 2, 2, "Available"), TeamMember("Bob", "Data Engineer", 1, 1, "Available"),
                TeamMember("Charlie", "Backend Dev", 1, 0, "Busy"), TeamMember("Diana", "Frontend Dev", 1, 0, "Available")
            ],
            budget_allocated=5500.0, budget_used=2800.0
        )
        
        self.projects[2] = HackathonProject(
            id=2, name="Content Recommendation Engine", team="Team Beta",
            status="In Progress", priority="HIGH", deadline="2025-10-20",
            steps=[
                ProjectStep(1, "Platform Setup", True, True, "2025-10-12 09:30", "2025-10-12 08:30", "Done", "Emma"),
                ProjectStep(2, "Data Upload", True, True, "2025-10-12 12:00", "2025-10-12 10:00", "Done", "Frank"),
                ProjectStep(3, "Recommendation Logic", False, True, None, "2025-10-12 12:30", "Building", "Emma"),
                ProjectStep(4, "Output Generation", False, False, None, None, "Pending", "Grace"),
                ProjectStep(5, "Multi-channel Extension", False, False, None, None, "Pending", "Henry")
            ],
            issues=[],
            team_members=[
                TeamMember("Emma", "ML Engineer", 2, 1, "Busy"), TeamMember("Frank", "Data Analyst", 1, 1, "Available"),
                TeamMember("Grace", "Full Stack", 1, 0, "Available"), TeamMember("Henry", "DevOps", 1, 0, "Available")
            ],
            budget_allocated=5000.0, budget_used=3200.0
        )
        
        self.projects[3] = HackathonProject(
            id=3, name="Marketing Insights Chatbot", team="Team Gamma",
            status="In Progress", priority="MEDIUM", deadline="2025-10-20",
            steps=[
                ProjectStep(1, "Chatbot Setup", True, True, "2025-10-12 09:00", "2025-10-12 08:00", "Done", "Ivy"),
                ProjectStep(2, "Data Integration", True, True, "2025-10-12 11:00", "2025-10-12 09:30", "Done", "Jack"),
                ProjectStep(3, "Query Design", True, True, "2025-10-12 13:00", "2025-10-12 11:30", "Done", "Ivy"),
                ProjectStep(4, "Response Generation", True, True, "2025-10-12 15:00", "2025-10-12 13:30", "Done", "Kate"),
                ProjectStep(5, "Multi-query Support", False, True, None, "2025-10-12 15:30", "Testing", "Leo")
            ],
            issues=[
                Issue("I3-001", "Dashboard API timeout", "High", "Open", "2025-10-12 14:00", None, "Jack", ""),
                Issue("I3-002", "Query parsing issues", "Medium", "Open", "2025-10-12 14:30", None, "Ivy", "")
            ],
            team_members=[
                TeamMember("Ivy", "AI Dev", 2, 2, "Busy"), TeamMember("Jack", "Integration", 1, 1, "Available"),
                TeamMember("Kate", "UI/UX", 1, 1, "Available"), TeamMember("Leo", "QA", 1, 0, "Busy")
            ],
            budget_allocated=4800.0, budget_used=3600.0
        )
        
        self.projects[4] = HackathonProject(
            id=4, name="Smart Lead Scorer", team="Team Delta",
            status="In Progress", priority="HIGH", deadline="2025-10-20",
            steps=[
                ProjectStep(1, "Setup Environment", True, True, "2025-10-12 10:00", "2025-10-12 09:00", "Done", "Mike"),
                ProjectStep(2, "Data Preparation", True, True, "2025-10-12 12:30", "2025-10-12 10:30", "Done", "Nina"),
                ProjectStep(3, "Model Config", True, True, "2025-10-12 14:30", "2025-10-12 13:00", "Done", "Mike"),
                ProjectStep(4, "Scoring & Ranking", False, True, None, "2025-10-12 15:00", "Running", "Oscar"),
                ProjectStep(5, "Dashboard Viz", False, False, None, None, "Pending", "Paula")
            ],
            issues=[],
            team_members=[
                TeamMember("Mike", "Data Scientist", 2, 2, "Busy"), TeamMember("Nina", "Data Engineer", 1, 1, "Available"),
                TeamMember("Oscar", "ML Engineer", 1, 0, "Busy"), TeamMember("Paula", "Frontend", 1, 0, "Available")
            ],
            budget_allocated=6000.0, budget_used=3900.0
        )
        
        self.projects[5] = HackathonProject(
            id=5, name="Social Media Sentiment Tracker", team="Team Epsilon",
            status="In Progress", priority="MEDIUM", deadline="2025-10-20",
            steps=[
                ProjectStep(1, "Data Collection", True, True, "2025-10-12 09:30", "2025-10-12 08:30", "Done", "Quinn"),
                ProjectStep(2, "Sentiment Analysis", True, True, "2025-10-12 11:30", "2025-10-12 10:00", "Done", "Rachel"),
                ProjectStep(3, "Data Cleaning", True, True, "2025-10-12 13:30", "2025-10-12 12:00", "Done", "Sam"),
                ProjectStep(4, "Visualization", True, True, "2025-10-12 15:30", "2025-10-12 14:00", "Done", "Tina"),
                ProjectStep(5, "Real-time Updates", False, True, None, "2025-10-12 16:00", "Implementing", "Uma")
            ],
            issues=[Issue("I5-001", "Twitter API rate limits", "Medium", "Open", "2025-10-12 16:00", None, "Quinn", "")],
            team_members=[
                TeamMember("Quinn", "API Dev", 1, 1, "Available"), TeamMember("Rachel", "NLP", 1, 1, "Available"),
                TeamMember("Sam", "Analyst", 1, 1, "Available"), TeamMember("Tina", "Viz Expert", 1, 1, "Available"),
                TeamMember("Uma", "Real-time Dev", 1, 0, "Busy")
            ],
            budget_allocated=5200.0, budget_used=3700.0
        )
    
    def _start_simulation(self):
        def simulate():
            while self.simulation_running:
                time.sleep(random.randint(5, 15))
                self._generate_random_event()
        threading.Thread(target=simulate, daemon=True).start()
    
    def _generate_random_event(self):
        event_types = ["step_progress", "issue_update", "budget_change", "team_update"]
        event_type = random.choice(event_types)
        project_id = random.choice(list(self.projects.keys()))
        project = self.projects[project_id]
        
        if event_type == "step_progress":
            incomplete = [s for s in project.steps if not s.completed and s.started]
            if incomplete:
                step = random.choice(incomplete)
                progress = random.randint(10, 30)
                self.log_event("Step Progress", project_id,
                             f"🔄 {project.name} - Step {step.step_number}: {progress}% progress on {step.description}", "INFO")
        
        elif event_type == "issue_update":
            if project.issues and random.random() > 0.7:
                open_issues = [i for i in project.issues if i.status == "Open"]
                if open_issues:
                    issue = random.choice(open_issues)
                    issue.status = "Resolved"
                    issue.resolved_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.log_event("Issue Resolved", project_id,
                                 f"✅ {project.name} - Resolved: {issue.description}", "SUCCESS")
        
        elif event_type == "budget_change":
            increase = random.randint(50, 200)
            if project.budget_used + increase <= project.budget_allocated:
                project.budget_used += increase
                self.log_event("Budget Update", project_id,
                             f"💰 {project.name} - Budget increased by ${increase}", "INFO")
        
        elif event_type == "team_update":
            member = random.choice(project.team_members)
            if member.availability == "Busy" and random.random() > 0.5:
                member.availability = "Available"
                self.log_event("Team Update", project_id,
                             f"👤 {project.name} - {member.name} is now available", "INFO")
    
    def log_event(self, event_type: str, project_id: Optional[int], 
                  description: str, severity: str = "INFO"):
        event = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": event_type,
            "project_id": project_id,
            "description": description,
            "severity": severity
        }
        self.events_log.append(event)
        if len(self.events_log) > 100:
            self.events_log = self.events_log[-100:]
    
    def get_system_analytics(self) -> Dict:
        total_projects = len(self.projects)
        total_steps = sum(len(p.steps) for p in self.projects.values())
        completed_steps = sum(sum(1 for s in p.steps if s.completed) for p in self.projects.values())
        total_issues = sum(len(p.issues) for p in self.projects.values())
        resolved_issues = sum(sum(1 for i in p.issues if i.status == "Resolved") for p in self.projects.values())
        avg_progress = sum(p.progress for p in self.projects.values()) / len(self.projects)
        
        return {
            'total_projects': total_projects,
            'completed_projects': sum(1 for p in self.projects.values() if p.status == "Completed"),
            'total_steps': total_steps,
            'completed_steps': completed_steps,
            'total_issues': total_issues,
            'resolved_issues': resolved_issues,
            'avg_progress': round(avg_progress, 2),
            'completion_rate': round((completed_steps / total_steps * 100) if total_steps else 0, 2),
            'issue_resolution_rate': round((resolved_issues / total_issues * 100) if total_issues else 0, 2),
            'open_issues': total_issues - resolved_issues
        }
    
    def get_project_details(self, project_id: int) -> Dict:
        if project_id not in self.projects:
            return {"error": "Project not found"}
        
        project = self.projects[project_id]
        return {
            "id": project.id,
            "name": project.name,
            "team": project.team,
            "status": project.status,
            "priority": project.priority,
            "progress": project.progress,
            "deadline": project.deadline,
            "days_until_deadline": project.days_until_deadline,
            "steps": [asdict(step) for step in project.steps],
            "issues": [asdict(issue) for issue in project.issues],
            "team_members": [asdict(tm) for tm in project.team_members],
            "budget_allocated": project.budget_allocated,
            "budget_used": project.budget_used,
            "budget_remaining": project.budget_remaining
        }
    
    def get_all_projects_summary(self) -> List[Dict]:
        return [self.get_project_details(pid) for pid in self.projects.keys()]
    
    def get_critical_alerts(self) -> List[Dict]:
        alerts = []
        for project in self.projects.values():
            if 0 <= project.days_until_deadline <= 3:
                alerts.append({
                    "type": "DEADLINE_WARNING",
                    "severity": "HIGH",
                    "project_id": project.id,
                    "project_name": project.name,
                    "message": f"{project.name} deadline in {project.days_until_deadline} days"
                })
            
            if project.priority == "HIGH" and project.progress < 50:
                alerts.append({
                    "type": "LOW_PROGRESS",
                    "severity": "HIGH",
                    "project_id": project.id,
                    "project_name": project.name,
                    "message": f"HIGH PRIORITY: {project.name} only {project.progress:.0f}% complete"
                })
            
            open_issues = [i for i in project.issues if i.status == "Open"]
            if len(open_issues) >= 2:
                alerts.append({
                    "type": "MULTIPLE_ISSUES",
                    "severity": "MEDIUM",
                    "project_id": project.id,
                    "project_name": project.name,
                    "message": f"{project.name} has {len(open_issues)} open issues"
                })
        
        return sorted(alerts, key=lambda x: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}.get(x["severity"], 3))

# Initialize Digital Twin
digital_twin = DigitalTwinHackathon()

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve the cyberpunk dashboard"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/analytics')
@require_api_key(permission="read")
def get_analytics():
    """Get system analytics"""
    return jsonify(digital_twin.get_system_analytics())

@app.route('/api/projects')
@require_api_key(permission="read")
def get_projects():
    """Get all projects"""
    return jsonify(digital_twin.get_all_projects_summary())

@app.route('/api/project/<int:project_id>')
@require_api_key(permission="read")
def get_project(project_id):
    """Get specific project"""
    return jsonify(digital_twin.get_project_details(project_id))

@app.route('/api/alerts')
@require_api_key(permission="read")
def get_alerts():
    """Get critical alerts"""
    return jsonify(digital_twin.get_critical_alerts())

@app.route('/api/events')
@require_api_key(permission="read")
def get_events():
    """Get recent events"""
    limit = request.args.get('limit', 50, type=int)
    return jsonify(digital_twin.events_log[-limit:])

@app.route('/api/stats')
@require_api_key(permission="read")
def get_stats():
    """Get quick stats"""
    analytics = digital_twin.get_system_analytics()
    return jsonify({
        "total_projects": analytics['total_projects'],
        "avg_progress": analytics['avg_progress'],
        "open_issues": analytics['open_issues'],
        "event_count": len(digital_twin.events_log)
    })

@app.route('/api/health')
def health_check():
    """Health check (public)"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "version": "1.0.0"
    })

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("🚀 AI HACKATHON MINDS 2025 - DIGITAL TWIN COMPLETE SYSTEM")
    print("   WITH CYBERPUNK UI")
    print("=" * 80)
    print("\n📊 System Information:")
    print(f"  • Total Projects: {len(digital_twin.projects)}")
    print(f"  • Real-time Simulation: ENABLED")
    print(f"  • Cyberpunk Dashboard: INCLUDED")
    print(f"  • API Authentication: ENABLED")
    
    print("\n🔑 Default API Keys:")
    print("  • Admin Key: admin_key_12345 (read, write, delete)")
    print("  • Read-Only Key: readonly_key_67890 (read only)")
    
    print("\n🌐 Access Points:")
    print("  • Cyberpunk Dashboard: http://localhost:5000")
    print("  • API Health Check: http://localhost:5000/api/health")
    
    print("\n📋 API Endpoints:")
    print("  GET  /api/analytics - System analytics")
    print("  GET  /api/projects - All projects")
    print("  GET  /api/project/<id> - Project details")
    print("  GET  /api/alerts - Critical alerts")
    print("  GET  /api/events - Events log")
    print("  GET  /api/stats - Quick statistics")
    
    print("\n🎨 UI Features:")
    print("  • Cyberpunk theme with neon green accents")
    print("  • Real-time auto-refresh (5 seconds)")
    print("  • Tabbed navigation (Projects/Alerts/Events)")
    print("  • Animated progress bars and transitions")
    print("  • Mobile responsive design")
    
    print("\n" + "=" * 80)
    print("✅ System Ready! Open http://localhost:5000 in your browser")
    print("=" * 80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5006)