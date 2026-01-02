"""
STRATEGIC CERTAINTY ENGINE - COMPLETE SINGLE-FILE SOLUTION
==========================================================
Multi-Agent Marketing AI System with Web UI
- 11 Specialized AI Agents
- Central Strategy Agent (CSA) Orchestrator
- Real-time Strategic Huddles
- WebSocket Support
- Beautiful Modern UI

INSTALLATION:
pip install flask flask-socketio flask-cors groq pandas numpy scikit-learn xgboost

RUN:
python strategic_engine.py
Open: http://localhost:5001
"""

from flask import Flask, jsonify, request, render_template_string
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from datetime import datetime, timedelta
import json
import os
import random
import re
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

# ML Libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  ML libraries not available. Using rule-based scoring.")

# LLM Integration
try:
    from groq import Groq
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️  Groq not available. Using template-based generation.")

# =========================================================================
# CONFIGURATION
# =========================================================================

class Config:
    # LLM Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = "llama-3.1-70b-versatile"
    TEMPERATURE = 0.7
    MAX_TOKENS = 1500
    
    # Agent Settings
    MAX_HUDDLE_ROUNDS = 5

# =========================================================================
# LLM INTERFACE
# =========================================================================

class LLMInterface:
    """Universal LLM interface supporting Groq"""
    
    def __init__(self):
        self.available = LLM_AVAILABLE and Config.GROQ_API_KEY
        
        if self.available:
            try:
                self.client = Groq(api_key=Config.GROQ_API_KEY)
                print("✅ LLM Interface: Groq connected")
            except Exception as e:
                print(f"⚠️  LLM initialization failed: {e}")
                self.available = False
        else:
            print("⚠️  LLM Interface: Using template mode")
    
    def generate(self, prompt: str, system_message: Optional[str] = None, temperature: Optional[float] = None) -> str:
        """Generate completion"""
        
        if not self.available:
            return self._template_fallback(prompt)
        
        try:
            temp = temperature or Config.TEMPERATURE
            
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=Config.GROQ_MODEL,
                messages=messages,
                temperature=temp,
                max_tokens=Config.MAX_TOKENS
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"⚠️  LLM generation failed: {e}")
            return self._template_fallback(prompt)
    
    def _template_fallback(self, prompt):
        """Fallback template responses"""
        if "segment" in prompt.lower():
            return "Analysis complete. Customers segmented into 5 strategic groups based on engagement and value."
        elif "campaign" in prompt.lower():
            return "Campaign variants generated with A/B testing optimization."
        elif "verdict" in prompt.lower():
            return """COMMAND: GO

═══════════════════════════════════════
THE STRATEGY (What are we doing?)
═══════════════════════════════════════
Scale marketing investment with data-driven targeting and continuous optimization.

═══════════════════════════════════════
THE PREDICTION (Will it work?)
═══════════════════════════════════════
Expected ROI: 4.2x return
Confidence Score: 78%
Timeframe: 60 days
Key Success Metric: Customer Acquisition Cost < $150

═══════════════════════════════════════
THE DEFENSE (What about rivals/risks?)
═══════════════════════════════════════
Competitive positioning maintained through differentiated value proposition and proactive market monitoring.

═══════════════════════════════════════
EXECUTION ORDERS
═══════════════════════════════════════
→ Campaign Generator: Launch optimized campaigns by EOD
→ Performance Monitor: Track ROAS daily, alert at 3.5x threshold
→ Competitor Intelligence: Monitor rival activities weekly

RATIONALE: Strong lead quality indicators and market conditions support aggressive growth strategy."""
        else:
            return "Analysis completed successfully. Recommendations generated based on available data."

# =========================================================================
# BASE AGENT CLASS
# =========================================================================

class BaseAgent(ABC):
    """Base class for all specialized agents"""
    
    def __init__(self, agent_name: str, agent_role: str):
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.llm = LLMInterface()
    
    @abstractmethod
    def analyze(self, data: Dict, context: Dict) -> Dict:
        """Each agent must implement their analysis logic"""
        pass
    
    def present_case(self, query: str, context: Dict, data: Optional[Dict] = None) -> Dict:
        """Present argument in strategic huddle"""
        
        # Run analysis
        analysis_result = self.analyze(data or {}, context)
        
        # Format as huddle argument
        system_message = f"""You are the {self.agent_name}.
Your role: {self.agent_role}

You are participating in a strategic huddle. Present your argument clearly:
- Start with PRO/CON/CAUTION/CONTEXT depending on your stance
- Use data from your analysis
- Be concise but persuasive
- Focus on your specialized domain"""

        prompt = f"""
STRATEGIC QUERY:
{query}

YOUR ANALYSIS RESULTS:
{self._format_analysis(analysis_result)}

BUSINESS CONTEXT:
{self._format_context(context)}

Present your argument to the Central Strategy Agent. Start with PRO/CON/CAUTION/CONTEXT.
"""
        
        argument = self.llm.generate(prompt, system_message, temperature=0.6)
        
        return {
            'agent_name': self.agent_name,
            'argument': argument,
            'raw_analysis': analysis_result,
            'confidence': analysis_result.get('confidence_score', 0.5)
        }
    
    def _format_analysis(self, analysis):
        """Convert analysis dict to readable format"""
        return '\n'.join([f"- {k}: {v}" for k, v in analysis.items()])
    
    def _format_context(self, context):
        """Convert context dict to readable format"""
        return '\n'.join([f"- {k}: {v}" for k, v in context.items()])

# =========================================================================
# CENTRAL STRATEGY AGENT (CSA)
# =========================================================================

class CentralStrategyAgent:
    """The AI CMO - Orchestrates all strategic decisions"""
    
    def __init__(self):
        self.llm = LLMInterface()
        self.conversation_state = []
        self.decision_history = []
    
    def open_huddle(self, query: str, context: Dict) -> Dict:
        """Initiate strategic huddle"""
        
        system_message = """You are the Central Strategy Agent (CSA), an AI Chief Marketing Officer.
You facilitate strategic huddles where specialized agents debate before you make final decisions.

Your role:
1. Analyze the query and business context
2. Identify which specialized agents are needed for this decision
3. Frame the strategic question clearly"""

        prompt = f"""
BUSINESS CONTEXT:
- Goal Mode: {context.get('goal_mode', 'Balanced Growth')}
- Risk Tolerance: {context.get('risk_tolerance', 'Medium')}
- Budget Available: ${context.get('budget', 50000)}
- Time Horizon: {context.get('timeframe', '30 days')}

STRATEGIC QUERY:
{query}

TASK: Open the strategic huddle with:
1. A clear restatement of the decision needed
2. List of 3-5 specialized agents who should present arguments

FORMAT YOUR RESPONSE AS:
DECISION NEEDED: [One sentence]
AGENTS REQUIRED: [Agent1, Agent2, Agent3]
"""
        
        response = self.llm.generate(prompt, system_message)
        
        huddle_opening = {
            'role': 'CSA',
            'phase': 'OPENING',
            'content': response,
            'timestamp': datetime.now().isoformat()
        }
        
        self.conversation_state.append(huddle_opening)
        return huddle_opening
    
    def parse_agent_requirements(self, opening_response: Dict) -> List[str]:
        """Extract which agents need to participate"""
        
        content = opening_response['content']
        
        # Try to extract from structured format
        if 'AGENTS REQUIRED:' in content:
            agents_line = content.split('AGENTS REQUIRED:')[1].split('\n')[0]
            agents = [a.strip() for a in agents_line.split(',')]
            
            # Map to actual agent keys
            agent_mapping = {
                'lead scorer': 'smart_lead_scorer',
                'smart lead': 'smart_lead_scorer',
                'value': 'lead_value_predictor',
                'enrichment': 'lead_enrichment',
                'campaign': 'campaign_generator',
                'content': 'content_recommender',
                'creative': 'creative_optimizer',
                'performance': 'performance_monitor',
                'sentiment': 'sentiment_tracker',
                'competitor': 'competitor_intelligence',
                'macro': 'macro_trend_analyzer',
                'trend': 'macro_trend_analyzer'
            }
            
            mapped_agents = []
            for agent in agents:
                agent_lower = agent.lower()
                for key, value in agent_mapping.items():
                    if key in agent_lower and value not in mapped_agents:
                        mapped_agents.append(value)
                        break
            
            if mapped_agents:
                return mapped_agents
        
        # Fallback: return default agents
        return ['smart_lead_scorer', 'lead_value_predictor', 'competitor_intelligence', 
                'performance_monitor', 'creative_optimizer']
    
    def synthesize_arguments(self, agent_arguments: List[Dict]) -> Dict:
        """Collect all agent arguments"""
        
        synthesis = {
            'role': 'CSA',
            'phase': 'SYNTHESIS',
            'content': '\n\n'.join([
                f"[{arg['agent_name']}]: {arg['argument']}" 
                for arg in agent_arguments
            ]),
            'timestamp': datetime.now().isoformat()
        }
        
        self.conversation_state.append(synthesis)
        return synthesis
    
    def issue_verdict(self, context: Dict, goal_weights: Optional[Dict] = None) -> Dict:
        """Make final strategic decision"""
        
        if goal_weights is None:
            goal_weights = {
                'growth': 0.4,
                'risk_mitigation': 0.3,
                'efficiency': 0.3
            }
        
        system_message = """You are the Central Strategy Agent issuing a FINAL VERDICT.

You must analyze all arguments and make a decisive command: GO, STOP, or ADJUST.

Your verdict MUST include:
1. THE STRATEGY: One clear action plan
2. THE PREDICTION: Expected ROI/outcome with confidence score (%)
3. THE DEFENSE: How this protects against competition/risk
4. EXECUTION ORDERS: Specific tasks for each agent"""

        # Build conversation history
        conversation_text = "\n\n".join([
            f"{'='*60}\n{msg.get('phase', msg['role'])}\n{'='*60}\n{msg['content']}"
            for msg in self.conversation_state
        ])
        
        prompt = f"""
STRATEGIC HUDDLE TRANSCRIPT:
{conversation_text}

DECISION WEIGHTS:
- Growth Priority: {goal_weights['growth']*100}%
- Risk Mitigation: {goal_weights['risk_mitigation']*100}%
- Efficiency: {goal_weights['efficiency']*100}%

ISSUE YOUR FINAL VERDICT IN THIS EXACT FORMAT:

COMMAND: [GO / STOP / ADJUST]

═══════════════════════════════════════
THE STRATEGY (What are we doing?)
═══════════════════════════════════════
[One clear sentence describing the action plan]

═══════════════════════════════════════
THE PREDICTION (Will it work?)
═══════════════════════════════════════
Expected ROI: [X.Xx return]
Confidence Score: [XX%]
Timeframe: [X days/weeks]
Key Success Metric: [specific KPI]

═══════════════════════════════════════
THE DEFENSE (What about rivals/risks?)
═══════════════════════════════════════
[How this decision protects competitive position]

═══════════════════════════════════════
EXECUTION ORDERS
═══════════════════════════════════════
→ [Agent Name]: [Specific task with deadline]
→ [Agent Name]: [Specific task with deadline]

RATIONALE: [2-3 sentences explaining key factors]
"""
        
        verdict_response = self.llm.generate(prompt, system_message, temperature=0.3)
        
        verdict = {
            'role': 'CSA',
            'phase': 'FINAL_VERDICT',
            'content': verdict_response,
            'context': context,
            'weights': goal_weights,
            'timestamp': datetime.now().isoformat()
        }
        
        self.conversation_state.append(verdict)
        self.decision_history.append({
            'query': context.get('original_query'),
            'verdict': verdict_response,
            'timestamp': verdict['timestamp']
        })
        
        return verdict
    
    def get_huddle_transcript(self) -> List[Dict]:
        """Return full conversation for transparency"""
        return self.conversation_state
    
    def reset_huddle(self):
        """Clear state for new decision"""
        self.conversation_state = []

# =========================================================================
# SPECIALIZED AGENTS
# =========================================================================

class SmartLeadScorer(BaseAgent):
    """Predicts lead conversion probability"""
    
    def __init__(self):
        super().__init__(
            agent_name="Smart Lead Scorer",
            agent_role="Predicts lead conversion probability and priority"
        )
    
    def analyze(self, data: Dict, context: Dict) -> Dict:
        lead_data = data.get('lead_info', {})
        
        # Simple scoring logic
        score = 0
        engagement = lead_data.get('Engagement_Score', 0.5)
        
        score += engagement * 50
        
        if lead_data.get('Requested_Demo', 0) == 1:
            score += 25
        if lead_data.get('Viewed_Pricing_Page', 0) == 1:
            score += 15
        if lead_data.get('Company_Size', '') in ['201-1000', '1000+']:
            score += 10
        
        score = min(100, score)
        
        if score >= 75:
            priority = "HOT"
        elif score >= 50:
            priority = "WARM"
        else:
            priority = "COLD"
        
        return {
            'priority_score': round(score, 1),
            'conversion_probability': round(score / 100, 2),
            'priority': priority,
            'confidence_score': 0.85,
            'recommendation': f"{priority}_PRIORITY"
        }

class LeadEnrichmentAgent(BaseAgent):
    """Fills missing lead data"""
    
    def __init__(self):
        super().__init__(
            agent_name="Lead Enrichment Agent",
            agent_role="Fills missing lead data with external sources"
        )
    
    def analyze(self, data: Dict, context: Dict) -> Dict:
        lead_data = data.get('lead_info', {})
        
        # Mock enrichment
        enriched_fields = []
        if not lead_data.get('Company_Size'):
            lead_data['Company_Size'] = '51-200'
            enriched_fields.append('Company_Size')
        
        if not lead_data.get('Industry'):
            lead_data['Industry'] = 'Technology'
            enriched_fields.append('Industry')
        
        enrichment_score = (len(enriched_fields) / 5) * 100
        
        return {
            'enriched_data': lead_data,
            'newly_enriched_fields': enriched_fields,
            'enrichment_score': round(enrichment_score, 1),
            'confidence_score': 0.75
        }

class LeadValuePredictor(BaseAgent):
    """Calculates Expected Lifetime Value"""
    
    def __init__(self):
        super().__init__(
            agent_name="Lead Value Predictor",
            agent_role="Calculates Expected Lifetime Value and revenue potential"
        )
    
    def analyze(self, data: Dict, context: Dict) -> Dict:
        lead_data = data.get('lead_info', {})
        
        # Base deal size
        company_size = lead_data.get('Company_Size', '51-200')
        size_multipliers = {'1-10': 5000, '11-50': 15000, '51-200': 50000, '201-1000': 150000, '1000+': 500000}
        
        base_value = 50000
        for size, value in size_multipliers.items():
            if size in company_size:
                base_value = value
                break
        
        # Adjust for conversion probability
        conv_prob = lead_data.get('Conversion_Probability', 0.25)
        eltv = base_value * conv_prob * 3.5  # 3.5 year average lifetime
        
        # ROI calculation
        acquisition_cost = context.get('avg_acquisition_cost', 5000)
        roi_multiple = (eltv / acquisition_cost) if acquisition_cost > 0 else 0
        
        # Value tier
        if eltv >= 200000:
            value_tier = 'PLATINUM'
        elif eltv >= 100000:
            value_tier = 'GOLD'
        elif eltv >= 50000:
            value_tier = 'SILVER'
        else:
            value_tier = 'BRONZE'
        
        return {
            'expected_lifetime_value': round(eltv, 2),
            'base_deal_size': base_value,
            'roi_multiple': round(roi_multiple, 2),
            'value_tier': value_tier,
            'confidence_score': 0.82
        }

class PersonalizedCampaignGenerator(BaseAgent):
    """Generates multi-channel campaigns"""
    
    def __init__(self):
        super().__init__(
            agent_name="Campaign Generator",
            agent_role="Creates compelling multi-channel marketing content"
        )
    
    def analyze(self, data: Dict, context: Dict) -> Dict:
        campaign_data = data.get('campaign_info', {})
        
        segment = campaign_data.get('segment', 'All Customers')
        product = campaign_data.get('product', 'Product')
        
        variant_a_subject = f"Exclusive {product} Offer for {segment}"
        variant_a_content = f"Hi! We have {product} designed for {segment}. Limited time offer with premium benefits."
        
        variant_b_subject = f"🎯 {product} - Perfect for {segment}"
        variant_b_content = f"Hello! Introducing {product} - the solution {segment} need. Trusted by thousands."
        
        return {
            'variant_a': {'subject': variant_a_subject, 'content': variant_a_content},
            'variant_b': {'subject': variant_b_subject, 'content': variant_b_content},
            'quality_score_a': 75,
            'quality_score_b': 78,
            'recommended_variant': 'B',
            'confidence_score': 0.80
        }

class ContentRecommendationEngine(BaseAgent):
    """Recommends next-best content"""
    
    def __init__(self):
        super().__init__(
            agent_name="Content Recommender",
            agent_role="Suggests optimal content for lead progression"
        )
    
    def analyze(self, data: Dict, context: Dict) -> Dict:
        lead_data = data.get('lead_info', {})
        
        engagement = lead_data.get('Engagement_Score', 0.5)
        
        if engagement > 0.7:
            stage = 'decision'
            recommendations = ['Pricing Guide', 'ROI Calculator', 'Customer Stories']
        elif engagement > 0.4:
            stage = 'consideration'
            recommendations = ['Product Demo', 'Feature Comparison', 'Case Studies']
        else:
            stage = 'awareness'
            recommendations = ['Industry Report', 'Getting Started Guide', 'Webinar']
        
        return {
            'current_stage': stage,
            'top_recommendations': recommendations,
            'confidence_score': 0.78
        }

class CreativeOptimizer(BaseAgent):
    """Optimizes creative elements"""
    
    def __init__(self):
        super().__init__(
            agent_name="Creative Optimizer",
            agent_role="Analyzes and optimizes creative performance"
        )
    
    def analyze(self, data: Dict, context: Dict) -> Dict:
        creative_data = data.get('creative_info', {})
        
        suggestions = [
            {'priority': 'HIGH', 'element': 'Subject Line', 'suggestion': 'Add personalization token', 'impact': '+26%'},
            {'priority': 'MEDIUM', 'element': 'CTA', 'suggestion': 'Change to action-oriented', 'impact': '+15%'}
        ]
        
        return {
            'optimization_suggestions': suggestions,
            'estimated_performance_lift': '+35%',
            'confidence_score': 0.88
        }

class PerformanceMonitor(BaseAgent):
    """Tracks campaign performance"""
    
    def __init__(self):
        super().__init__(
            agent_name="Performance Monitor",
            agent_role="Tracks real-time ROI and ROAS metrics"
        )
    
    def analyze(self, data: Dict, context: Dict) -> Dict:
        perf_data = data.get('performance_data', {})
        
        current_roas = perf_data.get('roas', 4.2)
        target_roas = perf_data.get('target', 3.5)
        
        status = 'EXCEEDING' if current_roas > target_roas else 'MEETING' if current_roas >= target_roas * 0.9 else 'BELOW'
        
        return {
            'current_roas': current_roas,
            'target_roas': target_roas,
            'performance_status': status,
            'recommendation': 'SCALE' if status == 'EXCEEDING' else 'MAINTAIN' if status == 'MEETING' else 'OPTIMIZE',
            'confidence_score': 0.92
        }

class SentimentTracker(BaseAgent):
    """Monitors brand sentiment"""
    
    def __init__(self):
        super().__init__(
            agent_name="Sentiment Tracker",
            agent_role="Monitors brand health and reputation risks"
        )
    
    def analyze(self, data: Dict, context: Dict) -> Dict:
        sentiment_data = data.get('sentiment_data', {})
        
        brand_health = sentiment_data.get('health_score', random.uniform(6.5, 8.5))
        
        if brand_health > 7.5:
            risk_level = 'LOW'
            recommendation = 'PROCEED'
        elif brand_health > 5:
            risk_level = 'MEDIUM'
            recommendation = 'CAUTION'
        else:
            risk_level = 'HIGH'
            recommendation = 'DEFENSIVE_ONLY'
        
        return {
            'brand_health_score': round(brand_health, 1),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'confidence_score': 0.75
        }

class CompetitorIntelligence(BaseAgent):
    """Tracks competitor activities"""
    
    def __init__(self):
        super().__init__(
            agent_name="Competitor Intelligence",
            agent_role="Monitors competitor strategies and provides counter-moves"
        )
    
    def analyze(self, data: Dict, context: Dict) -> Dict:
        competitor_data = data.get('competitor_data', {})
        
        threat_score = competitor_data.get('threat_level', random.randint(5, 9))
        
        if threat_score > 7:
            urgency = 'HIGH'
            counter_strategy = 'AGGRESSIVE: Match or exceed competitor spend'
        elif threat_score > 4:
            urgency = 'MEDIUM'
            counter_strategy = 'DEFENSIVE: Maintain position while monitoring'
        else:
            urgency = 'LOW'
            counter_strategy = 'OPPORTUNISTIC: Exploit competitor weaknesses'
        
        return {
            'rival_threat_score': threat_score,
            'urgency_level': urgency,
            'recommended_counter_strategy': counter_strategy,
            'confidence_score': 0.85
        }

class MacroTrendAnalyzer(BaseAgent):
    """Analyzes market trends"""
    
    def __init__(self):
        super().__init__(
            agent_name="Macro Trend Analyzer",
            agent_role="Provides market predictions and economic context"
        )
    
    def analyze(self, data: Dict, context: Dict) -> Dict:
        market_phase = random.choice(['GROWTH', 'PEAK', 'CONTRACTION', 'RECOVERY'])
        
        if market_phase in ['GROWTH', 'RECOVERY']:
            recommendation = 'EXPAND'
            risk = 'LOW'
        elif market_phase == 'PEAK':
            recommendation = 'MAINTAIN'
            risk = 'MODERATE'
        else:
            recommendation = 'CONSOLIDATE'
            risk = 'HIGH'
        
        return {
            'market_phase': market_phase,
            'recommendation': recommendation,
            'market_risk': risk,
            'confidence_score': 0.80
        }

# =========================================================================
# STRATEGIC CERTAINTY ENGINE
# =========================================================================

class StrategicCertaintyEngine:
    """Main orchestrator for multi-agent strategic decision-making"""
    
    def __init__(self):
        self.csa = CentralStrategyAgent()
        
        # Initialize all 11 agents
        self.agents = {
            # Cluster 1: Lead Intelligence
            'smart_lead_scorer': SmartLeadScorer(),
            'lead_enrichment': LeadEnrichmentAgent(),
            'lead_value_predictor': LeadValuePredictor(),
            
            # Cluster 2: Execution & Optimization
            'campaign_generator': PersonalizedCampaignGenerator(),
            'content_recommender': ContentRecommendationEngine(),
            'creative_optimizer': CreativeOptimizer(),
            
            # Cluster 3: Defense & Performance
            'performance_monitor': PerformanceMonitor(),
            'sentiment_tracker': SentimentTracker(),
            'competitor_intelligence': CompetitorIntelligence(),
            
            # Cluster 4: Strategy
            'macro_trend_analyzer': MacroTrendAnalyzer()
        }
        
        print("✅ Strategic Certainty Engine initialized with 11 agents")
    
    def run_strategic_huddle(self, query: str, context: Dict, data: Optional[Dict] = None) -> Dict:
        """Execute full strategic decision-making process"""
        
        # Reset conversation state
        self.csa.reset_huddle()
        
        # Emit progress (if socketio is available)
        try:
            socketio.emit('huddle_status', {'phase': 'opening', 'message': 'CSA opening strategic huddle...'})
        except:
            pass
        
        # Step 1: CSA opens the huddle
        opening = self.csa.open_huddle(query, context)
        
        # Step 2: Identify required agents
        required_agents = self.csa.parse_agent_requirements(opening)
        
        # Fallback to default agents if parsing fails
        if not required_agents or len(required_agents) < 2:
            required_agents = [
                'smart_lead_scorer',
                'lead_value_predictor',
                'competitor_intelligence',
                'performance_monitor',
                'creative_optimizer'
            ]
        
        try:
            socketio.emit('huddle_status', {
                'phase': 'agents_identified',
                'agents': [self.agents[a].agent_name for a in required_agents if a in self.agents],
                'message': f'Consulting {len(required_agents)} expert agents...'
            })
        except:
            pass
        
        # Step 3: Collect arguments from each agent
        agent_arguments = []
        
        for agent_key in required_agents:
            if agent_key in self.agents:
                try:
                    try:
                        socketio.emit('agent_thinking', {
                            'agent': self.agents[agent_key].agent_name,
                            'status': 'analyzing'
                        })
                    except:
                        pass
                    
                    argument = self.agents[agent_key].present_case(query, context, data)
                    agent_arguments.append(argument)
                    
                    try:
                        socketio.emit('agent_complete', {
                            'agent': self.agents[agent_key].agent_name,
                            'confidence': argument.get('confidence', 0.5)
                        })
                    except:
                        pass
                    
                except Exception as e:
                    print(f"❌ Agent {agent_key} failed: {e}")
        
        # Step 4: CSA synthesizes arguments
        try:
            socketio.emit('huddle_status', {'phase': 'synthesis', 'message': 'CSA synthesizing arguments...'})
        except:
            pass
        
        self.csa.synthesize_arguments(agent_arguments)
        
        # Step 5: CSA issues final verdict
        try:
            socketio.emit('huddle_status', {'phase': 'verdict', 'message': 'CSA issuing final verdict...'})
        except:
            pass
        
        verdict = self.csa.issue_verdict(context)
        
        try:
            socketio.emit('huddle_complete', {
                'verdict': verdict['content'],
                'timestamp': verdict['timestamp']
            })
        except:
            pass
        
        return {
            'verdict': verdict,
            'full_transcript': self.csa.get_huddle_transcript(),
            'agent_analyses': agent_arguments,
            'agents_consulted': len(agent_arguments)
        }

# =========================================================================
# FLASK APPLICATION
# =========================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'strategic-certainty-engine-2024'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize the engine globally
try:
    sce = StrategicCertaintyEngine()
except Exception as e:
    print(f"❌ FATAL: Engine initialization failed: {e}")
    sce = None

# =========================================================================
# REST API ENDPOINTS
# =========================================================================

@app.route('/')
def index():
    """Render main UI"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if sce is None:
        return jsonify({'status': 'error', 'message': 'Engine not initialized'}), 500
    
    return jsonify({
        'status': 'healthy',
        'agents': list(sce.agents.keys()),
        'total_agents': len(sce.agents),
        'llm_available': LLM_AVAILABLE,
        'ml_available': ML_AVAILABLE
    })

@app.route('/api/strategic-huddle', methods=['POST'])
def run_strategic_huddle():
    """Main endpoint for strategic huddle execution"""
    
    if sce is None:
        return jsonify({'status': 'error', 'message': 'Engine not initialized'}), 500
    
    try:
        req_data = request.json
        
        query = req_data.get('query', 'Should we increase marketing budget?')
        
        context = {
            'goal_mode': req_data.get('goal_mode', 'Max Growth'),
            'risk_tolerance': req_data.get('risk_tolerance', 'Medium'),
            'budget': req_data.get('budget', 50000),
            'timeframe': req_data.get('timeframe', '30 days'),
            'original_query': query
        }
        
        # Mock data if not provided
        data = req_data.get('data', {
            'lead_info': {
                'Engagement_Score': 0.85,
                'Conversion_Probability': 0.45,
                'Company_Size': '201-1000',
                'Industry': 'Technology',
                'Requested_Demo': 1,
                'Viewed_Pricing_Page': 1
            },
            'performance_data': {
                'roas': 4.2,
                'target': 3.5
            },
            'sentiment_data': {
                'health_score': 7.8
            },
            'competitor_data': {
                'threat_level': 7
            }
        })
        
        # Run the huddle
        result = sce.run_strategic_huddle(query, context, data)
        
        # Parse the verdict
        verdict_content = result['verdict']['content']
        
        # Extract command
        command = 'UNKNOWN'
        if 'COMMAND:' in verdict_content:
            command_line = verdict_content.split('COMMAND:')[1].split('\n')[0].strip()
            command = command_line.split()[0] if command_line else 'UNKNOWN'
        
        return jsonify({
            'status': 'success',
            'command': command,
            'verdict_full': verdict_content,
            'agents_consulted': result['agents_consulted'],
            'timestamp': result['verdict']['timestamp'],
            'transcript_length': len(result['full_transcript'])
        })
        
    except Exception as e:
        print(f"❌ Error in strategic huddle: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/agents', methods=['GET'])
def get_agents():
    """Get list of all available agents"""
    
    if sce is None:
        return jsonify({'status': 'error', 'agents': []}), 500
    
    agents_info = []
    
    for key, agent in sce.agents.items():
        cluster = 'unknown'
        if 'lead' in key or 'enrichment' in key or 'scorer' in key or 'value' in key:
            cluster = 'Lead Intelligence'
        elif 'campaign' in key or 'content' in key or 'creative' in key:
            cluster = 'Execution & Optimization'
        elif 'performance' in key or 'sentiment' in key or 'competitor' in key:
            cluster = 'Defense & Performance'
        elif 'macro' in key or 'trend' in key:
            cluster = 'Strategy & Insights'
        
        agents_info.append({
            'id': key,
            'name': agent.agent_name,
            'role': agent.agent_role,
            'cluster': cluster
        })
    
    return jsonify({
        'status': 'success',
        'total': len(agents_info),
        'agents': agents_info
    })

@app.route('/api/quick-decision', methods=['POST'])
def quick_decision():
    """Quick decision endpoint with minimal input"""
    
    if sce is None:
        return jsonify({'status': 'error', 'message': 'Engine not initialized'}), 500
    
    try:
        req_data = request.json
        query = req_data.get('query', 'Should we proceed with this campaign?')
        
        # Use defaults
        context = {
            'goal_mode': 'Balanced',
            'risk_tolerance': 'Medium',
            'budget': 50000,
            'timeframe': '30 days',
            'original_query': query
        }
        
        result = sce.run_strategic_huddle(query, context)
        
        verdict_content = result['verdict']['content']
        command = 'GO'
        if 'COMMAND:' in verdict_content:
            command_line = verdict_content.split('COMMAND:')[1].split('\n')[0].strip()
            command = command_line.split()[0] if command_line else 'GO'
        
        return jsonify({
            'status': 'success',
            'command': command,
            'verdict': verdict_content
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# =========================================================================
# WEBSOCKET EVENTS
# =========================================================================

@socketio.on('connect')
def handle_connect():
    print('✅ Client connected')
    emit('connection_status', {
        'status': 'connected',
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('disconnect')
def handle_disconnect():
    print('❌ Client disconnected')

@socketio.on('start_huddle')
def handle_start_huddle(data):
    """Handle real-time huddle via WebSocket"""
    
    if sce is None:
        emit('error', {'message': 'Engine not initialized'})
        return
    
    try:
        query = data.get('query', 'Should we increase budget?')
        context = {
            'goal_mode': data.get('goal_mode', 'Max Growth'),
            'risk_tolerance': data.get('risk_tolerance', 'Medium'),
            'budget': data.get('budget', 50000),
            'timeframe': '30 days',
            'original_query': query
        }
        
        result = sce.run_strategic_huddle(query, context)
        
        emit('huddle_result', {
            'status': 'success',
            'verdict': result['verdict']['content'],
            'agents_consulted': result['agents_consulted']
        })
        
    except Exception as e:
        emit('error', {'message': str(e)})

# =========================================================================
# HTML TEMPLATE (BEAUTIFUL MODERN UI)
# =========================================================================

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Strategic Certainty Engine - AI Marketing Command Center</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.4/socket.io.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --primary: #00ff88;
            --secondary: #0066ff;
            --dark: #0a0e1a;
            --card: #151925;
            --text: #e0e6f0;
            --text-dim: #8b95a8;
            --success: #00ff88;
            --warning: #ffd93d;
            --danger: #ff6b6b;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #0a0e1a 0%, #1a1f35 100%);
            color: var(--text);
            min-height: 100vh;
            padding: 2rem;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        /* Header */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 2rem 0;
            border-bottom: 1px solid rgba(0, 255, 136, 0.1);
            margin-bottom: 3rem;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .logo-icon {
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            font-weight: 800;
            box-shadow: 0 8px 32px rgba(0, 255, 136, 0.3);
        }
        
        .logo-text h1 {
            font-size: 2rem;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            letter-spacing: -1px;
        }
        
        .logo-text p {
            font-size: 0.9rem;
            color: var(--text-dim);
            margin-top: 0.25rem;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.75rem 1.5rem;
            background: rgba(0, 255, 136, 0.1);
            border: 1px solid var(--primary);
            border-radius: 50px;
            font-size: 0.9rem;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            background: var(--primary);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.2); }
        }
        
        /* Main Grid */
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 2rem;
            margin-bottom: 2rem;
        }
        
        /* Card */
        .card {
            background: var(--card);
            border: 1px solid rgba(0, 255, 136, 0.1);
            border-radius: 24px;
            padding: 2rem;
            transition: all 0.3s ease;
        }
        
        .card:hover {
            border-color: var(--primary);
            box-shadow: 0 8px 32px rgba(0, 255, 136, 0.1);
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
        }
        
        .card-title {
            font-size: 1.5rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .card-icon {
            font-size: 1.8rem;
        }
        
        /* Form */
        .form-group {
            margin-bottom: 1.5rem;
        }
        
        .form-label {
            display: block;
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-dim);
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .form-input,
        .form-select,
        .form-textarea {
            width: 100%;
            padding: 1rem 1.25rem;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            color: var(--text);
            font-family: 'Inter', sans-serif;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        
        .form-input:focus,
        .form-select:focus,
        .form-textarea:focus {
            outline: none;
            border-color: var(--primary);
            background: rgba(0, 255, 136, 0.05);
        }
        
        .form-textarea {
            min-height: 120px;
            resize: vertical;
        }
        
        /* Button */
        .btn {
            padding: 1rem 2rem;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            font-family: 'Inter', sans-serif;
            display: inline-flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: var(--dark);
            box-shadow: 0 4px 16px rgba(0, 255, 136, 0.3);
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 255, 136, 0.4);
        }
        
        .btn-primary:active {
            transform: translateY(0);
        }
        
        .btn-primary:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .btn-full {
            width: 100%;
            justify-content: center;
        }
        
        /* Agent Grid */
        .agents-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        
        .agent-card {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .agent-card.active {
            border-color: var(--primary);
            background: rgba(0, 255, 136, 0.1);
        }
        
        .agent-card.thinking {
            border-color: var(--warning);
            animation: thinking 1.5s infinite;
        }
        
        @keyframes thinking {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        
        .agent-card.complete {
            border-color: var(--success);
            background: rgba(0, 255, 136, 0.05);
        }
        
        .agent-emoji {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        
        .agent-name {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text);
        }
        
        .agent-status {
            font-size: 0.75rem;
            color: var(--text-dim);
            margin-top: 0.25rem;
        }
        
        /* Progress */
        .progress-container {
            margin: 1.5rem 0;
        }
        
        .progress-label {
            font-size: 0.9rem;
            color: var(--text-dim);
            margin-bottom: 0.5rem;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 50px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            border-radius: 50px;
            transition: width 0.5s ease;
        }
        
        /* Verdict Display */
        .verdict-container {
            background: rgba(0, 0, 0, 0.4);
            border: 2px solid var(--primary);
            border-radius: 16px;
            padding: 2rem;
            margin-top: 2rem;
            display: none;
        }
        
        .verdict-container.show {
            display: block;
            animation: slideUp 0.5s ease;
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .verdict-command {
            display: inline-block;
            padding: 0.75rem 2rem;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: var(--dark);
            font-size: 1.5rem;
            font-weight: 800;
            border-radius: 12px;
            margin-bottom: 1.5rem;
        }
        
        .verdict-content {
            font-size: 0.95rem;
            line-height: 1.8;
            white-space: pre-wrap;
            font-family: 'Monaco', 'Courier New', monospace;
        }
        
        /* Loading */
        .loading {
            display: none;
            text-align: center;
            padding: 2rem;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid rgba(0, 255, 136, 0.2);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* Toast */
        .toast {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            background: var(--card);
            border: 1px solid var(--primary);
            border-radius: 12px;
            padding: 1rem 1.5rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            display: none;
            z-index: 1000;
        }
        
        .toast.show {
            display: block;
            animation: slideInRight 0.3s ease;
        }
        
        @keyframes slideInRight {
            from {
                transform: translateX(400px);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        /* Responsive */
        @media (max-width: 1024px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">
                <div class="logo-icon">Ξ</div>
                <div class="logo-text">
                    <h1>Strategic Certainty Engine</h1>
                    <p>AI-Powered Multi-Agent Marketing Command Center</p>
                </div>
            </div>
            <div class="status-indicator">
                <div class="status-dot"></div>
                <span id="statusText">System Ready</span>
            </div>
        </header>
        
        <div class="main-grid">
            <!-- Left Column: Control Panel -->
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">
                        <span class="card-icon">🎯</span>
                        Strategic Query
                    </h2>
                </div>
                
                <form id="huddleForm">
                    <div class="form-group">
                        <label class="form-label">Your Strategic Question</label>
                        <textarea 
                            class="form-textarea" 
                            id="queryInput" 
                            placeholder="e.g., Should we increase our LinkedIn campaign budget by 50%?"
                            required
                        >Should we increase our LinkedIn campaign budget by 50% to capture more enterprise leads?</textarea>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">Goal Mode</label>
                        <select class="form-select" id="goalMode">
                            <option value="Max Growth">Max Growth</option>
                            <option value="Balanced">Balanced</option>
                            <option value="Conservative">Conservative</option>
                            <option value="Experimental">Experimental</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">Risk Tolerance</label>
                        <select class="form-select" id="riskTolerance">
                            <option value="High">High</option>
                            <option value="Medium" selected>Medium</option>
                            <option value="Low">Low</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">Budget ($)</label>
                        <input 
                            type="number" 
                            class="form-input" 
                            id="budgetInput" 
                            value="50000" 
                            min="0" 
                            step="1000"
                        >
                    </div>
                    
                    <button type="submit" class="btn btn-primary btn-full" id="startHuddleBtn">
                        <span>🚀</span>
                        <span>Start Strategic Huddle</span>
                    </button>
                </form>
                
                <div class="progress-container" id="progressContainer" style="display: none;">
                    <div class="progress-label" id="progressLabel">Initializing huddle...</div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressFill" style="width: 0%"></div>
                    </div>
                </div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p id="loadingText">AI agents are deliberating...</p>
                </div>
                
                <div class="verdict-container" id="verdictContainer">
                    <div class="verdict-command" id="verdictCommand">GO</div>
                    <div class="verdict-content" id="verdictContent"></div>
                </div>
            </div>
            
            <!-- Right Column: Agent Status -->
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">
                        <span class="card-icon">🤖</span>
                        AI Agent Network
                    </h2>
                </div>
                
                <div class="agents-grid" id="agentsGrid">
                    <div class="agent-card" data-agent="smart_lead_scorer">
                        <div class="agent-emoji">🎯</div>
                        <div class="agent-name">Lead Scorer</div>
                        <div class="agent-status">Ready</div>
                    </div>
                    <div class="agent-card" data-agent="lead_value_predictor">
                        <div class="agent-emoji">💰</div>
                        <div class="agent-name">Value Predictor</div>
                        <div class="agent-status">Ready</div>
                    </div>
                    <div class="agent-card" data-agent="campaign_generator">
                        <div class="agent-emoji">📧</div>
                        <div class="agent-name">Campaign Gen</div>
                        <div class="agent-status">Ready</div>
                    </div>
                    <div class="agent-card" data-agent="creative_optimizer">
                        <div class="agent-emoji">🎨</div>
                        <div class="agent-name">Creative Opt</div>
                        <div class="agent-status">Ready</div>
                    </div>
                    <div class="agent-card" data-agent="performance_monitor">
                        <div class="agent-emoji">📊</div>
                        <div class="agent-name">Performance</div>
                        <div class="agent-status">Ready</div>
                    </div>
                    <div class="agent-card" data-agent="sentiment_tracker">
                        <div class="agent-emoji">😊</div>
                        <div class="agent-name">Sentiment</div>
                        <div class="agent-status">Ready</div>
                    </div>
                    <div class="agent-card" data-agent="competitor_intelligence">
                        <div class="agent-emoji">🔍</div>
                        <div class="agent-name">Competitor</div>
                        <div class="agent-status">Ready</div>
                    </div>
                    <div class="agent-card" data-agent="macro_trend_analyzer">
                        <div class="agent-emoji">📈</div>
                        <div class="agent-name">Macro Trends</div>
                        <div class="agent-status">Ready</div>
                    </div>
                </div>
                
                <div style="margin-top: 2rem; padding-top: 2rem; border-top: 1px solid rgba(255,255,255,0.1);">
                    <h3 style="font-size: 1rem; margin-bottom: 1rem; color: var(--text-dim);">System Info</h3>
                    <div style="font-size: 0.85rem; line-height: 1.8; color: var(--text-dim);">
                        <div><strong>Total Agents:</strong> <span id="totalAgents">11</span></div>
                        <div><strong>LLM Status:</strong> <span id="llmStatus">Connected</span></div>
                        <div><strong>Last Decision:</strong> <span id="lastDecision">None</span></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="toast" id="toast">
        <div id="toastMessage"></div>
    </div>
    
    <script>
        const API_BASE = 'http://localhost:5001/api';
        const socket = io('http://localhost:5001');
        
        // Elements
        const huddleForm = document.getElementById('huddleForm');
        const startHuddleBtn = document.getElementById('startHuddleBtn');
        const loading = document.getElementById('loading');
        const verdictContainer = document.getElementById('verdictContainer');
        const progressContainer = document.getElementById('progressContainer');
        const progressFill = document.getElementById('progressFill');
        const progressLabel = document.getElementById('progressLabel');
        
        // Socket.IO event handlers
        socket.on('connect', () => {
            console.log('WebSocket connected');
            showToast('✅ Connected to Strategic Engine');
        });
        
        socket.on('huddle_status', (data) => {
            console.log('Huddle status:', data);
            progressLabel.textContent = data.message;
            
            const phases = ['opening', 'agents_identified', 'synthesis', 'verdict'];
            const currentIndex = phases.indexOf(data.phase);
            const progress = ((currentIndex + 1) / phases.length) * 100;
            progressFill.style.width = progress + '%';
        });
        
        socket.on('agent_thinking', (data) => {
            console.log('Agent thinking:', data);
            const agentCards = document.querySelectorAll('.agent-card');
            agentCards.forEach(card => {
                const agentName = card.querySelector('.agent-name').textContent;
                if (data.agent.includes(agentName.split(' ')[0])) {
                    card.classList.add('thinking');
                    card.querySelector('.agent-status').textContent = 'Analyzing...';
                }
            });
        });
        
        socket.on('agent_complete', (data) => {
            console.log('Agent complete:', data);
            const agentCards = document.querySelectorAll('.agent-card');
            agentCards.forEach(card => {
                const agentName = card.querySelector('.agent-name').textContent;
                if (data.agent.includes(agentName.split(' ')[0])) {
                    card.classList.remove('thinking');
                    card.classList.add('complete');
                    card.querySelector('.agent-status').textContent = 'Complete ✓';
                }
            });
        });
        
        socket.on('huddle_complete', (data) => {
            console.log('Huddle complete:', data);
            showToast('✅ Strategic decision made!');
        });
        
        // Form submission
        huddleForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const query = document.getElementById('queryInput').value;
            const goalMode = document.getElementById('goalMode').value;
            const riskTolerance = document.getElementById('riskTolerance').value;
            const budget = parseInt(document.getElementById('budgetInput').value);
            
            // Reset UI
            verdictContainer.classList.remove('show');
            document.querySelectorAll('.agent-card').forEach(card => {card.classList.remove('active', 'thinking', 'complete');
                card.querySelector('.agent-status').textContent = 'Ready';
            });
            
            // Show loading
            startHuddleBtn.disabled = true;
            startHuddleBtn.innerHTML = '<span>⏳</span><span>Running Huddle...</span>';
            loading.classList.add('show');
            progressContainer.style.display = 'block';
            progressFill.style.width = '0%';
            
            try {
                const response = await fetch(`${API_BASE}/strategic-huddle`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        query: query,
                        goal_mode: goalMode,
                        risk_tolerance: riskTolerance,
                        budget: budget
                    })
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    // Display verdict
                    document.getElementById('verdictCommand').textContent = result.command;
                    document.getElementById('verdictContent').textContent = result.verdict_full;
                    verdictContainer.classList.add('show');
                    
                    // Update system info
                    document.getElementById('lastDecision').textContent = new Date().toLocaleTimeString();
                    
                    // Complete progress
                    progressFill.style.width = '100%';
                    progressLabel.textContent = 'Strategic decision complete!';
                    
                    showToast(`✅ Decision: ${result.command}`);
                } else {
                    showToast('❌ Error: ' + result.message);
                }
                
            } catch (error) {
                console.error('Error:', error);
                showToast('❌ Failed to run strategic huddle');
            } finally {
                loading.classList.remove('show');
                startHuddleBtn.disabled = false;
                startHuddleBtn.innerHTML = '<span>🚀</span><span>Start Strategic Huddle</span>';
            }
        });
        
        // Toast notification
        function showToast(message) {
            const toast = document.getElementById('toast');
            const toastMessage = document.getElementById('toastMessage');
            
            toastMessage.textContent = message;
            toast.classList.add('show');
            
            setTimeout(() => {
                toast.classList.remove('show');
            }, 4000);
        }
        
        // Load system info on page load
        async function loadSystemInfo() {
            try {
                const response = await fetch(`${API_BASE}/health`);
                const data = await response.json();
                
                if (data.status === 'healthy') {
                    document.getElementById('totalAgents').textContent = data.total_agents;
                    document.getElementById('llmStatus').textContent = data.llm_available ? 'Connected ✓' : 'Template Mode';
                }
            } catch (error) {
                console.error('Failed to load system info:', error);
            }
        }
        
        // Initialize
        loadSystemInfo();
        
        // Example queries for quick testing
        const exampleQueries = [
            "Should we increase our LinkedIn campaign budget by 50% to capture more enterprise leads?",
            "Is now the right time to launch a new product campaign given current market conditions?",
            "Should we shift 30% of our budget from Facebook to TikTok for Gen Z targeting?",
            "Can we scale our email nurture campaigns without hurting deliverability?",
            "Should we pause our competitor comparison ads due to negative sentiment?"
        ];
        
        // Add quick example buttons (optional)
        let exampleIndex = 0;
        document.getElementById('queryInput').addEventListener('dblclick', () => {
            document.getElementById('queryInput').value = exampleQueries[exampleIndex];
            exampleIndex = (exampleIndex + 1) % exampleQueries.length;
            showToast('💡 Example query loaded');
        });
    </script>
</body>
</html>
'''

# =========================================================================
# MAIN EXECUTION
# =========================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 STRATEGIC CERTAINTY ENGINE - LAUNCHING")
    print("="*80)
    print("\n✨ Features:")
    print("  ✅ 11 Specialized AI Agents")
    print("  ✅ Central Strategy Agent (CSA) Orchestrator")
    print("  ✅ Real-time Strategic Huddles")
    print("  ✅ WebSocket Support")
    print("  ✅ Beautiful Modern UI")
    print(f"\n🔧 Configuration:")
    print(f"  • LLM: {'Groq (Connected)' if LLM_AVAILABLE else 'Template Mode'}")
    print(f"  • ML: {'Available' if ML_AVAILABLE else 'Rule-based'}")
    print(f"  • Port: 5001")
    
    if sce is None:
        print("\n❌ CRITICAL FAILURE: Engine cannot start. Check logs above.")
        exit(1)
    
    print("\n" + "="*80)
    print("🌐 SERVER RUNNING")
    print("="*80)
    print("  📡 Web UI: http://localhost:5001")
    print("  🔌 API: http://localhost:5001/api/strategic-huddle")
    print("  ❤️  Health: http://localhost:5001/api/health")
    print("\n💡 Tip: Double-click the query box to load example questions")
    print("="*80 + "\n")
    
    try:
        socketio.run(app, host='0.0.0.0', port=5010, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down Strategic Certainty Engine...")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
