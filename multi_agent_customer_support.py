"""
Multi-Agent Customer Support System with Agent Garden AI
Complete Implementation - Production Ready for Medium
=====================================================

This implementation demonstrates:
- Think-Act-Observe loop for each agent
- Parallel multi-agent execution
- Specialized agents for different domains
- Response coordination
- Memory management
- Comprehensive logging

Author: [Your Name]
Date: November 2025
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==========================================
# 1. DATA STRUCTURES & TYPES
# ==========================================

class ToolCategory(Enum):
    """Tool categories for organization and filtering"""
    DATABASE = "database"
    SEARCH = "search"
    COMPUTATION = "computation"
    EXTERNAL_API = "external_api"


@dataclass
class AgentRole:
    """Define an agent's role and responsibilities"""
    name: str
    goal: str
    expertise: str
    tools: List[str]


@dataclass
class Tool:
    """Tool definition with metadata and callable function"""
    name: str
    description: str
    category: ToolCategory
    function: Callable
    required_params: List[str]
    optional_params: List[str] = None
    
    def __post_init__(self):
        if self.optional_params is None:
            self.optional_params = []


# ==========================================
# 2. TOOL IMPLEMENTATIONS
# ==========================================

class ToolLibrary:
    """Registry and implementation of all available tools"""
    
    @staticmethod
    def check_invoice(customer_id: str, invoice_id: str) -> Dict[str, Any]:
        """Retrieve invoice details from billing system."""
        logger.debug(f"Checking invoice {invoice_id} for customer {customer_id}")
        return {
            "invoice_id": invoice_id,
            "customer_id": customer_id,
            "amount": 149.99,
            "date": "2025-11-01",
            "status": "paid",
            "items": ["Product A", "Service B"],
            "tax": 12.00,
            "total": 161.99,
            "source": "Billing System"
        }
    
    @staticmethod
    def get_payment_history(customer_id: str, months: int = 12) -> List[Dict[str, Any]]:
        """Get customer payment history for specified period."""
        logger.debug(f"Fetching {months} months of payment history for {customer_id}")
        return [
            {"date": "2025-11-01", "amount": 149.99, "status": "paid", "method": "credit_card"},
            {"date": "2025-10-01", "amount": 89.99, "status": "paid", "method": "credit_card"},
            {"date": "2025-09-01", "amount": 149.99, "status": "paid", "method": "bank_transfer"},
            {"date": "2025-08-01", "amount": 99.99, "status": "paid", "method": "credit_card"},
        ]
    
    @staticmethod
    def process_refund(customer_id: str, amount: float, reason: str) -> Dict[str, Any]:
        """Process a refund for customer."""
        logger.info(f"Processing ${amount} refund for {customer_id}. Reason: {reason}")
        return {
            "refund_id": f"REF_{customer_id}_{int(datetime.now().timestamp())}",
            "amount": amount,
            "status": "processing",
            "estimated_completion": "3-5 business days",
            "reason": reason,
            "processed_at": datetime.now().isoformat()
        }
    
    @staticmethod
    def search_knowledge_base(query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search technical knowledge base using semantic search."""
        logger.debug(f"Searching knowledge base for: {query}")
        
        # Simulate KB search results based on query
        kb_articles = {
            "password": {
                "title": "How to reset your password",
                "relevance_score": 0.95,
                "url": "https://kb.example.com/password-reset",
                "category": "Account Management"
            },
            "login": {
                "title": "Common login issues and solutions",
                "relevance_score": 0.87,
                "url": "https://kb.example.com/login-issues",
                "category": "Troubleshooting"
            },
            "billing": {
                "title": "Understanding your billing",
                "relevance_score": 0.92,
                "url": "https://kb.example.com/billing",
                "category": "Billing"
            },
            "refund": {
                "title": "Refund policy and process",
                "relevance_score": 0.88,
                "url": "https://kb.example.com/refunds",
                "category": "Billing"
            },
        }
        
        results = []
        query_lower = query.lower()
        for keyword, article in kb_articles.items():
            if keyword in query_lower:
                results.append(article)
        
        return results if results else list(kb_articles.values())[:limit]
    
    @staticmethod
    def check_system_logs(customer_id: str, hours: int = 24) -> Dict[str, Any]:
        """Check system logs for customer's account activity."""
        logger.debug(f"Checking system logs for {customer_id} (last {hours} hours)")
        return {
            "customer_id": customer_id,
            "period_hours": hours,
            "login_attempts": 5,
            "failed_logins": 2,
            "last_activity": datetime.now().isoformat(),
            "status": "account_locked_after_failed_attempts",
            "recommendation": "Reset password and enable 2FA",
            "source": "System Logs"
        }
    
    @staticmethod
    def get_order_status(customer_id: str, order_id: str) -> Dict[str, Any]:
        """Get order fulfillment status and tracking information."""
        logger.debug(f"Fetching order status for {order_id}")
        return {
            "order_id": order_id,
            "customer_id": customer_id,
            "status": "shipped",
            "tracking_number": f"1Z999AA{uuid.uuid4().hex[:10].upper()}",
            "carrier": "UPS",
            "estimated_delivery": "2025-11-16",
            "items": 2,
            "total_amount": 199.99,
            "source": "Fulfillment System"
        }
    
    @staticmethod
    def create_support_ticket(customer_id: str, issue: str, priority: str = "medium") -> Dict[str, Any]:
        """Create a support ticket for escalation."""
        logger.info(f"Creating {priority} priority ticket for {customer_id}")
        return {
            "ticket_id": f"TICKET_{int(datetime.now().timestamp())}",
            "customer_id": customer_id,
            "issue": issue,
            "priority": priority,
            "status": "open",
            "created_at": datetime.now().isoformat(),
            "assigned_to": "support_team",
            "source": "Ticketing System"
        }


def create_tool_registry() -> Dict[str, Tool]:
    """Create and return the tool registry with all available tools"""
    
    return {
        "check_invoice": Tool(
            name="check_invoice",
            description="Look up specific invoice details",
            category=ToolCategory.DATABASE,
            function=ToolLibrary.check_invoice,
            required_params=["customer_id", "invoice_id"]
        ),
        "get_payment_history": Tool(
            name="get_payment_history",
            description="Get customer payment history",
            category=ToolCategory.DATABASE,
            function=ToolLibrary.get_payment_history,
            required_params=["customer_id"],
            optional_params=["months"]
        ),
        "process_refund": Tool(
            name="process_refund",
            description="Process a refund for customer",
            category=ToolCategory.DATABASE,
            function=ToolLibrary.process_refund,
            required_params=["customer_id", "amount", "reason"]
        ),
        "search_knowledge_base": Tool(
            name="search_knowledge_base",
            description="Search technical knowledge base",
            category=ToolCategory.SEARCH,
            function=ToolLibrary.search_knowledge_base,
            required_params=["query"],
            optional_params=["limit"]
        ),
        "check_system_logs": Tool(
            name="check_system_logs",
            description="Check system logs for account issues",
            category=ToolCategory.DATABASE,
            function=ToolLibrary.check_system_logs,
            required_params=["customer_id"],
            optional_params=["hours"]
        ),
        "get_order_status": Tool(
            name="get_order_status",
            description="Get order fulfillment status",
            category=ToolCategory.DATABASE,
            function=ToolLibrary.get_order_status,
            required_params=["customer_id", "order_id"]
        ),
        "create_support_ticket": Tool(
            name="create_support_ticket",
            description="Create a support ticket",
            category=ToolCategory.DATABASE,
            function=ToolLibrary.create_support_ticket,
            required_params=["customer_id", "issue"],
            optional_params=["priority"]
        ),
    }


# ==========================================
# 3. AGENT IMPLEMENTATION
# ==========================================

class AdaptiveAgent:
    """Base adaptive agent with Think-Act-Observe reasoning loop"""
    
    def __init__(self, role: AgentRole, available_tools: Dict[str, Tool], agent_id: str = None):
        """Initialize agent with role and tools"""
        self.role = role
        self.tools = {t: available_tools[t] for t in role.tools if t in available_tools}
        self.agent_id = agent_id or role.name.lower().replace(" ", "_")
        
        # Memory systems
        self.memory = {
            "short_term": [],      # Current conversation context
            "long_term": [],       # Past learnings and outcomes
            "shared_context": {}   # Inter-agent communication
        }
        
        self.execution_log = []
        self.interaction_count = 0
        logger.info(f"Initialized agent: {self.agent_id} with tools: {list(self.tools.keys())}")
    
    def think(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reasoning phase: Analyze task and plan actions
        This is where the agent decides which tools to use
        """
        logger.info(f"[{self.agent_id}] 🧠 THINK phase - Analyzing: {task[:80]}...")
        
        # Create thinking output
        thinking_output = {
            "agent": self.agent_id,
            "task": task,
            "reasoning": f"As {self.role.name}, I will {self.role.goal}. Available tools: {', '.join(self.tools.keys())}",
            "planned_actions": self._plan_actions(task),
            "confidence": 0.85,
            "timestamp": datetime.now().isoformat()
        }
        
        return thinking_output
    
    def _plan_actions(self, task: str) -> List[Dict[str, Any]]:
        """
        Plan which tools to call based on the task
        Uses keyword matching and context analysis
        """
        actions = []
        task_lower = task.lower()
        
        # Analyze task and find relevant tools
        for tool_name in self.tools.keys():
            keywords = self._get_keywords_for_tool(tool_name)
            if any(keyword in task_lower for keyword in keywords):
                actions.append({
                    "tool": tool_name,
                    "parameters": self._extract_parameters_for_tool(tool_name, task),
                    "priority": 1,
                    "reason": f"Keywords matched for {tool_name}"
                })
        
        # Return planned actions or default
        return actions if actions else self._get_default_actions()
    
    def _get_keywords_for_tool(self, tool_name: str) -> List[str]:
        """Get keywords that trigger tool usage"""
        keyword_map = {
            "check_invoice": ["invoice", "charge", "billing", "charged", "payment", "amount"],
            "get_payment_history": ["payment", "history", "previous", "past payments"],
            "process_refund": ["refund", "return", "money back", "refund me", "get refund"],
            "search_knowledge_base": ["how", "help", "issue", "problem", "error", "troubleshoot", "fix"],
            "check_system_logs": ["login", "account", "access", "locked", "can't login", "password"],
            "get_order_status": ["order", "shipping", "delivery", "tracking", "where is", "status"],
            "create_support_ticket": ["escalate", "support", "urgent", "help", "ticket"],
        }
        return keyword_map.get(tool_name, [])
    
    def _extract_parameters_for_tool(self, tool_name: str, task: str) -> Dict[str, Any]:
        """
        Extract required parameters from task
        In production, this would use advanced NLP
        """
        # Default customer ID (in real system, would come from context/session)
        params = {"customer_id": "CUST_12345"}
        
        # Extract specific parameters based on tool
        if tool_name == "get_order_status":
            params["order_id"] = "ORD_789456"
        elif tool_name == "check_invoice":
            params["invoice_id"] = "INV_123456"
        elif tool_name == "process_refund":
            params["amount"] = 75.00
            params["reason"] = "Duplicate charge"
        elif tool_name == "search_knowledge_base":
            params["query"] = task
        elif tool_name == "check_system_logs":
            params["hours"] = 24
        elif tool_name == "create_support_ticket":
            params["issue"] = task
            params["priority"] = "high" if "urgent" in task.lower() else "medium"
        
        return params
    
    def _get_default_actions(self) -> List[Dict[str, Any]]:
        """Get default actions if no specific tool matches"""
        if "search_knowledge_base" in self.tools:
            return [{
                "tool": "search_knowledge_base",
                "parameters": {"query": "general customer support"},
                "priority": 1,
                "reason": "Default fallback action"
            }]
        return []
    
    def act(self, planned_actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execution phase: Call tools and capture results
        This is where actions are actually executed
        """
        logger.info(f"[{self.agent_id}] ⚙️  ACT phase - Executing {len(planned_actions)} actions")
        
        results = []
        
        for action in planned_actions:
            tool_name = action["tool"]
            params = action["parameters"]
            
            # Validate tool exists
            if tool_name not in self.tools:
                logger.warning(f"[{self.agent_id}] Tool not available: {tool_name}")
                results.append({
                    "tool": tool_name,
                    "status": "error",
                    "error": f"Tool {tool_name} not available"
                })
                continue
            
            try:
                logger.info(f"[{self.agent_id}] Calling tool: {tool_name}")
                tool = self.tools[tool_name]
                
                # Filter parameters to match tool signature
                filtered_params = {
                    k: v for k, v in params.items()
                    if k in tool.required_params or k in tool.optional_params
                }
                
                # Validate required parameters
                missing_params = [p for p in tool.required_params if p not in filtered_params]
                if missing_params:
                    raise ValueError(f"Missing required parameters: {missing_params}")
                
                # Execute the tool
                tool_result = tool.function(**filtered_params)
                
                results.append({
                    "tool": tool_name,
                    "status": "success",
                    "result": tool_result,
                    "timestamp": datetime.now().isoformat(),
                    "parameters_used": filtered_params
                })
                
                logger.info(f"[{self.agent_id}] ✅ Tool succeeded: {tool_name}")
                
            except Exception as e:
                logger.error(f"[{self.agent_id}] ❌ Tool failed: {tool_name} - {str(e)}")
                results.append({
                    "tool": tool_name,
                    "status": "error",
                    "error": str(e)
                })
        
        return results
    
    def observe(self, results: List[Dict[str, Any]]) -> str:
        """
        Reflection phase: Learn from results and update memory
        This is where agents reflect on outcomes
        """
        logger.info(f"[{self.agent_id}] 🔍 OBSERVE phase - Processing results")
        
        successful = len([r for r in results if r['status'] == 'success'])
        failed = len([r for r in results if r['status'] == 'error'])
        
        synthesis = f"Successfully executed {successful} tools. Failed: {failed}. "
        synthesis += "Processing results and updating memory."
        
        # Store in long-term memory
        self.memory["long_term"].append({
            "timestamp": datetime.now().isoformat(),
            "task_type": "tool_execution",
            "results_summary": {
                "total_tools": len(results),
                "successful": successful,
                "failed": failed
            },
            "synthesis": synthesis
        })
        
        # Keep only recent memories (memory management)
        if len(self.memory["long_term"]) > 100:
            self.memory["long_term"] = self.memory["long_term"][-100:]
        
        logger.info(f"[{self.agent_id}] Memory updated. Long-term memory size: {len(self.memory['long_term'])}")
        
        return synthesis
    
    def run(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Full Think-Act-Observe loop
        This is the main execution method that orchestrates all three phases
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"[{self.agent_id}] Starting full TAO loop")
        logger.info(f"{'='*60}")
        
        if context is None:
            context = self.memory["shared_context"]
        
        self.interaction_count += 1
        start_time = datetime.now()
        
        # Phase 1: THINK - Reasoning
        thinking = self.think(task, context)
        
        # Phase 2: ACT - Execution
        execution_results = self.act(thinking["planned_actions"])
        
        # Phase 3: OBSERVE - Reflection
        synthesis = self.observe(execution_results)
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "agent": self.role.name,
            "agent_id": self.agent_id,
            "task": task,
            "thinking": thinking,
            "execution": execution_results,
            "synthesis": synthesis,
            "status": "success" if all(r["status"] == "success" for r in execution_results) else "partial_success",
            "execution_time_seconds": elapsed_time,
            "interaction_count": self.interaction_count
        }


# ==========================================
# 4. SPECIALIZED AGENTS
# ==========================================

class QueryClassifierAgent(AdaptiveAgent):
    """Classifies customer queries into categories"""
    
    def think(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Classify the query into appropriate category"""
        
        task_lower = task.lower()
        
        # Classification logic
        if any(word in task_lower for word in ["invoice", "payment", "charge", "refund", "billing", "charged"]):
            classification = "billing"
        elif any(word in task_lower for word in ["login", "password", "account", "error", "bug", "crash", "access"]):
            classification = "technical"
        elif any(word in task_lower for word in ["order", "shipping", "delivery", "return", "product", "tracking"]):
            classification = "order"
        else:
            classification = "general"
        
        return {
            "classification": classification,
            "confidence": 0.92,
            "reasoning": f"Query contains keywords associated with {classification} category"
        }


class BillingAgent(AdaptiveAgent):
    """Handles billing and payment related issues"""
    
    def __init__(self, available_tools: Dict[str, Tool]):
        role = AgentRole(
            name="Billing Specialist",
            goal="Resolve all billing-related customer issues",
            expertise="Financial systems, invoice management, payment processing, refunds",
            tools=["check_invoice", "get_payment_history", "process_refund"]
        )
        super().__init__(role, available_tools, "billing_agent")


class TechnicalSupportAgent(AdaptiveAgent):
    """Handles technical and account-related issues"""
    
    def __init__(self, available_tools: Dict[str, Tool]):
        role = AgentRole(
            name="Technical Support Specialist",
            goal="Troubleshoot and resolve technical issues",
            expertise="System troubleshooting, error diagnosis, solution recommendation, account security",
            tools=["search_knowledge_base", "check_system_logs"]
        )
        super().__init__(role, available_tools, "tech_support_agent")


class OrderFulfillmentAgent(AdaptiveAgent):
    """Handles order tracking and fulfillment"""
    
    def __init__(self, available_tools: Dict[str, Tool]):
        role = AgentRole(
            name="Order Fulfillment Specialist",
            goal="Track orders and manage fulfillment",
            expertise="Order management, shipping, logistics, tracking",
            tools=["get_order_status", "create_support_ticket"]
        )
        super().__init__(role, available_tools, "order_agent")


class ResponseCoordinatorAgent(AdaptiveAgent):
    """Coordinates responses from specialist agents into coherent customer responses"""
    
    def __init__(self, available_tools: Dict[str, Tool]):
        role = AgentRole(
            name="Response Coordinator",
            goal="Create coherent, helpful customer responses",
            expertise="Communication, response synthesis, tone matching",
            tools=[]  # Doesn't call tools, synthesizes specialist results
        )
        super().__init__(role, available_tools, "coordinator_agent")
    
    def coordinate(self, specialist_results: Dict[str, Any], customer_query: str) -> str:
        """Synthesize specialist responses into one coherent answer"""
        
        logger.info(f"[{self.agent_id}] Coordinating responses from {len(specialist_results)} specialists")
        
        coordination = f"📋 Based on analysis of your query:\n\n"
        
        for agent_name, result in specialist_results.items():
            if result["status"] in ["success", "partial_success"]:
                synthesis = result.get('synthesis', 'Analysis completed')
                coordination += f"  • {agent_name}: {synthesis}\n"
        
        coordination += "\n✅ Next steps: Please let us know if you need further assistance."
        
        return coordination


# ==========================================
# 5. ORCHESTRATION
# ==========================================

class MultiAgentOrchestrator:
    """Orchestrates multiple agents to handle complex customer queries"""
    
    def __init__(self, agents: Dict[str, AdaptiveAgent]):
        """Initialize orchestrator with available agents"""
        self.agents = agents
        self.workflow_results = {}
        self.execution_history = []
        logger.info(f"Orchestrator initialized with agents: {list(agents.keys())}")
    
    def classify_query(self, customer_query: str) -> str:
        """Classify customer query into appropriate category"""
        
        classifier = self.agents.get("query_classifier")
        if not classifier:
            logger.warning("Classifier agent not found, defaulting to 'general'")
            return "general"
        
        result = classifier.think(customer_query, {})
        classification = result["classification"]
        logger.info(f"Query classified as: {classification} (confidence: {result['confidence']})")
        
        return classification
    
    async def parallel_specialist_execution(self,
                                           query: str,
                                           specialists: List[str],
                                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multiple specialist agents in parallel for efficiency"""
        
        logger.info(f"🚀 Starting parallel execution with {len(specialists)} specialists: {specialists}")
        
        tasks = {}
        for specialist_name in specialists:
            if specialist_name in self.agents:
                agent = self.agents[specialist_name]
                # Create async task for each agent
                task = asyncio.create_task(
                    asyncio.to_thread(agent.run, query, context)
                )
                tasks[specialist_name] = task
        
        # Wait for all tasks to complete
        results = {}
        for specialist_name, task in tasks.items():
            try:
                result = await task
                results[specialist_name] = result
                logger.info(f"✅ {specialist_name} completed in {result.get('execution_time_seconds', 0):.2f}s")
            except Exception as e:
                logger.error(f"❌ {specialist_name} failed: {str(e)}")
                results[specialist_name] = {
                    "status": "error",
                    "error": str(e),
                    "agent_id": specialist_name
                }
        
        return results
    
    def coordinate_response(self,
                           specialist_results: Dict[str, Any],
                           customer_query: str) -> str:
        """Coordinate specialist responses into one customer response"""
        
        coordinator = self.agents.get("response_coordinator")
        if not coordinator:
            logger.warning("Coordinator agent not found")
            return "Unable to process response"
        
        response = coordinator.coordinate(specialist_results, customer_query)
        return response
    
    async def handle_customer_query(self, customer_query: str) -> Dict[str, Any]:
        """
        Complete workflow: Classify → Select Specialists → Execute Parallel → Coordinate
        This is the main orchestration method
        """
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📋 NEW CUSTOMER QUERY: {customer_query}")
        logger.info(f"{'='*80}\n")
        
        # Step 1: CLASSIFY
        classification = self.classify_query(customer_query)
        logger.info(f"🏷️  Classification: {classification}\n")
        
        # Step 2: SELECT SPECIALISTS
        specialist_map = {
            "billing": ["billing_agent"],
            "technical": ["tech_support_agent"],
            "order": ["order_agent"],
            "general": ["tech_support_agent", "order_agent"]
        }
        
        specialists = specialist_map.get(classification, specialist_map["general"])
        
        # Step 3: EXECUTE SPECIALISTS IN PARALLEL
        context = {
            "classification": classification,
            "customer_query": customer_query,
            "timestamp": datetime.now().isoformat()
        }
        
        specialist_results = await self.parallel_specialist_execution(
            customer_query,
            specialists,
            context
        )
        
        logger.info(f"✅ All specialists completed\n")
        
        # Step 4: COORDINATE RESPONSE
        final_response = self.coordinate_response(specialist_results, customer_query)
        
        # Store result
        result = {
            "customer_query": customer_query,
            "classification": classification,
            "specialists_used": specialists,
            "specialist_results": specialist_results,
            "final_response": final_response,
            "timestamp": datetime.now().isoformat()
        }
        
        self.execution_history.append(result)
        
        return result


# ==========================================
# 6. MAIN EXECUTION
# ==========================================

async def main():
    """Main execution - complete multi-agent system in action"""
    
    print("\n" + "="*80)
    print("🤖 Multi-Agent Customer Support System with Agent Garden AI")
    print("="*80)
    print("Architecture: Think-Act-Observe Loop with Parallel Execution\n")
    
    # Initialize tools
    logger.info("Initializing tool registry...")
    tool_registry = create_tool_registry()
    logger.info(f"Tool registry created with {len(tool_registry)} tools\n")
    
    # Initialize agents
    logger.info("Initializing agents...")
    agents = {
        "query_classifier": QueryClassifierAgent(
            AgentRole(
                name="Query Classifier",
                goal="Accurately classify customer issues",
                expertise="Natural language understanding",
                tools=[]
            ),
            tool_registry,
            "query_classifier"
        ),
        "billing_agent": BillingAgent(tool_registry),
        "tech_support_agent": TechnicalSupportAgent(tool_registry),
        "order_agent": OrderFulfillmentAgent(tool_registry),
        "response_coordinator": ResponseCoordinatorAgent(tool_registry),
    }
    logger.info(f"Initialized {len(agents)} agents\n")
    
    # Create orchestrator
    orchestrator = MultiAgentOrchestrator(agents)
    
    # Test queries representing real-world scenarios
    test_queries = [
        "I was charged twice for my subscription this month. Can you help me get a refund?",
        "Where is my order? I placed it last week and haven't received tracking information.",
        "I can't log into my account. I keep getting an error message saying my password is wrong.",
    ]
    
    print("\n" + "="*80)
    print("PROCESSING CUSTOMER QUERIES")
    print("="*80)
    
    # Process each query
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'🔹'*40}")
        print(f"QUERY {i} OF {len(test_queries)}")
        print(f"{'🔹'*40}")
        print(f"\n📨 Customer: {query}\n")
        
        result = await orchestrator.handle_customer_query(query)
        
        print(f"\n📤 Final Response:")
        print(f"{'-'*80}")
        print(result['final_response'])
        print(f"{'-'*80}\n")
        
        # Show execution summary
        total_time = sum(r.get('execution_time_seconds', 0) for r in result['specialist_results'].values())
        print(f"⏱️  Total execution time: {total_time:.3f}s")
        print(f"📊 Specialists used: {', '.join(result['specialists_used'])}")
        print(f"🏷️  Classification: {result['classification']}")
    
    # Summary
    print("\n\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print(f"✅ Successfully processed {len(test_queries)} customer queries")
    print(f"📊 Total queries in history: {len(orchestrator.execution_history)}")
    print(f"🤖 Agents deployed: {len(agents)}")
    print(f"🛠️  Tools available: {len(tool_registry)}")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
