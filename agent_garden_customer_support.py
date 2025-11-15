"""
Building Multi-Agent Systems with Agent Garden AI
Complete Code Implementation
===========================================
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==========================================
# 1. DATA STRUCTURES & TYPES
# ==========================================

class ToolCategory(Enum):
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
    """Tool definition with metadata"""
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
    """Registry of available tools for agents"""
    
    @staticmethod
    def check_invoice(customer_id: str, invoice_id: str) -> Dict[str, Any]:
        """Retrieve invoice details from billing system."""
        # Simulated database call
        return {
            "invoice_id": invoice_id,
            "customer_id": customer_id,
            "amount": 149.99,
            "date": "2025-11-01",
            "status": "paid",
            "items": ["Product A", "Service B"],
            "tax": 12.00,
            "total": 161.99
        }
    
    @staticmethod
    def get_payment_history(customer_id: str, months: int = 12) -> List[Dict[str, Any]]:
        """Get customer payment history."""
        return [
            {"date": "2025-11-01", "amount": 149.99, "status": "paid", "method": "credit_card"},
            {"date": "2025-10-01", "amount": 89.99, "status": "paid", "method": "credit_card"},
            {"date": "2025-09-01", "amount": 149.99, "status": "paid", "method": "bank_transfer"},
        ]
    
    @staticmethod
    def process_refund(customer_id: str, amount: float, reason: str) -> Dict[str, Any]:
        """Process a refund for customer."""
        return {
            "refund_id": f"REF_{customer_id}_{datetime.now().timestamp()}",
            "amount": amount,
            "status": "processing",
            "estimated_completion": "3-5 business days",
            "reason": reason
        }
    
    @staticmethod
    def search_knowledge_base(query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search technical knowledge base using semantic search."""
        return [
            {
                "title": "How to reset your password",
                "relevance_score": 0.95,
                "url": "kb.example.com/password-reset",
                "category": "Account Management"
            },
            {
                "title": "Common login issues",
                "relevance_score": 0.87,
                "url": "kb.example.com/login-issues",
                "category": "Troubleshooting"
            }
        ]
    
    @staticmethod
    def check_system_logs(customer_id: str, hours: int = 24) -> Dict[str, Any]:
        """Check system logs for customer's account."""
        return {
            "customer_id": customer_id,
            "period_hours": hours,
            "login_attempts": 5,
            "failed_logins": 2,
            "last_activity": "2025-11-14 10:30:00",
            "status": "account_locked_after_failed_attempts",
            "recommendation": "Reset password and enable 2FA"
        }
    
    @staticmethod
    def get_order_status(customer_id: str, order_id: str) -> Dict[str, Any]:
        """Get order fulfillment status."""
        return {
            "order_id": order_id,
            "customer_id": customer_id,
            "status": "shipped",
            "tracking_number": "1Z999AA10123456784",
            "carrier": "UPS",
            "estimated_delivery": "2025-11-16",
            "items": 2,
            "total_amount": 199.99
        }
    
    @staticmethod
    def create_support_ticket(customer_id: str, issue: str, priority: str = "medium") -> Dict[str, Any]:
        """Create a support ticket in ticketing system."""
        return {
            "ticket_id": f"TICKET_{datetime.now().timestamp()}",
            "customer_id": customer_id,
            "issue": issue,
            "priority": priority,
            "status": "open",
            "created_at": datetime.now().isoformat(),
            "assigned_to": "support_team"
        }


def create_tool_registry() -> Dict[str, Tool]:
    """Create registry of available tools"""
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
    """Base adaptive agent with reasoning-action-observation loop"""
    
    def __init__(self, role: AgentRole, available_tools: Dict[str, Tool], agent_id: str = None):
        self.role = role
        self.tools = {t: available_tools[t] for t in role.tools if t in available_tools}
        self.agent_id = agent_id or role.name.lower().replace(" ", "_")
        
        # Memory systems
        self.memory = {
            "short_term": [],  # Current conversation
            "long_term": [],   # Past learnings
            "shared_context": {}  # Inter-agent communication
        }
        
        self.execution_log = []
        self.interaction_count = 0
    
    def think(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Reasoning phase: Decide what actions to take"""
        
        logger.info(f"[{self.agent_id}] Thinking about: {task[:100]}...")
        
        # Simulate reasoning
        thinking_output = {
            "agent": self.agent_id,
            "task": task,
            "reasoning": f"As {self.role.name}, I will {self.role.goal}. Available tools: {', '.join(self.tools.keys())}",
            "planned_actions": self._plan_actions(task),
            "confidence": 0.85
        }
        
        return thinking_output
    
    def _plan_actions(self, task: str) -> List[Dict[str, Any]]:
        """Plan which tools to call based on task"""
        # Simple heuristic planning
        actions = []
        
        task_lower = task.lower()
        
        for tool_name in self.tools.keys():
            if any(keyword in task_lower for keyword in self._get_keywords_for_tool(tool_name)):
                actions.append({
                    "tool": tool_name,
                    "parameters": self._extract_parameters_for_tool(tool_name, task),
                    "priority": 1
                })
        
        return actions if actions else self._get_default_actions()
    
    def _get_keywords_for_tool(self, tool_name: str) -> List[str]:
        """Get keywords that trigger tool usage"""
        keyword_map = {
            "check_invoice": ["invoice", "charge", "billing", "payment"],
            "get_payment_history": ["payment", "history", "previous"],
            "process_refund": ["refund", "return", "money back"],
            "search_knowledge_base": ["how", "help", "issue", "problem", "error"],
            "check_system_logs": ["login", "account", "access", "locked"],
            "get_order_status": ["order", "shipping", "delivery", "tracking"],
            "create_support_ticket": ["escalate", "support", "urgent"]
        }
        return keyword_map.get(tool_name, [])
    
    def _extract_parameters_for_tool(self, tool_name: str, task: str) -> Dict[str, Any]:
        """Extract required parameters from task"""
        # Simplified parameter extraction
        return {
            "customer_id": "CUST_12345",  # Would extract from context in real system
            "query": task
        }
    
    def _get_default_actions(self) -> List[Dict[str, Any]]:
        """Get default actions if no specific tool matches"""
        # Use search as fallback
        if "search_knowledge_base" in self.tools:
            return [{
                "tool": "search_knowledge_base",
                "parameters": {"query": "general inquiry"},
                "priority": 1
            }]
        return []
    
    def act(self, planned_actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execution phase: Call tools"""
        
        results = []
        
        for action in planned_actions:
            tool_name = action["tool"]
            params = action["parameters"]
            
            if tool_name not in self.tools:
                results.append({
                    "tool": tool_name,
                    "status": "error",
                    "error": f"Tool {tool_name} not available"
                })
                logger.warning(f"[{self.agent_id}] Tool not available: {tool_name}")
                continue
            
            try:
                logger.info(f"[{self.agent_id}] Executing tool: {tool_name}")
                tool = self.tools[tool_name]
                
                # Filter parameters to match tool signature
                filtered_params = {
                    k: v for k, v in params.items()
                    if k in tool.required_params or k in tool.optional_params
                }
                
                # Call the tool
                result = tool.function(**filtered_params)
                
                results.append({
                    "tool": tool_name,
                    "status": "success",
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                })
                
                self.execution_log.append({
                    "tool": tool_name,
                    "timestamp": datetime.now().isoformat(),
                    "status": "success"
                })
                
                logger.info(f"[{self.agent_id}] Tool succeeded: {tool_name}")
                
            except Exception as e:
                logger.error(f"[{self.agent_id}] Tool failed: {tool_name} - {str(e)}")
                results.append({
                    "tool": tool_name,
                    "status": "error",
                    "error": str(e)
                })
        
        return results
    
    def observe(self, results: List[Dict[str, Any]]) -> str:
        """Reflection phase: Learn from results"""
        
        synthesis = f"Successfully executed {len([r for r in results if r['status'] == 'success'])} tools. "
        synthesis += f"Failed: {len([r for r in results if r['status'] == 'error'])}. "
        synthesis += "Processing results and updating memory."
        
        # Store in long-term memory
        self.memory["long_term"].append({
            "timestamp": datetime.now().isoformat(),
            "task_type": "tool_execution",
            "results_summary": {
                "total_tools": len(results),
                "successful": len([r for r in results if r['status'] == 'success']),
                "failed": len([r for r in results if r['status'] == 'error'])
            },
            "synthesis": synthesis
        })
        
        # Keep only recent memories
        if len(self.memory["long_term"]) > 100:
            self.memory["long_term"] = self.memory["long_term"][-100:]
        
        return synthesis
    
    def run(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Full reasoning-action-observation loop"""
        
        if context is None:
            context = self.memory["shared_context"]
        
        self.interaction_count += 1
        start_time = datetime.now()
        
        # Phase 1: Think
        thinking = self.think(task, context)
        
        # Phase 2: Act
        execution_results = self.act(thinking["planned_actions"])
        
        # Phase 3: Observe
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
            "execution_time_seconds": elapsed_time
        }


# ==========================================
# 4. SPECIALIZED AGENTS
# ==========================================

class QueryClassifierAgent(AdaptiveAgent):
    """Classifies customer queries into categories"""
    
    def think(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Classify the query"""
        
        task_lower = task.lower()
        
        if any(word in task_lower for word in ["invoice", "payment", "charge", "refund", "billing"]):
            classification = "billing"
        elif any(word in task_lower for word in ["login", "password", "account", "error", "bug", "crash"]):
            classification = "technical"
        elif any(word in task_lower for word in ["order", "shipping", "delivery", "return", "product"]):
            classification = "order"
        else:
            classification = "general"
        
        return {
            "classification": classification,
            "confidence": 0.92,
            "reasoning": f"Query contains keywords associated with {classification} category"
        }


class BillingAgent(AdaptiveAgent):
    """Handles billing and payment issues"""
    
    def __init__(self, available_tools: Dict[str, Tool]):
        role = AgentRole(
            name="Billing Specialist",
            goal="Resolve all billing-related customer issues",
            expertise="Financial systems, invoice management, payment processing",
            tools=["check_invoice", "get_payment_history", "process_refund"]
        )
        super().__init__(role, available_tools, "billing_agent")


class TechnicalSupportAgent(AdaptiveAgent):
    """Handles technical issues"""
    
    def __init__(self, available_tools: Dict[str, Tool]):
        role = AgentRole(
            name="Technical Support Specialist",
            goal="Troubleshoot and resolve technical issues",
            expertise="System troubleshooting, error diagnosis, solution recommendation",
            tools=["search_knowledge_base", "check_system_logs"]
        )
        super().__init__(role, available_tools, "tech_support_agent")


class OrderFulfillmentAgent(AdaptiveAgent):
    """Handles order and fulfillment issues"""
    
    def __init__(self, available_tools: Dict[str, Tool]):
        role = AgentRole(
            name="Order Fulfillment Specialist",
            goal="Track orders and manage fulfillment",
            expertise="Order management, shipping, logistics",
            tools=["get_order_status", "create_support_ticket"]
        )
        super().__init__(role, available_tools, "order_agent")


class ResponseCoordinatorAgent(AdaptiveAgent):
    """Coordinates responses from specialist agents"""
    
    def __init__(self, available_tools: Dict[str, Tool]):
        role = AgentRole(
            name="Response Coordinator",
            goal="Create coherent, helpful customer responses",
            expertise="Communication, response synthesis, tone matching",
            tools=[]  # Doesn't call tools, uses specialist results
        )
        super().__init__(role, available_tools, "coordinator_agent")
    
    def coordinate(self, specialist_results: Dict[str, Any], customer_query: str) -> str:
        """Synthesize specialist responses into one coherent answer"""
        
        coordination = f"Based on analysis of your query about {customer_query[:50]}...\n\n"
        
        for agent_name, result in specialist_results.items():
            if result["status"] in ["success", "partial_success"]:
                coordination += f"From {agent_name}: {result.get('synthesis', 'Analysis completed')}\n"
        
        coordination += "\nNext steps: Please let us know if you need further assistance."
        
        return coordination


# ==========================================
# 5. ORCHESTRATION
# ==========================================

class MultiAgentOrchestrator:
    """Orchestrates multiple agents to handle complex tasks"""
    
    def __init__(self, agents: Dict[str, AdaptiveAgent]):
        self.agents = agents
        self.workflow_results = {}
        self.execution_history = []
    
    def classify_query(self, customer_query: str) -> str:
        """Classify customer query"""
        
        classifier = self.agents.get("query_classifier")
        if not classifier:
            return "general"
        
        result = classifier.think(customer_query, {})
        return result["classification"]
    
    async def parallel_specialist_execution(self,
                                           query: str,
                                           specialists: List[str],
                                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multiple specialist agents in parallel"""
        
        logger.info(f"Starting parallel execution with specialists: {specialists}")
        
        tasks = {}
        for specialist_name in specialists:
            if specialist_name in self.agents:
                agent = self.agents[specialist_name]
                # Run agent asynchronously
                task = asyncio.create_task(
                    asyncio.to_thread(agent.run, query, context)
                )
                tasks[specialist_name] = task
        
        # Gather all results
        results = {}
        for specialist_name, task in tasks.items():
            try:
                result = await task
                results[specialist_name] = result
                logger.info(f"✅ {specialist_name} completed")
            except Exception as e:
                logger.error(f"❌ {specialist_name} failed: {str(e)}")
                results[specialist_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return results
    
    def coordinate_response(self,
                           specialist_results: Dict[str, Any],
                           customer_query: str) -> str:
        """Coordinate specialist responses into one customer response"""
        
        coordinator = self.agents.get("response_coordinator")
        if not coordinator:
            return "Unable to process response"
        
        response = coordinator.coordinate(specialist_results, customer_query)
        return response
    
    async def handle_customer_query(self, customer_query: str) -> Dict[str, Any]:
        """Complete workflow: Classify → Parallelize → Coordinate"""
        
        logger.info(f"📋 New customer query: {customer_query[:100]}...")
        
        # Step 1: Classify
        classification = self.classify_query(customer_query)
        logger.info(f"🏷️  Classification: {classification}")
        
        # Step 2: Select specialists
        specialist_map = {
            "billing": ["billing_agent"],
            "technical": ["tech_support_agent"],
            "order": ["order_agent"],
            "general": ["tech_support_agent", "order_agent"]
        }
        
        specialists = specialist_map.get(classification, specialist_map["general"])
        
        # Step 3: Execute specialists in parallel
        context = {
            "classification": classification,
            "customer_query": customer_query
        }
        
        specialist_results = await self.parallel_specialist_execution(
            customer_query,
            specialists,
            context
        )
        
        logger.info(f"✅ All specialists completed")
        
        # Step 4: Coordinate response
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
    
    print("\n" + "="*60)
    print("🤖 Multi-Agent Customer Support System")
    print("="*60 + "\n")
    
    # Initialize tools
    tool_registry = create_tool_registry()
    
    # Initialize agents
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
    
    # Create orchestrator
    orchestrator = MultiAgentOrchestrator(agents)
    
    # Test queries
    test_queries = [
        "I was charged twice for my subscription this month. I also can't log into my account. Can you help?",
        "Where is my order? I placed it last week.",
        "I need to return a product. How do I do that?",
    ]
    
    # Process each query
    for query in test_queries:
        print(f"\n📨 Customer: {query}")
        print("-" * 60)
        
        result = await orchestrator.handle_customer_query(query)
        
        print(f"\n📤 Response:\n{result['final_response']}")
        print("-" * 60)
        print(f"Execution time: {sum(r.get('execution_time_seconds', 0) for r in result['specialist_results'].values()):.2f}s")
        print()
    
    print("\n" + "="*60)
    print("✅ All queries processed successfully!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
