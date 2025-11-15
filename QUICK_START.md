# Quick Start Guide - Multi-Agent Customer Support System

## 🚀 Get Running in 2 Minutes

### 1. Clone & Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/multi-agent-customer-support.git
cd multi-agent-customer-support

# No installation needed! Uses Python standard library only
```

### 2. Run the Demo

```bash
python multi_agent_customer_support.py
```

**Expected output:**
```
🤖 Multi-Agent Customer Support System
✅ Successfully processed 3 customer queries
📊 Total queries in history: 3
🤖 Agents deployed: 5
🛠️  Tools available: 7
```

---

## 📖 Understanding the Output

### Query Processing Flow

```
📨 Customer Query Input
    ↓
🏷️  Query Classification (billing/technical/order)
    ↓
🚀 Parallel Specialist Execution
    ├─ [AGENT] 🧠 THINK phase
    ├─ [AGENT] ⚙️  ACT phase (tool execution)
    └─ [AGENT] 🔍 OBSERVE phase (reflection)
    ↓
📤 Response Coordination
    ↓
✅ Final Customer Response
```

### Log Levels

- 🧠 THINK: Agent analyzing and planning
- ⚙️ ACT: Agent executing tools
- 🔍 OBSERVE: Agent reflecting on results
- ✅ SUCCESS: Tool executed successfully
- ❌ ERROR: Tool failed

---

## 💡 Modify Test Queries

Edit the test queries in the main function:

```python
# File: multi_agent_customer_support.py
# Find this section around line 670:

test_queries = [
    "Your custom query 1",
    "Your custom query 2",
    "Your custom query 3",
]
```

**Examples:**
```python
test_queries = [
    "I need to return an item I ordered",
    "My password reset isn't working",
    "Can I update my billing address?",
]
```

Then run:
```bash
python multi_agent_customer_support.py
```

---

## 🎯 Key Concepts (Simplified)

### Think-Act-Observe Loop

```
THINK    → "What tools do I need?"
ACT      → "Let me call those tools"
OBSERVE  → "Here's what I learned"
```

### Query Routing

```
"I was charged twice"  → 🏷️  BILLING
"I can't log in"       → 🏷️  TECHNICAL
"Where's my order?"    → 🏷️  ORDER
```

### Parallel Execution

```
Single Agent (SLOW):
Agent1 → Agent2 → Agent3 (3 seconds)

Multiple Agents (FAST):
Agent1 ─┐
Agent2 ─┼ (1 second)
Agent3 ─┘
```

---

## 🔧 Simple Modifications

### Add a New Test Query

```python
# Line 670 in multi_agent_customer_support.py

test_queries = [
    "I was charged twice for my subscription this month. Can you help me get a refund?",
    "Where is my order? I placed it last week and haven't received tracking information.",
    "I can't log into my account. I keep getting an error message saying my password is wrong.",
    "Add your new query here!",  # ← NEW QUERY
]
```

### Change Logging Level

```python
# Line 30 in multi_agent_customer_support.py

# From:
logging.basicConfig(level=logging.INFO, ...)

# To:
logging.basicConfig(level=logging.DEBUG, ...)  # See all details
logging.basicConfig(level=logging.WARNING, ...)  # See only warnings
```

### Add More Agents

```python
# Create your new agent
class MyCustomAgent(AdaptiveAgent):
    def __init__(self, available_tools):
        role = AgentRole(
            name="My Custom Agent",
            goal="Do something specific",
            expertise="My expertise area",
            tools=["tool_name"]
        )
        super().__init__(role, available_tools)

# Add to agents dict (around line 660)
agents = {
    ...existing agents...,
    "my_agent": MyCustomAgent(tool_registry),  # ← ADD HERE
}
```

---

## 📊 Understanding the Results

### Result Structure

```python
result = {
    "customer_query": "...",              # Original query
    "classification": "billing",           # Category
    "specialists_used": ["agent1", "agent2"],  # Which agents helped
    "specialist_results": {...},          # Detailed results
    "final_response": "...",              # Customer-facing response
    "timestamp": "..."                    # When processed
}
```

### Accessing Results

```python
# In your code:
result = await orchestrator.handle_customer_query(query)

# Access parts:
print(result['classification'])           # "billing"
print(result['final_response'])           # Full response
print(result['specialists_used'])         # ["billing_agent"]
print(result['timestamp'])                # "2025-11-15T..."
```

---

## 🔍 Debug Mode

See what each agent is thinking:

```python
# Add this after handling a query:

for agent_name, agent_result in result['specialist_results'].items():
    print(f"\n{agent_name}:")
    print(f"  Status: {agent_result['status']}")
    print(f"  Time: {agent_result['execution_time_seconds']}s")
    print(f"  Tools used: {len(agent_result['execution'])}")
    
    for tool_result in agent_result['execution']:
        print(f"    - {tool_result['tool']}: {tool_result['status']}")
```

---

## 📝 Common Issues & Fixes

### Issue: "No module named asyncio"
**Solution:** Upgrade Python to 3.9+
```bash
python --version  # Check your version
```

### Issue: No output appears
**Solution:** Run with python, not python3
```bash
python multi_agent_customer_support.py
# Not: python3 multi_agent_customer_support.py
```

### Issue: Want to suppress logs
**Solution:** Add this at the top of main():
```python
logging.getLogger().setLevel(logging.CRITICAL)
```

---

## 🎓 Learning Path

### Beginner (30 mins)
1. Run the demo
2. Read the README
3. Look at the test queries
4. Understand query routing

### Intermediate (1-2 hours)
1. Modify test queries
2. Change which tools agents use
3. Add a new tool to a tool library
4. Add custom logging

### Advanced (2-4 hours)
1. Create a new agent type
2. Implement custom reasoning
3. Add real API integration
4. Build a web interface

---

## 🌐 Web Interface Example

Here's a simple Flask app to use this system:

```python
from flask import Flask, request, jsonify
import asyncio

app = Flask(__name__)

@app.route('/support', methods=['POST'])
def handle_query():
    query = request.json['query']
    
    # Run async function
    result = asyncio.run(
        orchestrator.handle_customer_query(query)
    )
    
    return jsonify({
        'response': result['final_response'],
        'type': result['classification']
    })

if __name__ == '__main__':
    # Initialize orchestrator first
    app.run(debug=True)
```

---

## 📚 File Structure

```
multi-agent-customer-support/
├── multi_agent_customer_support.py  # Main code (689 lines)
├── README.md                         # Full documentation
├── requirements.txt                  # Dependencies
├── .gitignore                        # Git ignore rules
├── QUICK_START.md                    # This file
└── MEDIUM_ARTICLE.md                 # Article template
```

---

## ✅ Checklist: First Run

- [ ] Python 3.9+ installed
- [ ] Repository cloned
- [ ] No dependencies to install
- [ ] Run `python multi_agent_customer_support.py`
- [ ] See 3 queries processed
- [ ] Modify a test query
- [ ] Run again to see custom query processed
- [ ] Read the detailed README

---

## 🤔 Next Steps

1. **Understand the Code**
   - Read the docstrings
   - Trace a query through the system
   - Look at the Think-Act-Observe loop

2. **Customize**
   - Add your own test queries
   - Create custom agents
   - Add new tools

3. **Extend**
   - Integrate with real APIs
   - Add a database
   - Build a web interface

4. **Deploy**
   - Containerize with Docker
   - Deploy to cloud
   - Add monitoring

---

## 🆘 Need Help?

### Check These First
- README.md (comprehensive guide)
- MEDIUM_ARTICLE.md (walkthrough with examples)
- Code comments (each function is documented)
- Test queries (real-world examples)

### Then...
- Open an issue on GitHub
- Check the examples in the docstrings
- Review the agent implementations

---

## 🎉 You're Ready!

Congratulations! You now have a working multi-agent system. Start exploring and building! 🚀

**Quick command reminder:**
```bash
python multi_agent_customer_support.py
```

That's it! 🎊
