# AgentScope: Complete Developer Tutorial

AgentScope is a production-ready, multi-agent platform that enables developers to build sophisticated AI applications with transparent message-passing architecture and comprehensive fault tolerance. Developed by Alibaba Group with strong academic foundations, AgentScope bridges the gap between research prototypes and enterprise-grade deployments, offering both zero-code development tools and advanced customization capabilities for building scalable agent systems.

**What makes AgentScope unique**: Unlike conversation-based frameworks like AutoGen or role-based systems like CrewAI, AgentScope implements explicit message-passing with a three-layered architecture that prioritizes developer transparency—there's no "magic" happening behind the scenes. This approach enables seamless scaling from local development to distributed production deployment while maintaining identical code, making it the go-to choice for organizations serious about production multi-agent systems.

**Core value proposition**: The framework combines enterprise-grade reliability (built-in fault tolerance, sandboxed execution, comprehensive monitoring) with developer productivity (drag-and-drop interfaces, visual debugging, automatic parallelization) to create a platform that works equally well for rapid prototyping and production deployment at massive scale.

## Production-ready architecture that scales seamlessly

AgentScope's **three-layered architecture** forms the foundation for building reliable agent systems. The **Utility Layer** provides essential services including API invocation, data retrieval, and code execution. The **Manager and Wrapper Layer** handles resource management, fault tolerance, and error recovery. The **Agent Layer** focuses on communication patterns and workflow orchestration with streamlined syntax.

This architecture abstracts four foundational components with strong modular decoupling: **Messages** (unified communication medium), **Models** (LLM provider abstraction), **Memory** (state management), and **Tools** (function calling system). Each component can be independently customized and extended while maintaining compatibility with the broader system.

**Message-based communication** serves as the universal medium across all interactions. Messages use a simple Python dictionary structure with required fields (`name`, `content`) and optional metadata (`url`, `timestamp`). This design supports multimodal content through URL references while enabling efficient lazy loading for large datasets. Every message receives a unique ID for complete traceability across distributed systems.

The **Model Module** provides unified abstraction across diverse LLM providers (OpenAI, Anthropic, Google Gemini, DashScope, local models via Ollama). Provider-specific formatters handle API differences automatically, enabling developers to write code once and run it with any supported model. The system includes native async/await support with streaming capabilities and comprehensive usage tracking.

## Complete installation and environment setup

### System requirements and installation methods

**Python version requirements** vary by AgentScope version. The latest **AgentScope v1.0** requires Python 3.10 or higher and represents the actively developed branch with async execution, real-time steering, and comprehensive tooling. The legacy version supports Python 3.9+ but lacks the latest features.

**Basic installation** from PyPI provides the quickest start:
```bash
# Standard installation
pip install agentscope

# With distributed capabilities
pip install agentscope[distribute]
```

**Development installation** from source enables access to cutting-edge features and customization:
```bash
git clone -b main https://github.com/agentscope-ai/agentscope.git
cd agentscope
pip install -e .[dev]
pre-commit install  # For development workflow
```

### Environment configuration and API setup

**Virtual environment setup** prevents dependency conflicts and provides isolation:
```bash
# Using conda (recommended)
conda create -n agentscope python=3.10
conda activate agentscope

# Using virtualenv
python -m venv agentscope-env
# Windows: agentscope-env\Scripts\activate
# macOS/Linux: source agentscope-env/bin/activate
```

**API key configuration** enables model access through environment variables:
```bash
# Windows
set OPENAI_API_KEY=your_key_here
set DASHSCOPE_API_KEY=your_dashscope_key

# macOS/Linux
export OPENAI_API_KEY=your_key_here
export DASHSCOPE_API_KEY=your_dashscope_key
```

**Model configuration** uses JSON files for flexible deployment:
```json
{
  "config_name": "my_openai_config",
  "model_type": "openai_chat",
  "model_name": "gpt-4",
  "api_key": "YOUR_OPENAI_API_KEY"
}
```

## Multi-agent system creation and communication patterns

### Building your first agent system

**Agent creation** follows the ReAct (Reasoning + Acting) paradigm where agents alternate between thinking and tool use:
```python
from agentscope.agent import ReActAgent, UserAgent
from agentscope.model import DashScopeChatModel
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit, execute_python_code
import asyncio

async def create_first_agent():
    # Create toolkit with basic tools
    toolkit = Toolkit()
    toolkit.register_tool_function(execute_python_code)
    
    # Create ReAct agent with reasoning and acting capabilities
    agent = ReActAgent(
        name="Assistant",
        sys_prompt="You're a helpful AI assistant with coding capabilities.",
        model=DashScopeChatModel(
            model_name="qwen-max",
            api_key=os.environ["DASHSCOPE_API_KEY"],
            stream=True
        ),
        memory=InMemoryMemory(),
        toolkit=toolkit
    )
    
    # Create user interface agent
    user = UserAgent(name="user")
    
    return agent, user
```

### Advanced communication orchestration

**Pipeline patterns** enable sophisticated conversation flows using programming-like constructs:
```python
from agentscope.pipeline import SequentialPipeline, IfElsePipeline, MsgHub
from agentscope.message import Msg

async def advanced_communication_example():
    # Create specialized agents
    researcher = ReActAgent(name="Researcher", toolkit=research_tools)
    analyst = ReActAgent(name="Analyst", toolkit=analysis_tools)
    writer = DialogAgent(name="Writer")
    
    # Sequential pipeline for structured workflow
    analysis_pipe = SequentialPipeline([researcher, analyst, writer])
    
    # Conditional routing based on content
    conditional_pipe = IfElsePipeline(
        condition_func=lambda msg: "technical" in msg.content.lower(),
        if_body=technical_agent,
        else_body=general_agent
    )
    
    # Multi-agent hub for collaborative work
    async with MsgHub(participants=[researcher, analyst, writer]) as hub:
        # Broadcast initial task
        await hub.broadcast(Msg("coordinator", "Analyze market trends"))
        
        # Dynamic participant management
        if complex_analysis_needed:
            hub.add(specialist_agent)
        
        # Execute pipeline within hub context
        result = await analysis_pipe(initial_message)
        return result
```

**MsgHub system** provides broadcast messaging and dynamic participant management for group conversations. Agents automatically observe messages from other participants, enabling natural collaborative workflows. The hub supports real-time addition and removal of participants, making it ideal for adaptive multi-agent systems.

## RAG capabilities and knowledge integration

### Implementing retrieval-augmented generation

**Knowledge bank creation** integrates seamlessly with LlamaIndex ecosystem for comprehensive RAG support:
```python
from agentscope.rag.llama_index_knowledge import LlamaIndexKnowledge
from agentscope.agent import LlamaIndexAgent

async def setup_rag_system():
    # Create shared knowledge repository
    knowledge = LlamaIndexKnowledge.build_knowledge_instance(
        knowledge_id="company_docs",
        data_dirs_and_types={
            "./docs": [".md", ".pdf"],
            "./reports": [".docx", ".txt"]
        },
        emb_model_config_name="embedding_model",
        similarity_top_k=5
    )
    
    # Create RAG-enabled agent
    rag_agent = LlamaIndexAgent(
        name="Knowledge_Assistant",
        model_config_name="gpt-4",
        knowledge_list=[knowledge],
        sys_prompt="Answer questions using the provided knowledge base."
    )
    
    return rag_agent
```

**Multi-agent RAG** enables knowledge sharing across agent teams:
```python
async def collaborative_rag_example():
    # Shared knowledge available to all agents
    legal_knowledge = LlamaIndexKnowledge.build_knowledge_instance(
        knowledge_id="legal_docs",
        data_dirs_and_types={"./legal": [".pdf"]}
    )
    
    technical_knowledge = LlamaIndexKnowledge.build_knowledge_instance(
        knowledge_id="technical_docs", 
        data_dirs_and_types={"./tech": [".md", ".py"]}
    )
    
    # Specialized agents with domain-specific knowledge
    legal_agent = LlamaIndexAgent(
        name="Legal_Expert",
        knowledge_list=[legal_knowledge],
        model_config_name="gpt-4"
    )
    
    tech_agent = LlamaIndexAgent(
        name="Technical_Expert", 
        knowledge_list=[technical_knowledge],
        model_config_name="gpt-4"
    )
    
    # Coordinator with access to all knowledge
    coordinator = LlamaIndexAgent(
        name="Coordinator",
        knowledge_list=[legal_knowledge, technical_knowledge],
        model_config_name="gpt-4"
    )
    
    return legal_agent, tech_agent, coordinator
```

## Tool integration and function calling system

### Comprehensive tool ecosystem

**Built-in tools** provide immediate functionality for common tasks:
```python
from agentscope.tool import (
    execute_python_code,
    execute_shell_command, 
    view_text_file,
    write_text_file,
    dashscope_text_to_image
)

# Register multiple tools efficiently
toolkit = Toolkit()
for tool in [execute_python_code, view_text_file, write_text_file]:
    toolkit.register_tool_function(tool)

# Create tool groups for organized management
toolkit.create_tool_group("file_tools", "File manipulation tools")
toolkit.create_tool_group("execution_tools", "Code and shell execution")
```

**Custom tool creation** enables domain-specific functionality:
```python
def custom_database_query(query: str, database: str = "main") -> str:
    """Execute database query and return results.
    
    Args:
        query: SQL query to execute
        database: Database name to query
        
    Returns:
        Query results as JSON string
    """
    # Implementation here
    results = execute_db_query(query, database)
    return json.dumps(results)

# Register custom tool with automatic schema generation
toolkit.register_tool_function(custom_database_query)

# Tool automatically available to agents
agent = ReActAgent(
    name="Data_Analyst",
    toolkit=toolkit,
    sys_prompt="You can query databases using the custom_database_query tool."
)
```

### Model Context Protocol integration

**MCP integration** provides universal access to external services:
```python
from agentscope.mcp import HttpStatelessClient

async def setup_mcp_integration():
    # Connect to MCP server
    mcp_client = HttpStatelessClient(
        name="weather_service",
        url="https://weather-mcp.example.com/api"
    )
    
    # Discover and register available functions
    weather_func = await mcp_client.get_callable_function("get_weather")
    map_func = await mcp_client.get_callable_function("search_location")
    
    # Register with toolkit
    toolkit = Toolkit()
    toolkit.register_tool_function(weather_func)
    toolkit.register_tool_function(map_func)
    
    # Create agent with MCP tools
    agent = ReActAgent(
        name="Weather_Assistant",
        toolkit=toolkit,
        sys_prompt="Help users with weather and location queries."
    )
    
    return agent
```

## Memory management and state persistence

### Memory architecture and types

**Memory systems** provide flexible storage for different use cases:
```python
from agentscope.memory import InMemoryMemory, FileDumpMemory

# Short-term conversation memory
short_term = InMemoryMemory()

# Persistent file-based memory
persistent = FileDumpMemory("agent_history.json")

# Memory operations
memory.add(Msg("user", "Hello"))
memory.add(Msg("assistant", "Hi there!"))

# Retrieve recent conversation
recent = memory.get_memory(recent_n=10)

# Export for analysis
exported = memory.export()
```

**State management** enables complete system persistence:
```python
from agentscope.session import StateModule

class CustomAgentState(StateModule):
    def __init__(self):
        super().__init__()
        self.conversation_count = 0
        self.user_preferences = {}
    
    def update_preferences(self, preferences):
        self.user_preferences.update(preferences)
        self.conversation_count += 1

# Agent with custom state
agent = ReActAgent(name="Stateful_Agent")
agent.state = CustomAgentState()

# Save complete agent state
state_dict = agent.state_dict()
with open("agent_state.json", "w") as f:
    json.dump(state_dict, f)

# Restore agent state
with open("agent_state.json", "r") as f:
    saved_state = json.load(f)
agent.load_state_dict(saved_state)
```

## Monitoring and debugging with AgentScope Studio

### Visual development environment

**AgentScope Studio** provides comprehensive development and monitoring capabilities through a web-based interface. The studio offers real-time visualization of agent interactions, token usage tracking, and built-in development assistance through the "Friday" agent.

**Studio setup** and integration:
```python
import agentscope

# Initialize with Studio connection
agentscope.init(
    model_configs="./model_config.json",
    studio_url="http://localhost:8000",
    project_name="My Agent Project"
)

# Studio automatically tracks all agent interactions
agent = ReActAgent(name="Monitored_Agent")
# All conversations appear in Studio dashboard
```

**Production monitoring** with OpenTelemetry integration:
```python
from agentscope.tracing import trace_llm, setup_tracing

# Configure tracing for production
setup_tracing(
    service_name="my_agent_service",
    tracing_endpoint="http://jaeger:14268/api/traces"
)

@trace_llm
async def monitored_agent_call():
    response = await agent("Analyze this data")
    return response
```

## Distributed deployment and scaling strategies

### Seamless local-to-distributed migration

**Actor-based distribution** enables transparent scaling from local development to multi-machine deployment:
```python
# Local development version
agent = ReActAgent(name="worker")

# Distributed version - identical interface
distributed_agent = ReActAgent(
    name="worker",
    to_dist=True  # Simple distributed flag
)

# Advanced distributed configuration
production_agent = ReActAgent(
    name="worker", 
    to_dist={
        "host": "worker-node-1.cluster.internal",
        "port": "12345",
        "resources": {"cpu": 2, "memory": "4GB"}
    }
)
```

**Large-scale orchestration** with automatic parallelization:
```python
async def distributed_workflow():
    # Create agent pool across multiple machines
    agents = []
    for i in range(100):
        agent = ReActAgent(
            name=f"worker_{i}",
            to_dist={"host": f"worker-{i % 10}.cluster"}
        )
        agents.append(agent)
    
    # Parallel execution with automatic load balancing
    tasks = [generate_task(i) for i in range(1000)]
    results = await asyncio.gather(*[
        agents[i % len(agents)](task) 
        for i, task in enumerate(tasks)
    ])
    
    return results
```

## Best practices and common patterns

### Error handling and fault tolerance

**Robust error handling** prevents system failures:
```python
from agentscope.utils.exceptions import AgentCallError

async def resilient_agent_interaction():
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await agent("Complex reasoning task")
            return response
        except AgentCallError as e:
            if attempt == max_retries - 1:
                # Final fallback strategy
                return await fallback_agent("Simplified version of task")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

**Budget management** and cost control:
```python
from agentscope.utils import BudgetManager

budget = BudgetManager(max_tokens=10000, max_cost=5.00)

@budget.track_usage
async def cost_controlled_operation():
    response = await expensive_agent("Analysis task")
    if budget.exceeds_limit():
        await budget.send_alert("Budget limit approaching")
    return response
```

### Performance optimization strategies

**Parallel tool execution** maximizes efficiency:
```python
async def optimized_research_workflow():
    # Multiple tools execute concurrently
    search_task = toolkit.execute_tool_function("web_search", {"query": "AI trends"})
    code_task = toolkit.execute_tool_function("execute_python_code", {"code": analysis_script})
    file_task = toolkit.execute_tool_function("view_text_file", {"file_path": "data.csv"})
    
    # Wait for all tools to complete
    search_results, code_results, file_contents = await asyncio.gather(
        search_task, code_task, file_task
    )
    
    return combine_results(search_results, code_results, file_contents)
```

## Practical implementation examples

### Complete multi-agent research system

```python
import asyncio
from agentscope.agent import ReActAgent, DialogAgent
from agentscope.pipeline import MsgHub, SequentialPipeline
from agentscope.tool import Toolkit
from agentscope.memory import InMemoryMemory

async def build_research_team():
    # Create specialized toolkit
    research_toolkit = Toolkit()
    research_toolkit.register_tool_function(web_search)
    research_toolkit.register_tool_function(execute_python_code)
    research_toolkit.register_tool_function(write_text_file)
    
    # Research coordinator
    coordinator = ReActAgent(
        name="Research_Coordinator",
        sys_prompt="""You coordinate research projects by breaking down topics 
        into subtasks and assigning them to specialists.""",
        toolkit=research_toolkit,
        memory=InMemoryMemory()
    )
    
    # Specialized researchers
    data_researcher = ReActAgent(
        name="Data_Researcher", 
        sys_prompt="You specialize in finding and analyzing quantitative data.",
        toolkit=research_toolkit
    )
    
    literature_researcher = ReActAgent(
        name="Literature_Researcher",
        sys_prompt="You specialize in academic literature and theoretical frameworks.",
        toolkit=research_toolkit
    )
    
    # Synthesis agent
    synthesizer = DialogAgent(
        name="Research_Synthesizer",
        sys_prompt="You synthesize research findings into comprehensive reports."
    )
    
    return coordinator, data_researcher, literature_researcher, synthesizer

async def run_research_project():
    coordinator, data_researcher, lit_researcher, synthesizer = await build_research_team()
    
    # Collaborative research workflow
    async with MsgHub(participants=[
        coordinator, data_researcher, lit_researcher, synthesizer
    ]) as hub:
        
        # Initialize research project
        project_msg = Msg(
            "user", 
            "Research the impact of AI on employment in the next decade"
        )
        
        # Coordinator plans and delegates
        plan = await coordinator(project_msg)
        await hub.broadcast(plan)
        
        # Parallel research execution
        data_task = data_researcher("Find employment statistics and AI adoption rates")
        lit_task = lit_researcher("Survey academic literature on AI employment impact")
        
        data_results, lit_results = await asyncio.gather(data_task, lit_task)
        
        # Synthesis phase
        synthesis_input = Msg(
            "coordinator",
            f"Synthesize these findings: {data_results.content} | {lit_results.content}"
        )
        
        final_report = await synthesizer(synthesis_input)
        return final_report
```

### Enterprise customer service system

```python
async def build_customer_service_system():
    # Knowledge base with company information
    kb = LlamaIndexKnowledge.build_knowledge_instance(
        knowledge_id="company_kb",
        data_dirs_and_types={"./knowledge": [".md", ".pdf"]}
    )
    
    # Customer service tools
    service_toolkit = Toolkit()
    service_toolkit.register_tool_function(lookup_customer_account)
    service_toolkit.register_tool_function(create_support_ticket)
    service_toolkit.register_tool_function(escalate_to_human)
    
    # Multi-tier agent system
    tier1_agent = LlamaIndexAgent(
        name="Tier1_Support",
        knowledge_list=[kb],
        toolkit=service_toolkit,
        sys_prompt="Handle basic customer inquiries with company knowledge."
    )
    
    tier2_agent = ReActAgent(
        name="Tier2_Support",
        toolkit=service_toolkit,
        sys_prompt="Handle complex technical issues and account problems."
    )
    
    escalation_router = DialogAgent(
        name="Escalation_Router",
        sys_prompt="Route complex cases to appropriate specialists."
    )
    
    # Routing logic with conditional pipelines
    async def handle_customer_inquiry(inquiry):
        # Initial assessment
        complexity_check = await tier1_agent(inquiry)
        
        if "escalate" in complexity_check.content.lower():
            routing_decision = await escalation_router(complexity_check)
            return await tier2_agent(routing_decision)
        else:
            return complexity_check
    
    return handle_customer_inquiry
```

## Framework comparisons and selection guidance

### When to choose AgentScope over alternatives

**AgentScope excels** in production environments requiring transparency, fault tolerance, and distributed deployment. Its message-passing architecture provides explicit control over agent interactions, making debugging and optimization straightforward. The framework's enterprise features (sandboxed execution, comprehensive monitoring, cost management) make it ideal for organizations deploying agent systems at scale.

**Choose AgentScope when**:
- Building production systems with reliability requirements
- Need transparent, debuggable agent behavior
- Scaling from prototype to enterprise deployment
- Managing multi-modal data and complex workflows
- Requiring fault tolerance and error recovery

**Alternative frameworks** serve different use cases:
- **AutoGen**: Best for open-ended research and complex reasoning tasks
- **CrewAI**: Ideal for simple business workflows and role-based automation  
- **LangGraph**: Optimal for complex state management and cyclical workflows
- **OpenAI Swarm**: Perfect for lightweight experiments in OpenAI environments

### Migration strategies from other frameworks

**From conversation-based frameworks** (AutoGen):
```python
# AutoGen conversation pattern
# conversation = [agent1, agent2, agent3]

# AgentScope equivalent with explicit message passing
async def migrate_from_autogen():
    async with MsgHub(participants=[agent1, agent2, agent3]) as hub:
        result = await sequential_pipeline([agent1, agent2, agent3])
        return result
```

**From role-based frameworks** (CrewAI):
```python
# CrewAI role-based approach
# crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task])

# AgentScope structured workflow
research_pipeline = SequentialPipeline([researcher, writer])
result = await research_pipeline(initial_task)
```

## Advanced development patterns

### Real-time steering and interruption handling

**Human-in-the-loop systems** with real-time intervention:
```python
async def interactive_agent_system():
    agent = ReActAgent(
        name="Interactive_Assistant",
        enable_interruption=True  # Enable real-time steering
    )
    
    # Start agent task
    task = asyncio.create_task(agent("Long-running analysis task"))
    
    # Allow real-time intervention
    while not task.done():
        user_input = await get_user_input_async()
        if user_input == "stop":
            task.cancel()
            break
        elif user_input.startswith("redirect:"):
            # Dynamically redirect agent behavior
            new_instruction = user_input[9:]
            await agent.handle_interrupt(Msg("user", new_instruction))
        
        await asyncio.sleep(1)
    
    result = await task
    return result
```

### Dynamic tool provisioning and adaptation

**Adaptive agent capabilities** based on context:
```python
async def adaptive_agent_system():
    base_toolkit = Toolkit()
    base_toolkit.register_tool_function(execute_python_code)
    
    # Context-aware tool provisioning
    async def adapt_tools_for_context(context, agent):
        if "data analysis" in context:
            data_tools = [pandas_query, matplotlib_plot, statistical_analysis]
            for tool in data_tools:
                agent.toolkit.register_tool_function(tool)
                
        elif "web research" in context:
            web_tools = [web_search, webpage_extract, link_analysis]
            for tool in web_tools:
                agent.toolkit.register_tool_function(tool)
        
        # Update agent's available tools
        agent.reset_equipped_tools()
    
    agent = ReActAgent(name="Adaptive_Agent", toolkit=base_toolkit)
    
    # Dynamically adapt based on incoming requests
    context = "I need to analyze sales data"
    await adapt_tools_for_context(context, agent)
    
    response = await agent("Analyze quarterly sales trends")
    return response
```

AgentScope represents a mature, production-focused approach to multi-agent development that successfully bridges the gap between research prototypes and enterprise applications. Its emphasis on transparency, fault tolerance, and seamless scaling makes it particularly valuable for organizations ready to deploy reliable agent systems that work consistently at scale. The framework's comprehensive tooling ecosystem, from zero-code development interfaces to advanced customization capabilities, ensures it serves both rapid prototyping needs and complex production requirements effectively.
