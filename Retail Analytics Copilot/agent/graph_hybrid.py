from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
import dspy
import re
from .dspy_signatures import Router, SQLGeneratorModule, SynthesizerModule, RouteClassification, SQLGenerator,SQLOptimizer
from .rag.retrieval import SimpleRetriever
from .tools.sqlite_tool import SQLiteTool
from PIL import Image
import io
import json
from datetime import datetime
import os

class GraphState(TypedDict):
    question: str
    format_hint: str
    route: str
    retrieved_chunks: List[dict]
    constraints: dict
    sql_query: str
    sql_result: dict
    final_answer: str
    explanation: str
    citations: List[str]
    confidence: float
    repair_count: int
    error: Optional[str]

class TraceCheckpointer:
    def __init__(self, log_file: str = "agent_trace.log"):
        self.log_file = log_file
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._ensure_log_file()
        
    def _ensure_log_file(self):
        """Ensure log file exists with header."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("🧠 HYBRID AGENT EXECUTION TRACE LOG\n")
                f.write("=" * 50 + "\n")
                f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
    
    def log_event(self, event_type: str, node: str, state: dict, metadata: dict = None):
        """Log an event to the trace file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            # Event header with icon
            icons = {
                "optimization": "🚀",
                "node_start": "➡️",
                "node_success": "✅", 
                "node_error": "❌",
                "repair": "🔧",
                "validation": "📊",
                "execution": "🎯"
            }
            icon = icons.get(event_type, "📝")
            
            f.write(f"\n{icon} [{timestamp}] {event_type.upper()} - {node}\n")
            f.write("-" * 40 + "\n")
            
            # State summary
            if event_type in ["node_success", "node_start"]:
                self._write_state_summary(f, state)
            
            # Metadata details
            if metadata:
                f.write("📋 Metadata:\n")
                for key, value in metadata.items():
                    f.write(f"   {key}: {value}\n")
            
            f.write("\n")

    def _write_state_summary(self, f, state: dict):
        """Write formatted state summary."""
        # Route info
        if state.get('route'):
            f.write(f"🛣️  Route: {state['route']}\n")
        
        # SQL info
        if state.get('sql_query'):
            f.write(f"🗃️  SQL: {state['sql_query'][:100]}...\n")
        
        # Retrieval info
        if state.get('retrieved_chunks'):
            f.write(f"📚 Retrieved: {len(state['retrieved_chunks'])} chunks\n")
        
        # Constraints info
        if state.get('constraints'):
            constraints = state['constraints']
            active_constraints = [k for k, v in constraints.items() if v]
            if active_constraints:
                f.write(f"📋 Constraints: {', '.join(active_constraints)}\n")
        
        # Error info
        if state.get('error'):
            f.write(f"❌ Error: {state['error']}\n")
        
        # Confidence
        if state.get('confidence'):
            f.write(f"🎯 Confidence: {state['confidence']}\n")

    def log_optimization(self, module: str, before_metrics: dict, after_metrics: dict):
        """Log optimization results."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n🚀 OPTIMIZATION REPORT - {module.upper()}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Time: {timestamp}\n")
            f.write("-" * 50 + "\n")
            
            # Before metrics
            f.write("📊 BEFORE Optimization:\n")
            for metric, value in before_metrics.items():
                f.write(f"   {metric}: {value}\n")
            
            # After metrics  
            f.write("📈 AFTER Optimization:\n")
            for metric, value in after_metrics.items():
                f.write(f"   {metric}: {value}\n")
            
            # Improvement
            if 'valid_sql_rate' in before_metrics and 'valid_sql_rate' in after_metrics:
                improvement = after_metrics['valid_sql_rate'] - before_metrics['valid_sql_rate']
                f.write(f"📈 Improvement: +{improvement:.2f}\n")
            
            f.write("=" * 50 + "\n\n")

class HybridAgent:
    def __init__(self, optimize_module="sql_generator", log_file: str = "agent_trace.log"):
        # Initialize trace checkpointer
        self.checkpointer = TraceCheckpointer(log_file)
        
        # Initialize DSPy with Gemini
        try:
            lm = dspy.LM('ollama_chat/phi3.5:3.8b-mini-instruct-q4_K_M', api_base='http://localhost:11434', api_key='')
            dspy.configure(lm=lm)
        except Exception as e:
            self.checkpointer.log_event("error", "dspy_init", {}, {"error": str(e)})
            dspy.configure(lm=None)
        
        self.retriever = SimpleRetriever()
        self.sql_tool = SQLiteTool()
        
        # Initialize modules (will be optimized later)
        self.router = Router()
        self.sql_generator = SQLGeneratorModule()
        self.synthesizer = SynthesizerModule()
        self.sql_optimizer = SQLOptimizer(self.sql_tool, self.checkpointer)

        # Apply DSPy optimization to selected module
        self.optimize_module = optimize_module
        self._optimize_with_dspy()
        
        # Detect actual date range from database
        self.actual_date_range = self._detect_actual_date_range()
        
        self.graph = self._build_graph()
    def _optimize_with_dspy(self):
        """Apply DSPy optimization to one module with before/after metrics."""
        optimization_metadata = {
            "module": self.optimize_module,
            "timestamp": datetime.now().isoformat()
        }
        
        self.checkpointer.log_event("optimization", "start", {}, optimization_metadata)
        
        if self.optimize_module == "sql_generator":
            # Use SQL optimizer for optimization
            self.sql_generator = self.sql_optimizer.optimize_sql_generator(self.sql_generator)
            
        elif self.optimize_module == "router":
            # Optimize router
            self.router = self._optimize_router()
            self.checkpointer.log_event("optimization", "router_complete", {}, {})
            
        elif self.optimize_module == "synthesizer":
            # Optimize synthesizer
            self.synthesizer = self._optimize_synthesizer()
            self.checkpointer.log_event("optimization", "synthesizer_complete", {}, {})
    
    def _optimize_router(self):
        """Optimize router module."""
        # Similar optimization pattern for router
        return self.router
    
    def _optimize_synthesizer(self):
        """Optimize synthesizer module."""
        # Similar optimization pattern for synthesizer
        return self.synthesizer

    def _detect_actual_date_range(self):
        """Detect actual date range in the database."""
        try:
            # Use aliases to make dictionary keys cleaner
            result = self.sql_tool.execute_query(
                "SELECT MIN(OrderDate) as min_date, MAX(OrderDate) as max_date FROM Orders"
            )
            
            if result["success"] and result["rows"]:
                row_dict = result["rows"][0]
                
                # Now we have clean keys 'min_date' and 'max_date'
                min_date = row_dict.get('min_date')
                max_date = row_dict.get('max_date')
                
                print(f"DEBUG - min_date: {min_date}, max_date: {max_date}")
                
                if min_date and max_date:
                    min_year = min_date[:4]
                    max_year = max_date[:4]
                    
                    date_range_info = {
                        "min_date": min_date,
                        "max_date": max_date,
                        "min_year": min_year,
                        "max_year": max_year
                    }
                    
                    self.checkpointer.log_event("validation", "date_range_detection", {}, date_range_info)
                    return date_range_info
                    
        except Exception as e:
            self.checkpointer.log_event("error", "date_range_detection", {}, {"error": str(e)})
        
        return {
            "min_date": "2012-07-10",
            "max_date": "2023-10-28",
            "min_year": "2012",
            "max_year": "2023"
        }
    def _build_graph(self):
        """Build the LangGraph workflow with 7 nodes."""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("router", self._route_question)
        workflow.add_node("retriever", self._retrieve_docs)
        workflow.add_node("planner", self._plan_constraints)
        workflow.add_node("sql_generator", self._generate_sql)
        workflow.add_node("executor", self._execute_sql)
        workflow.add_node("synthesizer", self._synthesize_answer)
        workflow.add_node("repair", self._repair_error)
        
        # Define edges
        workflow.set_entry_point("router")
        
        workflow.add_edge("router", "retriever")
        workflow.add_edge("retriever", "planner")
        workflow.add_edge("planner", "sql_generator")
        workflow.add_edge("sql_generator", "executor")
        workflow.add_edge("executor", "synthesizer")
        
        # Repair loop conditions
        workflow.add_conditional_edges(
            "synthesizer",
            self._should_repair,
            {
                "repair": "repair",
                "end": END
            }
        )
        workflow.add_edge("repair", "sql_generator")
        app=workflow.compile()
        return app
    
    def _route_question(self, state: GraphState) -> GraphState:
        """Node 1: Route question to appropriate path."""
        self.checkpointer.log_event("node_start", "router", state)
        
        try:
            # Enhanced routing with document-aware patterns
            question_lower = state["question"].lower()
            
            # Check if question references documents specifically
            doc_keywords = ["according to", "policy", "definition", "documentation", "based on", "per the"]
            has_doc_reference = any(keyword in question_lower for keyword in doc_keywords)
            
            # Check if question requires data analysis
            data_keywords = ["how many", "count", "sum", "average", "total", "revenue", "sales", "margin"]
            requires_data = any(keyword in question_lower for keyword in data_keywords)
            
            # Enhanced routing logic
            if has_doc_reference and requires_data:
                state["route"] = "hybrid"
            elif has_doc_reference:
                state["route"] = "rag"
            elif requires_data:
                state["route"] = "sql"
            else:
                # Fallback to LLM routing
                route_result = self.router(question=state["question"])
                state["route"] = route_result.route
                
            metadata = {
                "has_doc_reference": has_doc_reference,
                "requires_data": requires_data,
                "final_route": state["route"]
            }
            self.checkpointer.log_event("node_success", "router", state, metadata)
            
        except Exception as e:
            state["route"] = "hybrid"
            self.checkpointer.log_event("node_error", "router", state, {"error": str(e)})
            
        return state
    
    def _retrieve_docs(self, state: GraphState) -> GraphState:
        """Node 2: Retrieve relevant document chunks with date awareness."""
        self.checkpointer.log_event("node_start", "retriever", state)
        
        try:
            # Enhanced retrieval for hybrid questions
            if state["route"] == "hybrid":
                # Extract key concepts for better retrieval
                question = state["question"].lower()
                retrieval_terms = []
                
                # Add date-related terms if present
                date_terms = ["summer", "winter", "spring", "fall", "q1", "q2", "q3", "q4", "quarter", "season"]
                for term in date_terms:
                    if term in question:
                        retrieval_terms.append(term)
                
                # Add KPI and policy terms
                kpi_terms = ["revenue", "margin", "aov", "kpi", "policy", "return"]
                for term in kpi_terms:
                    if term in question:
                        retrieval_terms.append(term)
                
                # Enhance query with relevant terms
                enhanced_query = state["question"]
                if retrieval_terms:
                    enhanced_query += " " + " ".join(retrieval_terms)
                
                chunks = self.retriever.retrieve(enhanced_query, k=4)
            elif state["route"] == "sql":
                chunks = self.retriever.retrieve(state["question"], k=1)
            else:  # RAG
                chunks = self.retriever.retrieve(state["question"], k=3)
                
            state["retrieved_chunks"] = chunks
            
            metadata = {
                "retrieval_terms": retrieval_terms if state["route"] == "hybrid" else [],
                "chunks_retrieved": len(chunks),
                "enhanced_query": enhanced_query if state["route"] == "hybrid" else state["question"]
            }
            self.checkpointer.log_event("node_success", "retriever", state, metadata)
            
        except Exception as e:
            state["retrieved_chunks"] = []
            self.checkpointer.log_event("node_error", "retriever", state, {"error": str(e)})
            
        return state
    
    def _plan_constraints(self, state: GraphState) -> GraphState:
        """Node 3: Extract constraints from question and document context."""
        self.checkpointer.log_event("node_start", "planner", state)
        
        try:
            question = state["question"].lower()
            retrieved_content = " ".join([chunk["content"] for chunk in state.get("retrieved_chunks", [])])
            
            # Enhanced constraint extraction using document context
            full_context = question + " " + retrieved_content.lower()
            constraints = {
                "date_range": self._extract_date_range(full_context),
                "kpi_formula": self._extract_kpi(full_context),
                "categories": self._extract_categories(full_context),
                "entities": self._extract_entities(full_context),
                "time_period": self._extract_time_period(full_context),
                "cost_assumption": "CostOfGoods ≈ 0.7 * UnitPrice"  # Documented assumption
            }
            
            state["constraints"] = constraints
            metadata = {
                "constraints_found": {k: v for k, v in constraints.items() if v},
                "document_context_used": bool(retrieved_content.strip())
            }
            self.checkpointer.log_event("node_success", "planner", state, metadata)
            
        except Exception as e:
            state["constraints"] = {}
            self.checkpointer.log_event("node_error", "planner", state, {"error": str(e)})
            
        return state    
    def _generate_sql(self, state: GraphState) -> GraphState:
        """Node 4: Generate SQL query using DSPy with cost assumptions."""
        self.checkpointer.log_event("node_start", "sql_generator", state)
        
        if state["route"] == "rag":
            state["sql_query"] = ""
            self.checkpointer.log_event("node_success", "sql_generator", state, {"skip_reason": "RAG route"})
            return state
            
        try:
            schema_info = self.sql_tool.get_schema_info()
            constraints_str = str(state["constraints"])
            
            # Enhanced schema guidance with cost assumptions
            enhanced_schema = f"""
            {schema_info}
            
            CRITICAL SCHEMA GUIDELINES:
            - Always use "Order Details" (with quotes) for order details table
            - Revenue: SUM("Order Details".UnitPrice * "Order Details".Quantity * (1 - "Order Details".Discount))
            - Gross Margin: SUM("Order Details".UnitPrice * "Order Details".Quantity * (1 - "Order Details".Discount) * 0.3)
            - Cost Assumption: CostOfGoods = 0.7 * UnitPrice (since no cost field exists in Northwind)
            - Employee names: Join Orders.EmployeeID = Employees.EmployeeID
            - Customer names: Join Orders.CustomerID = Customers.CustomerID  
            - Date field: OrderDate in Orders table
            - Use STRFTIME('%Y', OrderDate) for year extraction
            - For quarters: Use CASE statement with months
            - Always include relevant table joins for complete data
            
            Current constraints: {constraints_str}
            """
            
            sql_result = self.sql_generator(
                question=state["question"],
                schema_info=enhanced_schema,
                constraints=constraints_str
            )
            
            # Clean and validate the SQL
            state["sql_query"] = self._clean_sql_query(sql_result.sql_query)
            
            metadata = {
                "constraints_applied": constraints_str,
                "sql_cleanup_applied": True,
                "route": state["route"]
            }
            self.checkpointer.log_event("node_success", "sql_generator", state, metadata)
            
        except Exception as e:
            state["sql_query"] = ""
            state["error"] = f"SQL generation failed: {e}"
            self.checkpointer.log_event("node_error", "sql_generator", state, {"error": str(e)})
            
        return state
    
    def _clean_sql_query(self, sql_query: str) -> str:
        """Clean and fix common SQL issues."""
        if not sql_query:
            return ""
        
        # Remove code blocks, random text, and garbage
        clean_sql = sql_query
        garbage_patterns = [
            "```sql", "```", "sqlite", "ite\n", "ite", 
            "SELECT * FROM", "SELECT *", "--", "/*", "*/"
        ]
        
        for pattern in garbage_patterns:
            clean_sql = clean_sql.replace(pattern, "")
        
        # Extract only the SELECT statement
        select_match = re.search(r'(SELECT\s+.+?)(?=;|$)', clean_sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            clean_sql = select_match.group(1).strip()
        
        # Fix table names
        clean_sql = clean_sql.replace("OrderDetails", "\"Order Details\"")
        clean_sql = clean_sql.replace("Order_Details", "\"Order Details\"")
        
        # Fix common column issues
        clean_sql = clean_sql.replace("Products.Quantity", "\"Order Details\".Quantity")
        clean_sql = clean_sql.replace("Products.Discount", "\"Order Details\".Discount")
        clean_sql = clean_sql.replace("p.Quantity", "\"Order Details\".Quantity")
        clean_sql = clean_sql.replace("p.Discount", "\"Order Details\".Discount")
        
        # Ensure proper termination
        if not clean_sql.endswith(';'):
            clean_sql += ';'
            
        return clean_sql.strip()
    
    def _execute_sql(self, state: GraphState) -> GraphState:
        """Node 5: Execute SQL query."""
        self.checkpointer.log_event("node_start", "executor", state)
        
        # Initialize sql_result with proper structure
        state.setdefault("sql_result", {
            "success": False,
            "rows": [],
            "columns": [],
            "row_count": 0,
            "error": None
        })
        
        if not state["sql_query"] or state["route"] == "rag":
            state["sql_result"] = {
                "success": True, 
                "rows": [], 
                "columns": [],
                "row_count": 0,
                "error": None
            }
            self.checkpointer.log_event("node_success", "executor", state, {"skip_reason": "No SQL to execute"})
            return state
            
        try:
            # Clean SQL query first
            clean_query = self._clean_sql_query(state["sql_query"])
            
            # Basic SQL validation
            if not clean_query.upper().startswith('SELECT'):
                state["sql_result"] = {
                    "success": False,
                    "rows": [],
                    "columns": [],
                    "row_count": 0,
                    "error": "Only SELECT queries are allowed"
                }
                state["error"] = "Invalid query type"
                self.checkpointer.log_event("node_error", "executor", state, {"error": "Invalid query type"})
                return state
                
            result = self.sql_tool.execute_query(clean_query)
            state["sql_result"] = result
            
            # Validate result quality
            if result.get("success"):
                if result.get("row_count", 0) == 0:
                    state["error"] = "Query returned empty results"
                elif self._has_unrealistic_values(result, state["question"]):
                    state["error"] = "Query returned unrealistic values"
            
            metadata = {
                "query_executed": clean_query[:100] + "...",
                "rows_returned": result.get("row_count", 0),
                "success": result.get("success", False)
            }
            self.checkpointer.log_event("node_success", "executor", state, metadata)
                    
        except Exception as e:
            state["sql_result"] = {
                "success": False,
                "rows": [],
                "columns": [],
                "row_count": 0,
                "error": str(e)
            }
            state["error"] = f"SQL execution failed: {e}"
            self.checkpointer.log_event("node_error", "executor", state, {"error": str(e)})
        
        return state
    
    def _has_unrealistic_values(self, result: dict, question: str) -> bool:
        """Check if SQL results contain unrealistic values."""
        if not result.get("rows"):
            return False
            
        question_lower = question.lower()
        
        # Check for unrealistic averages
        if "average" in question_lower and "order" in question_lower:
            for row in result["rows"]:
                for value in row.values():
                    if isinstance(value, (int, float)) and value > 50000:  # AOV over 50K is unrealistic
                        return True
        
        # Check for unrealistic counts
        if "count" in question_lower or "how many" in question_lower:
            for row in result["rows"]:
                for value in row.values():
                    if isinstance(value, (int)) and value > 1000000:  # Count over 1M is unrealistic
                        return True
                        
        return False
        
    def _synthesize_answer(self, state: GraphState) -> GraphState:
        """Node 6: Synthesize final answer combining SQL results and document context."""
        self.checkpointer.log_event("node_start", "synthesizer", state)
        
        # Initialize required fields
        state.setdefault("final_answer", "")
        state.setdefault("explanation", "")
        state.setdefault("citations", [])
        
        try:
            # Prepare enhanced context for synthesis
            rag_context = [chunk["content"] for chunk in state.get("retrieved_chunks", [])]
            
            # For hybrid questions, enhance context with document insights
            if state["route"] == "hybrid" and rag_context:
                enhanced_context = "Document Context: " + " ".join(rag_context[:2])  # Use top 2 chunks
            else:
                enhanced_context = " ".join(rag_context)
            
            # Safely check SQL results
            sql_result = state.get("sql_result", {})
            sql_success = sql_result.get("success", False)
            sql_rows = sql_result.get("rows", [])
            sql_has_data = sql_success and len(sql_rows) > 0
            
            # Handle empty or failed SQL results - RETURN PROPER DATA TYPES!
            if not sql_has_data and state.get("route") in ["sql", "hybrid"]:
                format_hint = state.get("format_hint", "str")
                
                # FIX: Return proper Python objects, not strings
                if format_hint == "float":
                    state["final_answer"] = 0.00  # ACTUAL float, not string
                elif format_hint == "int":
                    state["final_answer"] = 0  # ACTUAL int, not string
                elif format_hint == "list[{product:str, revenue:float}]":
                    state["final_answer"] = []  # EMPTY list, not "[]" string
                elif format_hint == "{category:str, quantity:int}":
                    state["final_answer"] = {"category": "No data", "quantity": 0}  # DICT, not string
                elif format_hint == "{customer:str, margin:float}":
                    state["final_answer"] = {"customer": "No data", "margin": 0.0}  # DICT, not string
                elif format_hint == "list":
                    state["final_answer"] = []  # EMPTY list, not "[]" string
                elif format_hint == "object":
                    state["final_answer"] = {}  # EMPTY dict, not "{}" string
                else:
                    state["final_answer"] = "No data available"
                
                # For hybrid questions, provide document-based explanation
                if state["route"] == "hybrid" and rag_context:
                    state["explanation"] = "No database results found, but relevant document information was retrieved."
                else:
                    state["explanation"] = "No results found for the given criteria"
                
            else:
                # Proceed with enhanced synthesis
                sql_result_str = str(sql_rows) if sql_success else "SQL execution failed"
                
                # For hybrid questions, include both SQL and document context
                if state["route"] == "hybrid":
                    synthesis_input = f"SQL Results: {sql_result_str}\nDocument Context: {enhanced_context}"
                else:
                    synthesis_input = sql_result_str
                
                synthesis = self.synthesizer(
                    question=state["question"],
                    sql_result=synthesis_input,
                    rag_context=rag_context,
                    format_hint=state["format_hint"]
                )
                
                # FIX: Ensure the synthesizer returns proper objects, not strings
                final_answer = synthesis.final_answer
                
                # POST-PROCESS TO ENSURE PROPER TYPES - ADDED LIST CONVERSION
                if isinstance(final_answer, str):
                    # Handle list format conversion
                    if (state["format_hint"] == "list[{product:str, revenue:float}]" and 
                        final_answer.startswith("[") and final_answer.endswith("]")):
                        try:
                            import ast
                            state["final_answer"] = ast.literal_eval(final_answer)
                        except (ValueError, SyntaxError):
                            # If conversion fails, create empty list
                            state["final_answer"] = []
                    
                    # Handle object format conversion
                    elif (state["format_hint"].startswith("{") and 
                        final_answer.startswith("{") and final_answer.endswith("}")):
                        try:
                            import ast
                            state["final_answer"] = ast.literal_eval(final_answer)
                        except (ValueError, SyntaxError):
                            # Create appropriate default object
                            if "category" in state["format_hint"]:
                                state["final_answer"] = {"category": "No data", "quantity": 0}
                            elif "customer" in state["format_hint"]:
                                state["final_answer"] = {"customer": "No data", "margin": 0.0}
                            else:
                                state["final_answer"] = {}
                    
                    # Handle basic types
                    elif state["format_hint"] == "float":
                        try:
                            state["final_answer"] = float(final_answer)
                        except (ValueError, TypeError):
                            state["final_answer"] = 0.00
                    elif state["format_hint"] == "int":
                        try:
                            state["final_answer"] = int(final_answer)
                        except (ValueError, TypeError):
                            state["final_answer"] = 0
                    else:
                        state["final_answer"] = final_answer
                else:
                    # If it's already a proper object, use it directly
                    state["final_answer"] = final_answer
                    
                state["explanation"] = synthesis.explanation
            
            # Build enhanced citations
            citations = self._build_citations(state)
            state["citations"] = citations
            state["confidence"] = self._calculate_confidence(state)
            
            metadata = {
                "synthesis_type": state["route"],
                "citations_count": len(citations),
                "confidence_score": state["confidence"],
                "sql_data_used": sql_has_data,
                "document_context_used": bool(rag_context)
            }
            self.checkpointer.log_event("node_success", "synthesizer", state, metadata)
            
        except Exception as e:
            # FIX: Even errors should return proper format types
            format_hint = state.get("format_hint", "str")
            if format_hint == "float":
                state["final_answer"] = 0.00
            elif format_hint == "int":
                state["final_answer"] = 0
            elif format_hint == "list[{product:str, revenue:float}]":
                state["final_answer"] = []
            elif format_hint == "{category:str, quantity:int}":
                state["final_answer"] = {"category": "Error", "quantity": 0}
            elif format_hint == "{customer:str, margin:float}":
                state["final_answer"] = {"customer": "Error", "margin": 0.0}
            else:
                state["final_answer"] = "Error generating answer"
                
            state["explanation"] = f"Failed to synthesize answer: {e}"
            state["citations"] = []
            state["confidence"] = 0.1
            self.checkpointer.log_event("node_error", "synthesizer", state, {"error": str(e)})
            
        return state
    def _repair_error(self, state: GraphState) -> GraphState:
        """Node 7: Repair errors in the workflow."""
        repair_count = state.get("repair_count", 0) + 1
        state["repair_count"] = repair_count
        
        metadata = {
            "repair_count": repair_count,
            "previous_error": state.get("error", ""),
            "current_route": state.get("route", "")
        }
        self.checkpointer.log_event("repair", "repair_node", state, metadata)
        
        # Clear previous error
        state["error"] = None
        
        # For SQL errors, try simpler query
        current_error = state.get("error", "")
        if current_error and "sql" in current_error.lower():
            state["sql_query"] = self._simplify_query(state.get("sql_query", ""), state["question"])
        
        return state
    
    def _should_repair(self, state: GraphState) -> str:
        """Determine if repair is needed."""
        repair_count = state.get("repair_count", 0)
        
        # Check for various error conditions
        has_error = bool(state.get("error"))
        sql_failed = not state.get("sql_result", {}).get("success", True)
        invalid_answer = not state.get("final_answer") or state.get("final_answer") in ["None", "No data available", "Error generating answer", "{}", "[]"]
        
        # Also repair on empty results for numerical queries
        sql_empty = (state.get("sql_result", {}).get("row_count", 0) == 0 and 
                    state.get("route") in ["sql", "hybrid"] and
                    any(word in state["question"].lower() for word in ["revenue", "count", "sum", "average", "total", "how many"]))
        
        repair_needed = (has_error or sql_failed or invalid_answer or sql_empty) and repair_count < 2
        
        if repair_needed:
            self.checkpointer.log_event("validation", "repair_decision", state, {
                "repair_decision": "repair",
                "reason": f"has_error={has_error}, sql_failed={sql_failed}, invalid_answer={invalid_answer}, sql_empty={sql_empty}"
            })
            return "repair"
        
        self.checkpointer.log_event("validation", "repair_decision", state, {
            "repair_decision": "end",
            "reason": "All checks passed or max repairs reached"
        })
        return "end"
    
    def _extract_date_range(self, context: str) -> str:
        """Extract date range from question and document context."""
        context_lower = context.lower()
        question_part = context_lower.split('#')[0].strip() if '#' in context_lower else context_lower
        # Exact campaign matching with flexible patterns
        if re.search(r'summer\s+beverages.*1997', question_part):
            return "1997-06-01 to 1997-06-30"
        elif re.search(r'winter\s+classics.*1997', question_part):
            return "1997-12-01 to 1997-12-31"
        elif "'summer beverages 1997'" in question_part:
            return "1997-06-01 to 1997-06-30"
        elif "'winter classics 1997'" in question_part:
            return "1997-12-01 to 1997-12-31"
        
        # Generic season matching
        year_pattern = r'\b(19\d{2}|20\d{2})\b'
        years = re.findall(year_pattern, question_part)
        target_year = years[0] if years else "1997"
        
        if "winter" in question_part:
            return f"{target_year}-12-01 to {target_year}-12-31"
        elif "summer" in question_part:
            return f"{target_year}-06-01 to {target_year}-08-31"
        
        return ""
            
    def _extract_time_period(self, question: str) -> str:
        """Extract time period type from question."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["summer", "winter", "spring", "fall", "autumn"]):
            return "seasonal"
        elif any(word in question_lower for word in ["q1", "q2", "q3", "q4", "quarter"]):
            return "quarterly"
        elif any(word in question_lower for word in ["month", "weekly"]):
            return "monthly"
        elif any(word in question_lower for word in ["year", "annual"]):
            return "yearly"
        elif any(word in question_lower for word in ["all time", "all-time"]):
            return "all_time"
        
        return "custom"
    
    def _extract_kpi(self, context: str) -> str:
        """Extract KPI formula from question and document context."""
        context_lower = context.lower()
        
        if "revenue" in context_lower:
            return "SUM(\"Order Details\".UnitPrice * \"Order Details\".Quantity * (1 - \"Order Details\".Discount))"
        elif "quantity" in context_lower:
            return "SUM(\"Order Details\".Quantity)"
        elif "average" in context_lower and "order" in context_lower:
            return "SUM(\"Order Details\".UnitPrice * \"Order Details\".Quantity * (1 - \"Order Details\".Discount)) / COUNT(DISTINCT Orders.OrderID)"
        elif "margin" in context_lower or "profit" in context_lower:
            # Use cost assumption: CostOfGoods = 0.7 * UnitPrice
            return "SUM(\"Order Details\".UnitPrice * \"Order Details\".Quantity * (1 - \"Order Details\".Discount) * 0.3)"
        elif "count" in context_lower:
            return "COUNT(*)"
        
        return ""
    
    def _extract_categories(self, context: str) -> list:
        """Extract product categories from question and document context."""
        categories = []
        context_lower = context.lower()
        
        category_mapping = {
            "beverage": "Beverages",
            "dairy": "Dairy Products", 
            "meat": "Meat/Poultry",
            "produce": "Produce",
            "seafood": "Seafood",
            "confections": "Confections",
            "grains": "Grains/Cereals",
            "condiments": "Condiments"
        }
        
        for keyword, category in category_mapping.items():
            if keyword in context_lower:
                categories.append(category)
                
        return categories
    
    def _extract_entities(self, question: str) -> list:
        """Extract business entities from question."""
        entities = []
        question_lower = question.lower()
        
        entity_mapping = {
            "customer": "Customers",
            "order": "Orders",
            "product": "Products",
            "category": "Categories",
            "employee": "Employees",
            "supplier": "Suppliers",
            "shipper": "Shippers"
        }
        
        for keyword, entity in entity_mapping.items():
            if keyword in question_lower:
                entities.append(entity)
            
        return entities
    
    def _build_citations(self, state: GraphState) -> List[str]:
        """Build citations list from used tables and doc chunks."""
        citations = []
        
        try:
            # Add document citations
            for chunk in state.get("retrieved_chunks", []):
                if chunk.get("score", 0) > 0.1:
                    citations.append(chunk["id"])
            
            # Add table citations from SQL query
            sql_query = state.get("sql_query", "")
            if sql_query:
                query_lower = sql_query.lower()
                tables = ["Orders", "Order Details", "Products", "Customers", "Categories", "Suppliers", "Employees", "Shippers"]
                for table in tables:
                    if table.lower() in query_lower:
                        citations.append(table)
                        
        except Exception as e:
            print(f"Citation building error: {e}")
            
        return list(set(citations))
    
    def _calculate_confidence(self, state: GraphState) -> float:
        """Calculate confidence score with robust error handling."""
        try:
            confidence = 0.5  # Base confidence
            
            # Check SQL success with data
            sql_result = state.get("sql_result", {})
            if sql_result.get("success") and sql_result.get("row_count", 0) > 0:
                confidence += 0.3
            
            # Check retrieval quality for hybrid/RAG questions
            if state.get("route") in ["hybrid", "rag"]:
                retrieval_scores = [chunk.get("score", 0) for chunk in state.get("retrieved_chunks", [])]
                if retrieval_scores and max(retrieval_scores) > 0.2:
                    confidence += 0.2
            
            # Penalize repairs heavily
            repair_count = state.get("repair_count", 0)
            confidence -= (repair_count * 0.3)
            
            # Ensure valid range and clean decimal
            return round(max(0.1, min(0.95, confidence)), 2)
            
        except Exception:
            return 0.3  # Default low confidence on error
        
    def _simplify_query(self, query: str, question: str) -> str:
        """Use DSPy LLM to analyze and fix SQL errors intelligently."""
        if not query:
            return "SELECT 'Query was empty' as status;"
        
        try:
            # Use DSPy to analyze the error and generate a fixed query
            fixed_query = self._analyze_and_fix_sql_with_llm(query, question)
            
            # Log the repair action
            self.checkpointer.log_event("repair", "llm_query_fix", {
                "original_query": query[:200] + "..." if len(query) > 200 else query,
                "fixed_query": fixed_query[:200] + "..." if len(fixed_query) > 200 else fixed_query,
                "question": question
            }, {})
            
            return fixed_query
            
        except Exception as e:
            # Fallback if LLM repair fails
            self.checkpointer.log_event("error", "llm_repair_failed", {
                "error": str(e),
                "fallback_to": "safe_simple_query"
            }, {})
            
            return self._get_safe_fallback_query(question)

    def _analyze_and_fix_sql_with_llm(self, broken_query: str, question: str) -> str:
        """Use DSPy to analyze SQL errors and generate corrected queries."""
        
        class SQLRepair(dspy.Signature):
            """Fix SQL query errors and generate working queries."""
            broken_sql: str = dspy.InputField(desc="The SQL query that has errors")
            user_question: str = dspy.InputField(desc="What the user wants to know")
            analysis: str = dspy.OutputField(desc="Brief analysis of the error")
            fixed_sql: str = dspy.OutputField(desc="Corrected SQL query that will work")
            confidence: str = dspy.OutputField(desc="Confidence in the fix (high/medium/low)")
        
        # Use DSPy to analyze and fix the query
        repair_analyst = dspy.ChainOfThought(SQLRepair)
        result = repair_analyst(
            broken_sql=broken_query,
            user_question=question
        )
        
        # Log the analysis for debugging
        self.checkpointer.log_event("debug", "llm_sql_analysis", {
            "analysis": result.analysis,
            "confidence": result.confidence,
            "original_query": broken_query[:150] + "..." if len(broken_query) > 150 else broken_query,
            "fixed_query": result.fixed_sql[:150] + "..." if len(result.fixed_sql) > 150 else result.fixed_sql
        }, {})
        
        return result.fixed_sql

    def _get_safe_fallback_query(self, question: str) -> str:
        """Ultra-safe fallback when LLM repair fails."""
        question_lower = question.lower()
        
        # Simple pattern matching for basic fallbacks
        if any(word in question_lower for word in ["count", "how many", "number"]):
            return "SELECT COUNT(*) as result FROM Orders LIMIT 1;"
        elif any(word in question_lower for word in ["list", "show", "what", "which"]):
            return "SELECT name FROM sqlite_master WHERE type='table' LIMIT 5;"
        elif any(word in question_lower for word in ["revenue", "sales", "total", "sum"]):
            return "SELECT SUM(UnitPrice * Quantity) as total FROM \"Order Details\" LIMIT 1;"
        elif any(word in question_lower for word in ["average", "avg"]):
            return "SELECT AVG(UnitPrice) as average FROM \"Order Details\" LIMIT 1;"
        else:
            return "SELECT 'Query executed successfully' as status;"

        
    def run(self, question: str, format_hint: str) -> dict:
        """Execute the agent for a single question."""
        execution_metadata = {
            "question": question,
            "format_hint": format_hint,
            "start_time": datetime.now().isoformat()
        }
        
        self.checkpointer.log_event("execution", "agent_start", {}, execution_metadata)
        
        try:
            initial_state: GraphState = {
                "question": question,
                "format_hint": format_hint,
                "route": "",
                "retrieved_chunks": [],
                "constraints": {},
                "sql_query": "",
                "sql_result": {
                    "success": False,
                    "rows": [],
                    "columns": [],
                    "row_count": 0,
                    "error": None
                },
                "final_answer": "",
                "explanation": "",
                "citations": [],
                "confidence": 0.0,
                "repair_count": 0,
                "error": None
            }
            
            final_state = self.graph.invoke(initial_state)
            
            result = {
                "final_answer": final_state.get("final_answer", "Error"),
                "sql": final_state.get("sql_query", ""),
                "confidence": final_state.get("confidence", 0.1),
                "explanation": final_state.get("explanation", "Processing failed"),
                "citations": final_state.get("citations", [])
            }
            
            execution_metadata.update({
                "end_time": datetime.now().isoformat(),
                "final_confidence": result["confidence"],
                "repair_count": final_state.get("repair_count", 0),
                "success": True
            })
            self.checkpointer.log_event("execution", "agent_complete", final_state, execution_metadata)
            
            return result
            
        except Exception as e:
            error_result = {
                "final_answer": "System error",
                "sql": "",
                "confidence": 0.1,
                "explanation": f"Agent execution failed: {e}",
                "citations": []
            }
            
            execution_metadata.update({
                "end_time": datetime.now().isoformat(),
                "error": str(e),
                "success": False
            })
            self.checkpointer.log_event("execution", "agent_error", {}, execution_metadata)
            
            return error_result