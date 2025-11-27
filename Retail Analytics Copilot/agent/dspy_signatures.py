import dspy
from typing import Literal
from dspy.teleprompt import BootstrapFewShot

class RouteClassification(dspy.Signature):
    """Classify whether question requires SQL, RAG, or both."""
    question: str = dspy.InputField()
    route: Literal["sql", "rag", "hybrid"] = dspy.OutputField(description="Use 'sql' for numerical queries, 'rag' for policy questions, 'hybrid' for both")

class QueryPlanner(dspy.Signature):
    """Extract constraints and requirements from the question."""
    question: str = dspy.InputField()
    retrieved_context: list = dspy.InputField()
    date_range: str = dspy.OutputField(description="Extracted date range if any")
    kpi_formula: str = dspy.OutputField(description="KPI formula needed")
    categories: list = dspy.OutputField(description="Product categories mentioned")
    entities: list = dspy.OutputField(description="Business entities mentioned")

class SQLGenerator(dspy.Signature):
    """Generate SQLite query based on question and schema."""
    question: str = dspy.InputField()
    schema_info: str = dspy.InputField()
    constraints: str = dspy.InputField()
    sql_query: str = dspy.OutputField(description="Valid SQLite query")

class AnswerSynthesizer(dspy.Signature):
    """Synthesize final answer with proper formatting."""
    question: str = dspy.InputField()
    sql_result: str = dspy.InputField()
    rag_context: list = dspy.InputField()
    format_hint: str = dspy.InputField()
    final_answer: str = dspy.OutputField(description="Answer matching format_hint exactly")
    explanation: str = dspy.OutputField(description="Brief explanation <= 2 sentences")

class Router(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classifier = dspy.ChainOfThought(RouteClassification)
    
    def forward(self, question):
        return self.classifier(question=question)

class SQLGeneratorModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(SQLGenerator)
    
    def forward(self, question, schema_info, constraints):
        return self.generator(
            question=question,
            schema_info=schema_info,
            constraints=constraints
        )

class SynthesizerModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.synthesizer = dspy.ChainOfThought(AnswerSynthesizer)
    
    def forward(self, question, sql_result, rag_context, format_hint):
        return self.synthesizer(
            question=question,
            sql_result=sql_result,
            rag_context=rag_context,
            format_hint=format_hint
        )
class SQLOptimizer:
    def __init__(self, sql_tool, checkpointer):
        self.sql_tool = sql_tool
        self.checkpointer = checkpointer
    
    def evaluate_sql_generator(self, sql_generator):
        """Evaluate SQL generator with sample questions."""
        test_cases = [
            {
                "question": "What are the top 3 products by revenue?",
                "expected_tables": ["Products", "Order Details"],
                "expected_aggregation": "SUM"
            },
            {
                "question": "How many orders were placed in 2023?",
                "expected_tables": ["Orders"],
                "expected_aggregation": "COUNT"
            },
            {
                "question": "What is the average order value?",
                "expected_tables": ["Orders", "Order Details"],
                "expected_aggregation": "AVG"
            },
            {
                "question": "Which employee has the highest sales?",
                "expected_tables": ["Employees", "Orders", "Order Details"],
                "expected_aggregation": "SUM"
            }
        ]
        
        valid_count = 0
        for test_case in test_cases:
            try:
                schema_info = self.sql_tool.get_schema_info()
                result = sql_generator(
                    question=test_case["question"],
                    schema_info=schema_info,
                    constraints="{}"
                )
                
                sql_query = result.sql_query
                # Basic validation
                if (sql_query and 
                    sql_query.upper().startswith("SELECT") and
                    all(table.lower() in sql_query.lower() for table in test_case["expected_tables"]) and
                    test_case["expected_aggregation"].upper() in sql_query.upper()):
                    valid_count += 1
                    
            except Exception:
                pass
        
        return {"valid_sql_rate": valid_count / len(test_cases)}
    
    def optimize_sql_generator(self, sql_generator):
        """Optimize SQL generator using BootstrapFewShot with before/after metrics."""
        
        # Before optimization metrics
        before_metrics = self.evaluate_sql_generator(sql_generator)
        
        # Training examples for SQL generation
        trainset = [
            dspy.Example(
                question="What are the top 3 products by revenue all time?",
                schema_info="Products(ProductID, ProductName), Order Details(OrderID, ProductID, UnitPrice, Quantity, Discount)",
                constraints="{}",
                sql_query='SELECT P.ProductName, SUM(OD.UnitPrice * OD.Quantity * (1 - OD.Discount)) as revenue FROM Products P JOIN "Order Details" OD ON P.ProductID = OD.ProductID GROUP BY P.ProductName ORDER BY revenue DESC LIMIT 3;'
            ).with_inputs('question', 'schema_info', 'constraints'),
            
            dspy.Example(
                question="How many orders were placed in 2023?",
                schema_info="Orders(OrderID, OrderDate)",
                constraints="{'date_range': '2023-01-01 to 2023-12-31'}",
                sql_query="SELECT COUNT(*) as order_count FROM Orders WHERE STRFTIME('%Y', OrderDate) = '2023';"
            ).with_inputs('question', 'schema_info', 'constraints'),
            
            dspy.Example(
                question="What is the total revenue from Beverages category in 2022?",
                schema_info="Orders(OrderID, OrderDate), Order Details(OrderID, ProductID, UnitPrice, Quantity, Discount), Products(ProductID, ProductName, CategoryID), Categories(CategoryID, CategoryName)",
                constraints="{'date_range': '2022-01-01 to 2022-12-31', 'categories': ['Beverages']}",
                sql_query='SELECT SUM(OD.UnitPrice * OD.Quantity * (1 - OD.Discount)) as revenue FROM Orders O JOIN "Order Details" OD ON O.OrderID = OD.OrderID JOIN Products P ON OD.ProductID = P.ProductID JOIN Categories C ON P.CategoryID = C.CategoryID WHERE C.CategoryName = "Beverages" AND STRFTIME("%Y", O.OrderDate) = "2022";'
            ).with_inputs('question', 'schema_info', 'constraints'),
            
            dspy.Example(
                question="Which employee has the highest gross margin? Assume CostOfGoods is 70% of UnitPrice.",
                schema_info="Employees(EmployeeID, FirstName, LastName), Orders(OrderID, EmployeeID, OrderDate), Order Details(OrderID, ProductID, UnitPrice, Quantity, Discount)",
                constraints="{'kpi_formula': 'gross_margin'}",
                sql_query='SELECT E.FirstName || " " || E.LastName as employee, SUM(OD.UnitPrice * OD.Quantity * (1 - OD.Discount) * 0.3) as gross_margin FROM Orders O JOIN "Order Details" OD ON O.OrderID = OD.OrderID JOIN Employees E ON O.EmployeeID = E.EmployeeID GROUP BY E.EmployeeID ORDER BY gross_margin DESC LIMIT 1;'
            ).with_inputs('question', 'schema_info', 'constraints')
        ]
        
        # Define validation metric
        def validate_sql(example, pred, trace=None):
            sql = pred.sql_query or ""
            # Check for basic SQL validity
            if not sql.strip():
                return False
            if not sql.upper().startswith("SELECT"):
                return False
            if ";" not in sql:
                return False
            return True
        
        # Optimize using BootstrapFewShot
        teleprompter = BootstrapFewShot(metric=validate_sql, max_bootstrapped_demos=2, max_labeled_demos=2)
        optimized_module = teleprompter.compile(SQLGeneratorModule(), trainset=trainset)
        
        # After optimization metrics
        after_metrics = self.evaluate_sql_generator(optimized_module)
        
        # Log optimization results
        self.checkpointer.log_optimization("sql_generator", before_metrics, after_metrics)
        
        return optimized_module    