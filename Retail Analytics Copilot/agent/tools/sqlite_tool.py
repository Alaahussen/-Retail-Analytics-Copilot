import sqlite3
import json
from typing import List, Dict, Any

class SQLiteTool:
    def __init__(self, db_path: str = "data/northwind.sqlite"):
        self.db_path = db_path
    
    def get_schema_info(self) -> str:
        """Get schema information for all tables."""
        schema_info = []
        tables = [
            "Orders", "Order Details", "Products", 
            "Customers", "Categories", "Suppliers"
        ]
        
        with sqlite3.connect(self.db_path) as conn:
            for table in tables:
                try:
                    # Get table schema
                    cursor = conn.execute(f"PRAGMA table_info({table})")
                    columns = cursor.fetchall()
                    schema_info.append(f"Table: {table}")
                    schema_info.append("Columns:")
                    for col in columns:
                        schema_info.append(f"  {col[1]} ({col[2]})")
                    schema_info.append("")
                except sqlite3.Error:
                    continue
        
        return "\n".join(schema_info)
    
    def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute SQL query and return results with metadata."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query)
                
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description] if cursor.description else []
                
                return {
                    "success": True,
                    "columns": columns,
                    "rows": [dict(row) for row in rows],
                    "row_count": len(rows),
                    "error": None
                }
        except sqlite3.Error as e:
            return {
                "success": False,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "error": str(e)
            }
    
    def get_table_names(self) -> List[str]:
        """Get list of all tables in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            return [row[0] for row in cursor.fetchall()]