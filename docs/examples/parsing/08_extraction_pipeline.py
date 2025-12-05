"""
Structured Data Extraction Pipeline
===================================

This example demonstrates a complete real-world workflow combining multiple
parsing techniques to extract, validate, and process structured information
from complex LLM outputs.

Main concepts:
- Multi-stage extraction pipeline
- Combining JSON, code, and text extraction
- Validation at each stage
- Error handling and recovery
- End-to-end data processing workflow
"""

from typing import List, Dict, Any, Optional
from kerb.parsing import (
    extract_json,
    extract_code_blocks,
    extract_xml_tag,
    parse_markdown_table,
    validate_json_schema,
    parse_json,
    ParseMode
)


def simulate_complex_llm_analysis() -> str:
    """Simulate a complex LLM response analyzing a software project."""
    return """# Software Project Analysis Report

# %%
# Setup and Imports
# -----------------

<executive_summary>
The codebase shows good architecture with room for optimization. 
Performance bottlenecks identified in the data processing pipeline.
Security audit passed with minor recommendations.
</executive_summary>

## Code Quality Metrics

Here are the detailed metrics I extracted:

```json
{
  "overall_score": 7.8,
  "total_files": 156,
  "total_lines": 48230,
  "languages": {
    "python": 85,
    "javascript": 45,
    "typescript": 26
  },
  "test_coverage": 76.5
}
```

## Performance Issues Found

<issues>
1. Database query optimization needed in user service
2. Memory leak in background worker process
3. Inefficient loop in data processor
</issues>

### Critical Code Issue

**Location:** src/processors/data_processor.py

```python

# %%
# Process Data
# ------------

def process_data(items):
    results = []
    for item in items:
        if item.status == 'active':
            # Inefficient: Making DB call in loop
            user = db.get_user(item.user_id)
            results.append({
                'item_id': item.id,
                'user_name': user.name,
                'processed': True
            })
    return results
```

<recommendation>
Use batch query to fetch all users at once, reducing database calls
from N queries to 1 query for N items.
</recommendation>

### Recommended Fix

```python

# %%
# Process Data
# ------------

def process_data(items):
    # Batch fetch all users
    user_ids = [item.user_id for item in items if item.status == 'active']
    users = db.get_users_batch(user_ids)
    users_dict = {u.id: u for u in users}
    
    results = []
    for item in items:
        if item.status == 'active':
            user = users_dict.get(item.user_id)
            if user:
                results.append({
                    'item_id': item.id,
                    'user_name': user.name,
                    'processed': True
                })
    return results
```

## Security Findings

| Severity | Issue | Location | Status |
|----------|-------|----------|--------|
| High | SQL Injection Risk | auth/login.py:45 | Fixed |
| Medium | Weak Password Policy | auth/validators.py:12 | Open |
| Low | Missing CORS Headers | api/server.py:8 | Open |
| Low | Debug Mode Enabled | config/settings.py:5 | Open |

## Dependencies Analysis

<dependencies>
{
  "outdated": [
    {"name": "requests", "current": "2.28.0", "latest": "2.31.0"},
    {"name": "django", "current": "3.2.0", "latest": "4.2.0"}
  ],
  "vulnerable": [
    {"name": "pillow", "current": "9.0.0", "cve": "CVE-2023-12345"}
  ],
  "total_dependencies": 45
}
</dependencies>

## Action Items

<action_items>
- Update vulnerable dependencies immediately
- Implement database query batching in data processor
- Strengthen password validation requirements
- Disable debug mode in production configuration
- Add comprehensive API rate limiting
</action_items>

<priority>
HIGH: Security vulnerabilities must be addressed within 48 hours
MEDIUM: Performance optimizations should be completed this sprint
LOW: Code quality improvements can be scheduled for next quarter
</priority>"""


class ProjectAnalysisExtractor:
    """Extract and validate project analysis data from LLM output."""
    

# %%
#   Init  
# --------

    def __init__(self, llm_output: str):
        self.raw_output = llm_output
        self.errors = []
        self.warnings = []
        self.data = {}
    

# %%
# Extract Executive Summary
# -------------------------

    def extract_executive_summary(self) -> Optional[str]:
        """Extract executive summary from XML tag."""
        summaries = extract_xml_tag(self.raw_output, "executive_summary")
        if summaries:
            return summaries[0]
        else:
            self.warnings.append("No executive summary found")
            return None
    
    def extract_metrics(self) -> Optional[Dict[str, Any]]:
        """Extract and validate code quality metrics."""
        result = extract_json(self.raw_output, mode=ParseMode.LENIENT)
        
        if not result.success:
            self.errors.append(f"Failed to extract metrics: {result.error}")
            return None
        
        # Validate metrics schema
        schema = {
            "type": "object",
            "properties": {
                "overall_score": {"type": "number", "minimum": 0, "maximum": 10},
                "total_files": {"type": "integer", "minimum": 0},
                "total_lines": {"type": "integer", "minimum": 0},
                "test_coverage": {"type": "number", "minimum": 0, "maximum": 100}
            },
            "required": ["overall_score", "total_files", "total_lines"]
        }
        
        validation = validate_json_schema(result.data, schema)
        if not validation.valid:
            self.errors.append(f"Metrics validation failed: {validation.errors}")
            return None
        
        return result.data
    

# %%
# Extract Code Issues
# -------------------

    def extract_code_issues(self) -> List[Dict[str, str]]:
        """Extract code blocks showing issues and fixes."""
        code_blocks = extract_code_blocks(self.raw_output, language="python")
        
        issues = []
        for i, block in enumerate(code_blocks):
            issues.append({
                "block_number": i + 1,
                "code": block["code"],
                "lines": len(block["code"].splitlines())
            })
        
        return issues
    
    def extract_security_findings(self) -> List[Dict[str, str]]:
        """Extract security findings from markdown table."""
        table_data = parse_markdown_table(self.raw_output)
        
        if not table_data:
            self.warnings.append("No security findings table found")
            return []
        
        # Filter for security-related tables
        security_findings = []
        for row in table_data:
            if "Severity" in row and "Issue" in row:
                security_findings.append(row)
        
        return security_findings
    

# %%
# Extract Dependencies
# --------------------

    def extract_dependencies(self) -> Optional[Dict[str, Any]]:
        """Extract dependencies analysis."""
        deps = extract_xml_tag(self.raw_output, "dependencies")
        
        if not deps:
            self.warnings.append("No dependencies section found")
            return None
        
        result = parse_json(deps[0], mode=ParseMode.LENIENT)
        if result.success:
            return result.data
        else:
            self.errors.append(f"Failed to parse dependencies: {result.error}")
            return None
    
    def extract_action_items(self) -> List[str]:
        """Extract action items from XML tag."""
        items = extract_xml_tag(self.raw_output, "action_items")
        
        if not items:
            self.warnings.append("No action items found")
            return []
        
        # Split by lines and clean up
        action_items = [
            line.strip().lstrip('-').strip()
            for line in items[0].split('\n')
            if line.strip() and line.strip().startswith('-')
        ]
        
        return action_items
    

# %%
# Extract Priority Info
# ---------------------

    def extract_priority_info(self) -> Optional[str]:
        """Extract priority information."""
        priorities = extract_xml_tag(self.raw_output, "priority")
        if priorities:
            return priorities[0]
        return None
    
    def extract_all(self) -> Dict[str, Any]:
        """Extract all information from the analysis."""
        
        print("Starting extraction pipeline...")
        print("-" * 60)
        
        # Extract each component
        self.data["executive_summary"] = self.extract_executive_summary()
        print(f"[1/7] Executive Summary: {'OK' if self.data['executive_summary'] else 'MISSING'}")
        
        self.data["metrics"] = self.extract_metrics()
        print(f"[2/7] Metrics: {'OK' if self.data['metrics'] else 'FAILED'}")
        
        self.data["code_issues"] = self.extract_code_issues()
        print(f"[3/7] Code Issues: {len(self.data['code_issues'])} blocks found")
        
        self.data["security_findings"] = self.extract_security_findings()
        print(f"[4/7] Security Findings: {len(self.data['security_findings'])} issues found")
        
        self.data["dependencies"] = self.extract_dependencies()
        print(f"[5/7] Dependencies: {'OK' if self.data['dependencies'] else 'MISSING'}")
        
        self.data["action_items"] = self.extract_action_items()
        print(f"[6/7] Action Items: {len(self.data['action_items'])} items found")
        
        self.data["priority_info"] = self.extract_priority_info()
        print(f"[7/7] Priority Info: {'OK' if self.data['priority_info'] else 'MISSING'}")
        
        print("-" * 60)
        print(f"Extraction complete: {len(self.errors)} errors, {len(self.warnings)} warnings")
        
        return self.data



# %%
# Main
# ----

def main():
    """Run structured extraction pipeline example."""
    
    print("="*80)
    print("STRUCTURED DATA EXTRACTION PIPELINE")
    print("="*80)
    
    # Get complex LLM output
    llm_output = simulate_complex_llm_analysis()
    
    print("\nReceived complex LLM analysis report")
    print(f"Total length: {len(llm_output)} characters")
    print()
    
    # Create extractor and run pipeline
    extractor = ProjectAnalysisExtractor(llm_output)
    data = extractor.extract_all()
    
    # Display extracted data
    print("\n\n" + "="*80)
    print("EXTRACTED DATA SUMMARY")
    print("="*80)
    
    # Executive Summary
    if data["executive_summary"]:
        print("\nExecutive Summary:")
        print(f"  {data['executive_summary'][:120]}...")
    
    # Metrics
    if data["metrics"]:
        print("\n\nCode Quality Metrics:")
        print(f"  Overall Score: {data['metrics'].get('overall_score')}/10")
        print(f"  Total Files: {data['metrics'].get('total_files')}")
        print(f"  Total Lines: {data['metrics'].get('total_lines'):,}")
        print(f"  Test Coverage: {data['metrics'].get('test_coverage')}%")
    
    # Code Issues
    if data["code_issues"]:
        print("\n\nCode Issues:")
        print(f"  Found {len(data['code_issues'])} code blocks")
        for i, issue in enumerate(data["code_issues"], 1):
            print(f"  Block {i}: {issue['lines']} lines")
    
    # Security Findings
    if data["security_findings"]:
        print("\n\nSecurity Findings:")
        severity_counts = {}
        for finding in data["security_findings"]:
            severity = finding.get("Severity", "Unknown")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        for severity, count in sorted(severity_counts.items()):
            print(f"  {severity}: {count} issue(s)")
    
    # Dependencies
    if data["dependencies"]:
        deps = data["dependencies"]
        print("\n\nDependencies Analysis:")
        print(f"  Total: {deps.get('total_dependencies')}")
        print(f"  Outdated: {len(deps.get('outdated', []))}")
        print(f"  Vulnerable: {len(deps.get('vulnerable', []))}")
        
        if deps.get('vulnerable'):
            print("\n  Critical Vulnerabilities:")
            for vuln in deps['vulnerable']:
                print(f"    - {vuln['name']} {vuln['current']}: {vuln['cve']}")
    
    # Action Items
    if data["action_items"]:
        print("\n\nAction Items:")
        for i, item in enumerate(data["action_items"], 1):
            print(f"  {i}. {item}")
    
    # Priority Info
    if data["priority_info"]:
        print("\n\nPriority Information:")
        for line in data["priority_info"].split('\n'):
            if line.strip():
                print(f"  {line.strip()}")
    
    # Show any errors or warnings
    if extractor.errors:
        print("\n\nERRORS:")
        for error in extractor.errors:
            print(f"  - {error}")
    
    if extractor.warnings:
        print("\n\nWARNINGS:")
        for warning in extractor.warnings:
            print(f"  - {warning}")
    
    # Generate structured report
    print("\n\n" + "="*80)
    print("STRUCTURED REPORT GENERATION")
    print("="*80)
    
    report = {
        "analysis_type": "software_project",
        "summary": data.get("executive_summary"),
        "metrics": data.get("metrics"),
        "issues": {
            "security": len(data.get("security_findings", [])),
            "code": len(data.get("code_issues", []))
        },
        "dependencies": data.get("dependencies"),
        "action_count": len(data.get("action_items", []))
    }
    
    print("\nGenerated Report Structure:")
    import json
    print(json.dumps(report, indent=2))
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
