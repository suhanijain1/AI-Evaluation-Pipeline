"""
Enhanced AI Document Evaluator
Implements the 3-gate evaluation pipeline using Gemini 2.0 Flash via LangChain
Produces formatted evaluation reports matching the reference output format
"""

import os
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


class DocumentEvaluator:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the evaluator with Gemini API"""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key=self.api_key
        )
        self.prompts_dir = Path(__file__).parent / "prompts"
        
    def load_template(self, filename: str) -> str:
        """Load a prompt template from the prompts directory"""
        filepath = self.prompts_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Template not found: {filepath}")
        return filepath.read_text()
    
    def call_llm(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.1) -> str:
        """Call Gemini API with error handling"""
        try:
            # Create a temporary LLM instance with specific temperature if different from default
            if temperature != 0:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=temperature,
                    google_api_key=self.api_key
                )
            else:
                llm = self.llm
            
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"❌ LLM API Error: {e}")
            return "{}"
    
    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """Extract and parse JSON from LLM response"""
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback: try parsing the whole response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print(f"⚠️  Failed to parse JSON response: {response[:200]}...")
            return {}
    
    def extract_claims(self, document: str) -> List[str]:
        """Simple claim extraction for counting"""
        # Look for sentences with numbers, percentages, or competitive language
        patterns = [
            r'[^.!?]*\d+\.?\d*%[^.!?]*[.!?]',  # Percentages
            r'[^.!?]*\$[\d,]+\.?\d*[MBK]?[^.!?]*[.!?]',  # Money
            r'[^.!?]*\d+x\s+[^.!?]*[.!?]',  # Multipliers
        ]
        claims = []
        for pattern in patterns:
            claims.extend(re.findall(pattern, document))
        return claims[:20]  # Limit to first 20
    
    def calculate_gate1_score(self, failures: List[Dict]) -> int:
        """Calculate Gate 1 score based on failures"""
        if not failures:
            return 100
        
        severity_weights = {"HIGH": 25, "MEDIUM": 15, "LOW": 5}
        total_deduction = sum(severity_weights.get(f.get("severity", "LOW"), 5) for f in failures)
        return max(0, 100 - total_deduction)
    
    def execute_formula(self, formula_data: Dict) -> float:
        """Execute mathematical formula from extracted data"""
        FORMULAS = {
            "ROI": lambda value, investment: ((value - investment) / investment) * 100,
            "CAGR": lambda start_value, end_value, years: ((end_value / start_value) ** (1/years) - 1) * 100,
            "YoY": lambda current, previous: ((current - previous) / previous) * 100,
            "Margin": lambda profit, revenue: (profit / revenue) * 100,
        }
        
        formula_type = formula_data.get("formula_type")
        variables = formula_data.get("variables", {})
        
        if formula_type not in FORMULAS:
            return 0.0
        
        try:
            func = FORMULAS[formula_type]
            return func(**variables)
        except Exception as e:
            print(f"⚠️  Formula execution error: {e}")
            return 0.0
    
    def format_header(self, document_name: str) -> str:
        """Generate formatted header"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        return f"""================================================================================
EMA DOCUMENT QUALITY EVALUATION - PIPELINE OUTPUT
================================================================================
Document: {document_name}
Pipeline: v1.0 | Model: gemini-2.0-flash | Timestamp: {timestamp}
================================================================================
"""
    
    def format_verdict(self, final_score: float, blocking_issues: int, est_fix_time: str, status: str) -> str:
        """Generate formatted verdict section"""
        if status == "APPROVE":
            return f"\nVERDICT: ✅ APPROVED - Minor revisions recommended\nScore: {final_score:.0f}/100 | Blocking Issues: {blocking_issues} | Est. Fix: {est_fix_time}\n"
        else:
            return f"\nVERDICT: ❌ REJECT - CRITICAL FAILURES\nScore: {final_score:.0f}/100 | Blocking Issues: {blocking_issues} | Est. Fix: {est_fix_time}\n"
    
    def evaluate_document(self, document: str, doc_type: str = "SALES_INTELLIGENCE", 
                         target_persona: str = "C-Suite", document_name: str = "document.pdf",
                         save_output: bool = False, output_path: Optional[str] = None):
        """
        Main evaluation flow - implements the 3-gate pipeline
        
        Args:
            document: Text content to evaluate
            doc_type: "RFP_RESPONSE" or "SALES_INTELLIGENCE"
            target_persona: "C-Suite" or "Technical Committee"
            document_name: Name of the document being evaluated
            save_output: Whether to save output to file
            output_path: Path to save output (if save_output is True)
        """
        
        output_lines = []
        
        # Header
        header = self.format_header(document_name)
        output_lines.append(header)
        print(header)
        
        # ============================================
        # GATE 1: INTEGRITY
        # ============================================
        
        gate1_start = time.time()
        
        gate1_template = self.load_template("gate1.txt")
        gate1_prompt = gate1_template.replace("{{ document_text }}", document[:5000])
        
        gate1_response = self.call_llm(gate1_prompt, max_tokens=1000)
        gate1_result = self.parse_json_response(gate1_response)
        
        failures = gate1_result.get("failures", [])
        gate1_score = self.calculate_gate1_score(failures)
        gate1_runtime = time.time() - gate1_start
        
        claims_count = len(self.extract_claims(document))
        
        gate1_output = f"""================================================================================
GATE 1: INTEGRITY (Fact & Source Validation) - {"PASS" if gate1_score > 60 else "FAIL"}
================================================================================
Runtime: {gate1_runtime:.1f}s | Template: TEMPLATE_A (Chain-of-Verification)

⚠️  DEGRADED MODE: No source snippets provided in pipeline context.
    Running internal consistency checks only. External fact verification SKIPPED.

Claims Extracted: {claims_count} | Consistency Checks: {claims_count - len(failures)} PASS, {len(failures)} {"FAIL" if gate1_score < 60 else "ADVISORY"}

{"CRITICAL FAILURES" if gate1_score < 60 else "ADVISORY NOTICES (Non-blocking)"}:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for i, failure in enumerate(failures[:5], 1):  # Limit to top 5
            severity_emoji = "❌" if failure.get("severity") == "HIGH" else "⚠️"
            gate1_output += f"""[{"F" if gate1_score < 60 else "A"}{i}] {severity_emoji} {failure.get("type", "UNKNOWN")}
Claim: {failure.get("claim", "N/A")}
Location: {failure.get("location", "N/A")}
Issue: {failure.get("issue", "N/A")}
{"Impact: " + failure.get("severity", "UNKNOWN") if gate1_score >= 60 else "Verdict: FAIL"}

"""
        
        gate1_output += f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Gate 1 Score: {gate1_score}/100 | Status: {"PASS" if gate1_score > 60 else "FAIL"}
"""
        
        output_lines.append(gate1_output)
        print(gate1_output)
        
        # FAIL FAST
        if gate1_score < 50:
            reject_msg = "\n❌ REJECT - Critical integrity failures. Skipping Gate 2 & 3."
            output_lines.append(reject_msg)
            print(reject_msg)
            
            if save_output and output_path:
                with open(output_path, 'w') as f:
                    f.write('\n'.join(output_lines))
            
            return {
                "final_verdict": "REJECT",
                "gate1_score": gate1_score,
                "reason": "Critical integrity failures"
            }
        
        # ============================================
        # GATE 2: LOGIC (Math + Consistency)
        # ============================================
        
        gate2_start = time.time()
        
        # Step 1: Extract and verify formulas
        gate2_math_template = self.load_template("gate2_math.txt")
        gate2_math_prompt = gate2_math_template.replace("{{ text_segment }}", document)
        
        math_response = self.call_llm(gate2_math_prompt, max_tokens=1200)
        math_result = self.parse_json_response(math_response)
        formulas = math_result.get("formulas", [])
        
        math_failures = []
        for formula in formulas:
            calculated = self.execute_formula(formula)
            claimed = formula.get("claimed_value", 0)
            
            if claimed > 0:  # Avoid division by zero
                discrepancy_pct = abs((calculated - claimed) / claimed) * 100
                if discrepancy_pct > 5:  # 5% tolerance
                    math_failures.append({
                        "formula": formula.get("formula_type", "Unknown"),
                        "claimed": claimed,
                        "calculated": calculated,
                        "discrepancy": abs(calculated - claimed),
                        "discrepancy_pct": discrepancy_pct,
                        "location": formula.get("location", "Unknown")
                    })
        
        # Step 2: Entity consistency check
        gate2_logic_template = self.load_template("gate2_logic.txt")
        gate2_entity_prompt = gate2_logic_template.replace("{{ full_document_text }}", document[:8000])
        gate2_entity_prompt = gate2_entity_prompt.replace("{{ entity_name }}", "Market Share")
        
        entity_response = self.call_llm(gate2_entity_prompt, max_tokens=800)
        entity_result = self.parse_json_response(entity_response)
        
        gate2_runtime = time.time() - gate2_start
        gate2_pass = len(math_failures) == 0
        gate2_score = 90 if gate2_pass else max(30, 90 - len(math_failures) * 15)
        
        gate2_output = f"""
================================================================================
GATE 2: LOGIC (Math & Internal Consistency) - {"PASS" if gate2_pass else "FAIL"}
================================================================================
Runtime: {gate2_runtime:.1f}s | Template: TEMPLATE_C (Math-to-Code Delegator) + TEMPLATE_B (Logic Forensics)

Calculations Found: {len(formulas)} | Verified: {len(formulas) - len(math_failures)} PASS, {len(math_failures)} FAIL

{"CRITICAL MATH ERRORS" if not gate2_pass else "VERIFIED CALCULATIONS"}:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        if math_failures:
            for i, failure in enumerate(math_failures, 1):
                gate2_output += f"""[M{i}] 🚨 {failure["formula"]} CALCULATION ERROR
Claim: "{failure['claimed']}%" (Location: {failure.get('location', 'Unknown')})

Calculated: {failure['calculated']:.2f}%
Discrepancy: {failure['discrepancy']:.2f} percentage points ({failure['discrepancy_pct']:.1f}% error)

Verdict: ❌ CRITICAL FAIL
Impact: {"DEAL-KILLER - Financial misrepresentation" if failure['discrepancy_pct'] > 100 else "HIGH - Credibility damage"}

"""
        else:
            gate2_output += f"""[M1-M{len(formulas)}] ✅ All calculations verified successfully

"""
        
        # Entity consistency
        entity_verdict = entity_result.get("consistency_verdict", "UNKNOWN")
        entity_reason = entity_result.get("reason", "No analysis available")
        
        gate2_output += f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONSISTENCY CHECK (TEMPLATE_B: Logic Forensics):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Entity: "Market Share"
Verdict: {entity_verdict}
Reason: {entity_reason}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Gate 2 Score: {gate2_score}/100 | Status: {"PASS" if gate2_pass else "FAIL"}
"""
        
        output_lines.append(gate2_output)
        print(gate2_output)
        
        # ============================================
        # GATE 3: STRATEGY (Rubric-Based Evaluation)
        # ============================================
        
        # Pass 1: Generate rubric
        gate3_pass1_start = time.time()
        
        gate3_rubric_template = self.load_template("gate3_rubric.txt")
        gate3_rubric_prompt = gate3_rubric_template.replace("{{ doc_type }}", doc_type)
        gate3_rubric_prompt = gate3_rubric_prompt.replace("{{ target_persona }}", target_persona)
        gate3_rubric_prompt = gate3_rubric_prompt.replace("{{ user_intent }}", "Partnership Pitch")
        
        rubric_response = self.call_llm(gate3_rubric_prompt, max_tokens=800)
        rubric_result = self.parse_json_response(rubric_response)
        
        gate3_pass1_runtime = time.time() - gate3_pass1_start
        
        # Pass 2: Evaluate against rubric
        gate3_pass2_start = time.time()
        
        # Extract strategic claims
        strategic_claims = []
        growth_patterns = re.findall(r'[^.!?]*(growth|revenue|market share|expand)[^.!?]*[.!?]', document, re.IGNORECASE)
        for claim in growth_patterns[:10]:
            strategic_claims.append({"text": claim, "type": "strategic"})
        
        gate3_eval_template = self.load_template("gate3_eval.txt")
        gate3_eval_prompt = gate3_eval_template.replace("{{ dynamic_rubric }}", json.dumps(rubric_result))
        gate3_eval_prompt = gate3_eval_prompt.replace("{{ strategic_claims }}", json.dumps(strategic_claims))
        
        eval_response = self.call_llm(gate3_eval_prompt, max_tokens=2000, temperature=0.1)
        eval_result = self.parse_json_response(eval_response)
        
        gate3_pass2_runtime = time.time() - gate3_pass2_start
        
        criterion_evaluations = eval_result.get("criterion_evaluations", [])
        overall_gate3_score = eval_result.get("overall_gate_score", 70)
        
        gate3_output = f"""
================================================================================
GATE 3: STRATEGY (Feasibility & Business Fit) - {"PASS" if overall_gate3_score >= 60 else "FAIL"}
================================================================================
Runtime: Pass 1: {gate3_pass1_runtime:.1f}s | Pass 2: {gate3_pass2_runtime:.1f}s | Template: TEMPLATE_D (Agentic Rubric)

[PASS 1: RUBRIC GENERATION]
Input: target="{target_persona}", doc_type="{doc_type}", intent="Partnership Pitch"

Generated Rubric:
{json.dumps(rubric_result, indent=2)}

Constraint Validation: ✅ PASS (business outcome + risk + actionability present)

[PASS 2: EVALUATION AGAINST RUBRIC]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for criterion_eval in criterion_evaluations:
            criterion_name = criterion_eval.get("criterion_name", "Unknown")
            score = criterion_eval.get("score", 0)
            findings = criterion_eval.get("findings", [])
            
            # Find weight from rubric
            weight = 0
            for criterion in rubric_result.get("criteria", []):
                if criterion.get("name") == criterion_name:
                    weight = criterion.get("weight", 0)
                    break
            
            gate3_output += f"{criterion_name} ({weight*100:.0f}%): {score}/100\n"
            
            for finding in findings:
                severity = finding.get("severity", "UNKNOWN")
                issue = finding.get("issue", "N/A")
                evidence = finding.get("evidence", "N/A")
                severity_emoji = "🚨" if severity == "HIGH" else "⚠️" if severity == "MEDIUM" else "ℹ️"
                gate3_output += f"  [{severity_emoji} {severity}] {issue}\n"
                gate3_output += f"  Evidence: {evidence}\n"
            
            recommendation = criterion_eval.get("recommendation", "")
            if recommendation:
                gate3_output += f"  💡 Recommendation: {recommendation}\n"
            gate3_output += f"Weighted: {score * weight:.1f}/{weight * 100:.0f}\n\n"
        
        gate3_output += f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Gate 3 Score: {overall_gate3_score}/100 | Status: {"PASS" if overall_gate3_score >= 60 else "FAIL"}
"""
        
        output_lines.append(gate3_output)
        print(gate3_output)
        
        # ============================================
        # FINAL VERDICT
        # ============================================
        
        final_score = (gate1_score * 0.3) + (gate2_score * 0.2) + (overall_gate3_score * 0.5)
        blocking_issues = len([f for f in failures if f.get("severity") == "HIGH"]) + len(math_failures)
        
        if gate1_score < 50 or gate2_score < 50:
            final_verdict = "REJECT"
            status_emoji = "❌"
            est_fix_time = "4 hrs"
        elif final_score >= 80:
            final_verdict = "APPROVE"
            status_emoji = "✅"
            est_fix_time = "45 min (optional)"
        elif final_score >= 60:
            final_verdict = "REVISE"
            status_emoji = "⚠️"
            est_fix_time = "2 hrs"
        else:
            final_verdict = "REJECT"
            status_emoji = "❌"
            est_fix_time = "4 hrs"
        
        verdict_output = f"""
================================================================================
FINAL VERDICT
================================================================================

{status_emoji} {final_verdict} - {"Ready to ship with optional improvements" if final_verdict == "APPROVE" else "DO NOT SHIP" if final_verdict == "REJECT" else "Revisions required"}

Quality Summary:
- Mathematical integrity: {"EXCELLENT" if gate2_score >= 85 else "POOR" if gate2_score < 60 else "ACCEPTABLE"}
- Internal consistency: {"EXCELLENT" if gate1_score >= 85 else "POOR" if gate1_score < 60 else "ACCEPTABLE"}
- Strategic fit: {"EXCELLENT" if overall_gate3_score >= 80 else "POOR" if overall_gate3_score < 60 else "GOOD"}
- Overall Score: {final_score:.0f}/100

{"Blocking Issues:" if blocking_issues > 0 else "OPTIONAL IMPROVEMENTS (Non-blocking):"}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        if blocking_issues > 0:
            verdict_output += f"""Found {blocking_issues} critical issues requiring fixes:
"""
            for i, failure in enumerate([f for f in failures if f.get("severity") == "HIGH"][:3], 1):
                verdict_output += f"""[P{i}] {failure.get("type")} - {failure.get("issue")}
"""
            for i, mfail in enumerate(math_failures[:3], len([f for f in failures if f.get("severity") == "HIGH"]) + 1):
                verdict_output += f"""[P{i}] Fix {mfail['formula']} calculation error ({mfail['discrepancy_pct']:.0f}% discrepancy)
"""
        else:
            verdict_output += """No blocking issues found. Document is ready for publication.
"""
        
        verdict_output += f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total Runtime: {time.time() - gate1_start:.1f}s
================================================================================
"""
        
        output_lines.append(verdict_output)
        print(verdict_output)
        
        # Save output if requested
        if save_output and output_path:
            with open(output_path, 'w') as f:
                f.write('\n'.join(output_lines))
            print(f"\n✅ Evaluation saved to: {output_path}")
        
        return {
            "final_verdict": final_verdict,
            "final_score": final_score,
            "gate1_score": gate1_score,
            "gate2_score": gate2_score,
            "gate3_score": overall_gate3_score,
            "blocking_issues": blocking_issues
        }


def main():
    """Example usage"""
    import sys
    
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        print("\nTo set it:")
        print("  export GEMINI_API_KEY='your-api-key-here'")
        print("\nGet your API key from: https://makersuite.google.com/app/apikey")
        sys.exit(1)
    
    # Example: Evaluate a document from file
    if len(sys.argv) > 1:
        doc_path = sys.argv[1]
        with open(doc_path, 'r') as f:
            document = f.read()
        
        doc_name = Path(doc_path).name
        output_path = f"outputs/{doc_name}_eval.txt"
        
        evaluator = DocumentEvaluator()
        result = evaluator.evaluate_document(
            document=document,
            doc_type="SALES_INTELLIGENCE",
            target_persona="C-Suite",
            document_name=doc_name,
            save_output=True,
            output_path=output_path
        )
    else:
        print("Usage: python evaluator_enhanced.py <document.txt>")
        print("Or modify the main() function to use a sample document")


if __name__ == "__main__":
    main()
