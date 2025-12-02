"""
Run evaluation on PDFs
Extracts text from PDF and runs the enhanced evaluator
Usage: python run_pdf_evaluation.py [rfp|sales|both]
"""

import os
import sys
from pathlib import Path

# Try to import PDF extraction libraries
try:
    import PyPDF2
    PDF_EXTRACTOR = "PyPDF2"
except ImportError:
    try:
        import pdfplumber
        PDF_EXTRACTOR = "pdfplumber"
    except ImportError:
        print("❌ No PDF extraction library found!")
        print("\nPlease install one of:")
        print("  pip install PyPDF2")
        print("  pip install pdfplumber")
        sys.exit(1)

from orchestrator import DocumentEvaluator


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using available library"""
    
    if PDF_EXTRACTOR == "PyPDF2":
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    elif PDF_EXTRACTOR == "pdfplumber":
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    
    return ""


def evaluate_single_pdf(pdf_path: str, doc_type: str, target_persona: str, output_name: str):
    """Evaluate a single PDF"""
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return None
    
    print(f"\n{'='*80}")
    print(f"📄 Evaluating: {Path(pdf_path).name}")
    print(f"{'='*80}")
    print(f"📄 Extracting text from PDF using {PDF_EXTRACTOR}...")
    
    document_text = extract_text_from_pdf(pdf_path)
    
    if not document_text or len(document_text) < 100:
        print("❌ Failed to extract text from PDF or document is too short")
        print(f"   Extracted {len(document_text)} characters")
        return None
    
    print(f"✅ Extracted {len(document_text)} characters from PDF")
    
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    # Run evaluation
    print("\n" + "="*80)
    print(f"Starting {doc_type} Evaluation")
    print("="*80 + "\n")
    
    evaluator = DocumentEvaluator()
    
    output_path = f"outputs/{output_name}_eval.txt"
    
    result = evaluator.evaluate_document(
        document=document_text,
        doc_type=doc_type,
        target_persona=target_persona,
        document_name=Path(pdf_path).name,
        save_output=True,
        output_path=output_path
    )
    
    print("\n" + "="*80)
    print(f"✅ Evaluation Complete!")
    print(f"Final Verdict: {result['final_verdict']}")
    if 'final_score' in result:
        print(f"Final Score: {result['final_score']:.0f}/100")
    print("="*80)
    
    # Save summary
    summary_path = f"outputs/{output_name}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"""Evaluation Summary
==================

Document: {Path(pdf_path).name}
Document Type: {doc_type}
Target Persona: {target_persona}

Final Verdict: {result['final_verdict']}
Final Score: {result.get('final_score', 'N/A')}/100

Gate Scores:
- Gate 1 (Integrity): {result.get('gate1_score', 'N/A')}/100
- Gate 2 (Logic): {result.get('gate2_score', 'N/A')}/100
- Gate 3 (Strategy): {result.get('gate3_score', 'N/A')}/100

Blocking Issues: {result.get('blocking_issues', 'Unknown')}

Full evaluation report: {output_path}
""")
    
    print(f"\n📊 Summary saved to: {summary_path}")
    print(f"📄 Full report: {output_path}")
    
    return result


def main():
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        print("\nTo set it:")
        print("  export GEMINI_API_KEY='your-api-key-here'")
        print("\nOr create a .env file with:")
        print("  GEMINI_API_KEY=your-api-key-here")
        sys.exit(1)
    
    # Parse command line arguments
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    
    if mode not in ["rfp", "sales", "both"]:
        print("Usage: python run_pdf_evaluation.py [rfp|sales|both]")
        print("\nOptions:")
        print("  rfp   - Evaluate RFP Response PDF only")
        print("  sales - Evaluate Sales Intelligence PDF only")
        print("  both  - Evaluate both PDFs (default)")
        sys.exit(1)
    
    print("="*80)
    print("🚀 AI DOCUMENT EVALUATION PIPELINE")
    print("="*80)
    
    results = []
    
    # Evaluate RFP
    if mode in ["rfp", "both"]:
        result = evaluate_single_pdf(
            pdf_path="ACME CORPORATION - RFP RESPONSE DOCUMENT.pdf",
            doc_type="RFP_RESPONSE",
            target_persona="Technical Committee",
            output_name="acme_rfp"
        )
        if result:
            results.append({"name": "RFP Response", "verdict": result['final_verdict'], "score": result.get('final_score', 'N/A')})
    
    # Evaluate Sales Intelligence
    if mode in ["sales", "both"]:
        result = evaluate_single_pdf(
            pdf_path="SALES INTELLIGENCE RESPONSE.pdf",
            doc_type="SALES_INTELLIGENCE",
            target_persona="C-Suite",
            output_name="sales_intelligence"
        )
        if result:
            results.append({"name": "Sales Intelligence", "verdict": result['final_verdict'], "score": result.get('final_score', 'N/A')})
    
    # Print summary
    if results:
        print("\n" + "="*80)
        print("📊 EVALUATION SUMMARY")
        print("="*80)
        for r in results:
            emoji = "✅" if r['verdict'] == "APPROVE" else "⚠️" if r['verdict'] == "REVISE" else "❌"
            score_str = f"{r['score']:.0f}/100" if isinstance(r['score'], (int, float)) else r['score']
            print(f"{emoji} {r['name']}: {r['verdict']} ({score_str})")
        print("="*80)


if __name__ == "__main__":
    main()
