#!/usr/bin/env python3
import json
import jsonlines
import argparse
import time
from agent.graph_hybrid import HybridAgent

def main():
    parser = argparse.ArgumentParser(description="Retail Analytics Copilot")
    parser.add_argument("--batch", required=True, help="Input JSONL file with questions")
    parser.add_argument("--out", required=True, help="Output JSONL file for results")
    args = parser.parse_args()
    
    agent = HybridAgent()
    results = []
    
    # Read all questions first to show progress
    with jsonlines.open(args.batch) as reader:
        questions = list(reader)
    
    total_questions = len(questions)
    print(f"🚀 Starting processing of {total_questions} questions...")
    
    # Process each question in batch
    for i, line in enumerate(questions, 1):
        start_time = time.time()
        
        print(f"\n📊 Processing question {i}/{total_questions}")
        print(f"   ID: {line['id']}")
        print(f"   Question: {line['question']}")
        print(f"   Format hint: {line.get('format_hint', 'str')}")
        print("   " + "="*50)
        
        result = agent.run(
            question=line["question"],
            format_hint=line.get("format_hint", "str")
        )
        
        processing_time = time.time() - start_time
        
        output = {
            "id": line["id"],
            "final_answer": result["final_answer"],
            "sql": result["sql"],
            "confidence": result["confidence"],
            "explanation": result["explanation"],
            "citations": result["citations"]
        }
        results.append(output)
        
        # Show progress and results
        print(f"   ✅ Completed in {processing_time:.2f}s")
        print(f"   SQL Generated: {'Yes' if result['sql'] else 'No'}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Citations: {len(result['citations'])} items")
        print(f"   Answer: {result['final_answer']}")
        print(f"   Explanation: {result['explanation']}")
        
        # Show progress percentage
        progress = (i / total_questions) * 100
        print(f"\n📈 Overall Progress: {progress:.1f}% ({i}/{total_questions})")
        print("   " + "="*50)
    
    # Write results
    with jsonlines.open(args.out, mode='w') as writer:
        for result in results:
            writer.write(result)
    
    print(f"\n🎉 All done! Results written to {args.out}")
    print(f"   Total questions processed: {total_questions}")
    print(f"   Output file: {args.out}")

if __name__ == "__main__":
    main()