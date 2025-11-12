"""Quick validation test for bulk testing system."""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bulk_testing import BulkTestRunner, BulkTestConfig
from dataset_adapters import FinanceBenchAdapter

def test_single_question():
    """Test processing a single question."""
    print("="*60)
    print("BULK TESTING SYSTEM - VALIDATION TEST")
    print("="*60)

    # Create config
    config = BulkTestConfig(
        dataset_name="financebench",
        top_k_retrieval=5,
        temperature=0.0,
        max_tokens=512
    )

    # Create adapter
    adapter = FinanceBenchAdapter()

    # Load dataset
    print("\n1. Loading dataset...")
    df = adapter.load_dataset()
    print(f"   ✓ Loaded {len(df)} questions")

    # Get first question
    question = df.iloc[22]['question']
    gold_answer = df.iloc[22]['answer']
    question_type = df.iloc[22]['question_type']

    print(f"\n2. Test Question:")
    print(f"   Type: {question_type}")
    print(f"   Question: {question}")
    print(f"   Gold Answer: {gold_answer}")

    # Initialize framework
    print("\n3. Initializing framework...")
    runner = BulkTestRunner(config)
    if not runner.initialize_framework():
        print("   ✗ Initialization failed")
        return False
    print("   ✓ Framework initialized")

    # Process single question
    print("\n4. Processing question...")
    try:
        result = runner.process_single_question(question, 1)

        if result['error']:
            print(f"   ✗ Error: {result['error']}")
            return False

        print(f"   ✓ Prediction: {result['predicted_answer']}")
        print(f"   ✓ Retrieval time: {result['retrieval_time_ms']:.1f} ms")
        print(f"   ✓ Generation time: {result['generation_time_ms']:.1f} ms")
        print(f"   ✓ Sources: {len(result['sources'])} documents")

        # Test metrics
        print("\n5. Testing evaluation metrics...")
        from evaluation.metrics import embedding_similarity

        sem_sim = embedding_similarity(
            result['predicted_answer'],
            gold_answer,
            runner.embeddings
        )
        print(f"   ✓ Semantic similarity: {sem_sim:.4f}")

        # Check if answer quality improved with new prompt
        if sem_sim > 0.7:
            print("   ✓ HIGH similarity - answer quality is good!")
        elif sem_sim > 0.5:
            print("   ⚠ MEDIUM similarity - answer partially matches gold answer")
        else:
            print("   ⚠ LOW similarity - answer may not be accurate")

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe bulk testing system is ready to use!")
        print("Run: python bulk_testing.py --dataset financebench")
        print("="*60 + "\n")

        return True

    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_question()
    exit(0 if success else 1)
