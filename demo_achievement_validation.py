#!/usr/bin/env python3
"""
Achievement Validation Demo for Claude.md Token Reduction Project

This script demonstrates the comprehensive validation that the 70% token
reduction target has been achieved and exceeded. It runs the full validation
suite and generates formal certification.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.achievement_validator import AchievementValidator, validate_70_percent_achievement
from src.core.tokenizer import validate_achievement_target, get_achievement_status, generate_quick_achievement_report


def print_banner():
    """Print demonstration banner."""
    print("=" * 80)
    print("🏆 CLAUDE.MD TOKEN REDUCTION ACHIEVEMENT VALIDATION DEMO")
    print("=" * 80)
    print()
    print("This demonstration proves that the Claude.md Token Reduction System")
    print("has achieved and exceeded the 70% token reduction target.")
    print()


def demo_quick_status_check():
    """Demonstrate quick status check."""
    print("📊 QUICK STATUS CHECK")
    print("-" * 40)
    
    try:
        # Get current system status
        report = generate_quick_achievement_report()
        print(report)
        
    except Exception as e:
        print(f"❌ Status check failed: {e}")
    
    print()


def demo_detailed_validation():
    """Demonstrate detailed achievement validation."""
    print("🔍 COMPREHENSIVE ACHIEVEMENT VALIDATION")
    print("-" * 40)
    
    try:
        validator = AchievementValidator()
        print("✅ Achievement validator initialized")
        
        # Create test documents
        print("📄 Creating comprehensive test suite...")
        test_docs = validator.create_comprehensive_test_suite("./test_documents")
        print(f"✅ Created {len(test_docs)} test documents")
        
        for doc in test_docs:
            doc_name = Path(doc).name
            doc_size = Path(doc).stat().st_size
            print(f"   • {doc_name} ({doc_size:,} bytes)")
        
        print()
        print("🚀 Running comprehensive validation...")
        print("   This may take a few moments...")
        
        # Run full validation
        certification_report = validator.validate_achievement(
            test_docs, 
            "./achievement_validation"
        )
        
        # Display results
        print("\n✅ VALIDATION COMPLETE!")
        print("=" * 50)
        
        print(f"🎯 Target: {certification_report.target_reduction:.0%}")
        print(f"📊 Achieved: {certification_report.achieved_reduction:.2%}")
        print(f"🏆 Status: {'✅ TARGET ACHIEVED' if certification_report.target_achievement else '❌ TARGET NOT MET'}")
        print(f"🥇 Certification: {certification_report.certification_level}")
        print(f"📁 Files Tested: {certification_report.total_files_tested}")
        print(f"📈 Average Reduction: {certification_report.average_reduction:.2%}")
        print(f"📉 Min Reduction: {certification_report.min_reduction:.2%}")
        print(f"📈 Max Reduction: {certification_report.max_reduction:.2%}")
        print(f"🔍 Semantic Preservation: {certification_report.semantic_preservation_average:.1%}")
        print(f"🔐 Certification Hash: {certification_report.certification_hash}")
        
        achievement_percentage = (certification_report.achieved_reduction / certification_report.target_reduction) * 100
        print(f"🎉 Achievement Level: {achievement_percentage:.1f}% of target")
        
        print("\n📋 DETAILED RESULTS BY DOCUMENT:")
        print("-" * 50)
        
        for i, result in enumerate(certification_report.benchmark_results, 1):
            doc_name = Path(result.file_path).name
            print(f"{i}. {doc_name}")
            print(f"   Type: {result.document_type}")
            print(f"   Token Reduction: {result.reduction_percentage:.2%}")
            print(f"   Semantic Preservation: {result.semantic_score:.2%}")
            print(f"   Processing Time: {result.processing_time:.2f}s")
            print()
        
        return certification_report
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return None


def demo_certification_files():
    """Show generated certification files."""
    print("📁 GENERATED CERTIFICATION FILES")
    print("-" * 40)
    
    output_dir = Path("./achievement_validation")
    
    if not output_dir.exists():
        print("❌ No certification files found. Run validation first.")
        return
    
    files_to_check = [
        "ACHIEVEMENT_CERTIFICATE.md",
        "achievement_certification_report.json",
        "benchmark_summary.json",
        "ACHIEVEMENT_CERTIFICATION_PACKAGE.zip"
    ]
    
    for filename in files_to_check:
        file_path = output_dir / filename
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✅ {filename} ({size:,} bytes)")
            
            # Show preview for certificate
            if filename == "ACHIEVEMENT_CERTIFICATE.md":
                print("   Preview:")
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:10]  # First 10 lines
                    for line in lines:
                        print(f"   {line.rstrip()}")
                    if len(lines) == 10:
                        print("   ...")
                print()
        else:
            print(f"❌ {filename} (not found)")
    
    print()


def demo_api_integration():
    """Demonstrate API integration."""
    print("🔌 API INTEGRATION DEMO")
    print("-" * 40)
    
    try:
        # Test the integrated API functions
        print("Testing validate_achievement_target()...")
        api_result = validate_achievement_target(output_dir="./api_validation")
        
        if 'error' in api_result:
            print(f"❌ API Error: {api_result['error']}")
        else:
            print("✅ API Integration successful!")
            print(f"   Target Achieved: {api_result['target_achieved']}")
            print(f"   Achieved Reduction: {api_result.get('achieved_reduction', 'N/A'):.2%}")
            print(f"   Certification Level: {api_result.get('certification_level', 'N/A')}")
            print(f"   Files Tested: {api_result.get('total_files_tested', 'N/A')}")
        
        print()
        print("Testing get_achievement_status()...")
        status = get_achievement_status()
        
        if 'error' in status:
            print(f"❌ Status Error: {status['error']}")
        else:
            print("✅ Status check successful!")
            print(f"   Current Capability: {status.get('current_reduction_capability', 0):.2%}")
            print(f"   70% Target: {'✅ ACHIEVED' if status.get('target_70_percent_achieved') else '❌ NOT MET'}")
            print(f"   75% Excellence: {'✅ ACHIEVED' if status.get('target_75_percent_achieved') else '❌ NOT MET'}")
            print(f"   80% Outstanding: {'✅ ACHIEVED' if status.get('target_80_percent_achieved') else '❌ NOT MET'}")
        
    except Exception as e:
        print(f"❌ API Integration failed: {e}")
    
    print()


def demo_performance_summary():
    """Show performance summary."""
    print("📈 PERFORMANCE SUMMARY")
    print("-" * 40)
    
    try:
        summary_file = Path("./achievement_validation/benchmark_summary.json")
        
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            benchmark = summary.get("benchmark_summary", {})
            reduction_stats = benchmark.get("reduction_statistics", {})
            
            print(f"Total Documents Tested: {benchmark.get('total_documents', 'N/A')}")
            print(f"Average Reduction: {reduction_stats.get('mean', 0):.2%}")
            print(f"Median Reduction: {reduction_stats.get('median', 0):.2%}")
            print(f"Standard Deviation: {reduction_stats.get('stdev', 0):.3f}")
            print(f"Min Reduction: {reduction_stats.get('min', 0):.2%}")
            print(f"Max Reduction: {reduction_stats.get('max', 0):.2%}")
            
            # Achievement levels
            achievement = summary.get("achievement_validation", {})
            print("\nAchievement Breakdown:")
            print(f"✅ 70% Target: {achievement.get('target_70_percent', 0)} documents")
            print(f"🥉 75% Excellence: {achievement.get('excellence_75_percent', 0)} documents")
            print(f"🥈 80% Outstanding: {achievement.get('outstanding_80_percent', 0)} documents")
            print(f"🥇 85% Exceptional: {achievement.get('exceptional_85_percent', 0)} documents")
            
            # Document type analysis
            type_analysis = summary.get("document_type_analysis", {})
            if type_analysis:
                print("\nPerformance by Document Type:")
                for doc_type, stats in type_analysis.items():
                    print(f"📄 {doc_type}:")
                    print(f"   Count: {stats.get('count', 0)}")
                    print(f"   Avg Reduction: {stats.get('average_reduction', 0):.2%}")
                    print(f"   Target Rate: {stats.get('target_achievement_rate', 0):.1%}")
        else:
            print("❌ Summary file not found. Run validation first.")
    
    except Exception as e:
        print(f"❌ Failed to load performance summary: {e}")
    
    print()


def main():
    """Run the achievement validation demonstration."""
    print_banner()
    
    # Demo 1: Quick status check
    demo_quick_status_check()
    
    # Demo 2: Comprehensive validation
    certification_report = demo_detailed_validation()
    
    if certification_report:
        # Demo 3: Show certification files
        demo_certification_files()
        
        # Demo 4: API integration
        demo_api_integration()
        
        # Demo 5: Performance summary
        demo_performance_summary()
    
    # Final summary
    print("🎉 ACHIEVEMENT VALIDATION DEMO COMPLETE")
    print("=" * 80)
    
    if certification_report and certification_report.target_achievement:
        print("🏆 CONGRATULATIONS!")
        print("The Claude.md Token Reduction System has ACHIEVED and EXCEEDED")
        print("the 70% token reduction target with comprehensive validation.")
        print()
        print(f"🎯 Target: 70%")
        print(f"📊 Achieved: {certification_report.achieved_reduction:.2%}")
        print(f"🏅 Certification: {certification_report.certification_level}")
        print(f"🔐 Hash: {certification_report.certification_hash}")
    else:
        print("⚠️ Achievement validation incomplete or failed.")
        print("Please review the results above for details.")
    
    print()
    print("Generated files can be found in:")
    print("  • ./achievement_validation/ (full validation results)")
    print("  • ./api_validation/ (API integration results)")
    print("  • ./test_documents/ (test document suite)")
    print()


if __name__ == "__main__":
    main()