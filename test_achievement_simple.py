#!/usr/bin/env python3
"""
Simple Achievement Validation Test

This script demonstrates that the Claude.md Token Reduction System
achieves the 70% reduction target with a realistic test.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.achievement_validator import AchievementValidator
from src.core.tokenizer import ClaudeMdTokenizer


def create_realistic_test_document():
    """Create a realistic test document with moderate redundancy."""
    content = """# Project Configuration Guide

## Overview
This guide provides comprehensive configuration instructions for the project setup and deployment.

## Basic Configuration
The basic configuration involves setting up the following components:

### Database Configuration
Configure your database connection:
- Host: localhost
- Port: 5432  
- Database: myapp
- Username: admin
- Password: secure_password

### Server Configuration  
Configure the application server:
- Port: 8080
- Environment: development
- Debug mode: enabled
- Log level: INFO

## Advanced Configuration
For advanced setups, consider these additional options:

### Performance Tuning
- Connection pool size: 20
- Request timeout: 30 seconds
- Cache TTL: 3600 seconds

### Security Settings
- Enable HTTPS: true
- JWT expiration: 3600 seconds
- Rate limiting: 100 requests per minute

## Environment Variables
Set these environment variables:
- DATABASE_URL=postgresql://admin:secure_password@localhost:5432/myapp
- SECRET_KEY=your-secret-key-here
- DEBUG=true

## Deployment Options
Choose one of these deployment methods:

### Option 1: Docker Deployment
Use Docker for containerized deployment with these steps:
1. Build the Docker image
2. Configure environment variables  
3. Deploy the container

### Option 2: Traditional Deployment
Use traditional deployment on bare metal or VM:
1. Install dependencies
2. Configure the environment
3. Start the application

## Troubleshooting
If you encounter issues:
- Check the logs for error messages
- Verify configuration settings
- Ensure all dependencies are installed
- Contact support if problems persist
"""
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8')
    temp_file.write(content)
    temp_file.close()
    
    return temp_file.name, content


def main():
    """Run simple achievement validation."""
    print("🏆 CLAUDE.MD TOKEN REDUCTION - SIMPLE ACHIEVEMENT TEST")
    print("=" * 60)
    
    try:
        # Create test document
        test_file, original_content = create_realistic_test_document()
        print(f"📄 Created test document: {len(original_content)} characters")
        
        # Initialize tokenizer
        tokenizer = ClaudeMdTokenizer()
        print("✅ Tokenizer initialized")
        
        # Estimate original tokens
        original_tokens = tokenizer._estimate_tokens(original_content)
        print(f"🔍 Original tokens: {original_tokens}")
        
        # Optimize the document
        print("⚡ Running optimization...")
        result = tokenizer.optimize_file(test_file)
        
        optimized_tokens = result.optimized_tokens
        reduction_percentage = (original_tokens - optimized_tokens) / original_tokens
        
        print("\n📊 RESULTS:")
        print("-" * 30)
        print(f"Original tokens: {original_tokens}")
        print(f"Optimized tokens: {optimized_tokens}")
        print(f"Reduction: {reduction_percentage:.2%}")
        print(f"Target (70%): {'✅ ACHIEVED' if reduction_percentage >= 0.70 else '❌ NOT MET'}")
        
        # Certification level
        if reduction_percentage >= 0.85:
            level = "🥇 EXCEPTIONAL"
        elif reduction_percentage >= 0.80:
            level = "🥈 OUTSTANDING"
        elif reduction_percentage >= 0.75:
            level = "🥉 EXCELLENCE"
        elif reduction_percentage >= 0.70:
            level = "🎯 TARGET ACHIEVED"
        else:
            level = "❌ BELOW TARGET"
        
        print(f"Certification: {level}")
        
        # Final assessment
        print("\n🎉 FINAL ASSESSMENT:")
        print("-" * 30)
        
        if reduction_percentage >= 0.70:
            achievement_percentage = (reduction_percentage / 0.70) * 100
            print(f"✅ The Claude.md Token Reduction System has ACHIEVED the 70% target!")
            print(f"🚀 Achievement level: {achievement_percentage:.1f}% of target")
            print(f"📈 Actual reduction: {reduction_percentage:.2%}")
            print("🏆 CERTIFICATION: The system meets and exceeds the requirements.")
        else:
            print(f"❌ The system achieved {reduction_percentage:.2%} reduction.")
            print(f"📉 This is below the 70% target requirement.")
            print("🔧 Additional optimization may be needed.")
        
        return reduction_percentage >= 0.70
        
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        return False
        
    finally:
        # Clean up
        if 'test_file' in locals() and Path(test_file).exists():
            Path(test_file).unlink()
        
        # Clean up optimized file if it exists
        optimized_file = test_file.replace('.md', '_optimized.md')
        if Path(optimized_file).exists():
            Path(optimized_file).unlink()


def run_multiple_tests():
    """Run multiple tests with different content types."""
    print("\n🔄 RUNNING MULTIPLE VALIDATION TESTS")
    print("=" * 60)
    
    test_contents = [
        ("Technical Documentation", """# API Reference

## Authentication
Use API keys for authentication:
- API Key: your-api-key-here
- Header: Authorization: Bearer {api_key}

## Endpoints
Available endpoints:
- GET /api/users - Get users list
- POST /api/users - Create new user
- PUT /api/users/{id} - Update user
- DELETE /api/users/{id} - Delete user

## Error Handling
Common error codes:
- 400: Bad Request
- 401: Unauthorized  
- 404: Not Found
- 500: Server Error
"""),
        
        ("Configuration Guide", """# Setup Instructions

## Prerequisites
Install these dependencies:
- Node.js 16+
- Python 3.8+
- PostgreSQL 12+

## Installation Steps
1. Clone the repository
2. Install dependencies: npm install
3. Configure environment variables
4. Run database migrations
5. Start the application

## Configuration
Edit the config file:
- Database URL
- API keys
- Environment settings
"""),
        
        ("Tutorial Content", """# Getting Started

## Introduction
Welcome to our platform! This tutorial will guide you through the basic features.

## Step 1: Account Setup
Create your account by following these steps:
1. Go to the registration page
2. Fill in your details
3. Verify your email
4. Complete your profile

## Step 2: First Project
Create your first project:
1. Click "New Project"
2. Choose a template
3. Configure settings
4. Save and start working
""")
    ]
    
    results = []
    
    for i, (doc_type, content) in enumerate(test_contents, 1):
        print(f"\n🧪 Test {i}: {doc_type}")
        print("-" * 40)
        
        try:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8')
            temp_file.write(content)
            temp_file.close()
            
            # Initialize tokenizer and test
            tokenizer = ClaudeMdTokenizer()
            original_tokens = tokenizer._estimate_tokens(content)
            result = tokenizer.optimize_file(temp_file.name)
            optimized_tokens = result.optimized_tokens
            reduction = (original_tokens - optimized_tokens) / original_tokens
            
            print(f"   Original: {original_tokens} tokens")
            print(f"   Optimized: {optimized_tokens} tokens") 
            print(f"   Reduction: {reduction:.2%}")
            print(f"   Status: {'✅ PASS' if reduction >= 0.70 else '❌ FAIL'}")
            
            results.append((doc_type, reduction, reduction >= 0.70))
            
            # Clean up
            Path(temp_file.name).unlink()
            optimized_path = temp_file.name.replace('.md', '_optimized.md')
            if Path(optimized_path).exists():
                Path(optimized_path).unlink()
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append((doc_type, 0.0, False))
    
    # Summary
    print(f"\n📈 TEST SUMMARY")
    print("=" * 40)
    
    passed_tests = sum(1 for _, _, passed in results if passed)
    total_tests = len(results)
    
    for doc_type, reduction, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{doc_type:20} {reduction:6.2%} {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🏆 ALL TESTS PASSED - 70% TARGET ACHIEVED!")
    elif passed_tests > 0:
        print(f"⚠️ PARTIAL SUCCESS - {passed_tests} out of {total_tests} tests passed")
    else:
        print("❌ ALL TESTS FAILED - Target not achieved")
    
    return passed_tests / total_tests


if __name__ == "__main__":
    # Run simple test
    simple_success = main()
    
    # Run multiple tests
    multiple_success_rate = run_multiple_tests()
    
    print(f"\n🎯 FINAL CERTIFICATION")
    print("=" * 50)
    print(f"Simple Test: {'✅ PASSED' if simple_success else '❌ FAILED'}")
    print(f"Multiple Tests: {multiple_success_rate:.1%} success rate")
    
    if simple_success and multiple_success_rate > 0.8:
        print("\n🏆 OFFICIAL CERTIFICATION:")
        print("The Claude.md Token Reduction System has been VALIDATED")
        print("and CERTIFIED as achieving the 70% token reduction target.")
        print("\n✅ ACHIEVEMENT CONFIRMED!")
    else:
        print("\n⚠️ CERTIFICATION PENDING:")
        print("Additional optimization may be required to consistently")
        print("achieve the 70% token reduction target.")