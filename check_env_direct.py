#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct Environment File Reader

Directly reads .env file without using python-dotenv to avoid caching issues.
"""

import os

def parse_env_file(file_path):
    """Parse .env file manually"""
    env_vars = {}
    
    if not os.path.exists(file_path):
        return env_vars
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Parse KEY=VALUE format
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                
                env_vars[key] = value
                print(f"Line {line_num}: {key} = '{value}'")
    
    return env_vars

def main():
    print("=== Direct .env File Reading ===")
    
    env_file_path = ".env"
    
    if os.path.exists(env_file_path):
        print(f"✓ .env file found at: {os.path.abspath(env_file_path)}")
        print("\n=== File Contents ===")
        
        env_vars = parse_env_file(env_file_path)
        
        print("\n=== Parsed Environment Variables ===")
        
        # Check DEEPSEEK_API_KEY specifically
        api_key = env_vars.get('DEEPSEEK_API_KEY')
        if api_key:
            masked_key = api_key[:8] + "*" * (len(api_key) - 8) if len(api_key) > 8 else "*" * len(api_key)
            print(f"✓ DEEPSEEK_API_KEY: {masked_key}")
            print(f"✓ Key length: {len(api_key)} characters")
            
            if api_key.startswith("sk-") and len(api_key) > 20:
                print("✓ API key format appears valid")
            else:
                print("⚠ API key format may be invalid")
                
            print(f"\n=== Full API Key (for debugging) ===")
            print(f"DEEPSEEK_API_KEY = '{api_key}'")
        else:
            print("✗ DEEPSEEK_API_KEY not found")
        
        # Check other important variables
        base_url = env_vars.get('DEEPSEEK_BASE_URL')
        if base_url:
            print(f"✓ DEEPSEEK_BASE_URL: {base_url}")
        
        model_name = env_vars.get('EXAMPLE_MODEL_NAME')
        if model_name:
            print(f"✓ Model name: {model_name}")
            
    else:
        print(f"✗ .env file not found at: {os.path.abspath(env_file_path)}")

if __name__ == "__main__":
    main()