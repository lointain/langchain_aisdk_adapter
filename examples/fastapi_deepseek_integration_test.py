#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI DeepSeek Integration Test

This test verifies:
1. FastAPI backend connection to DeepSeek API
2. Stream processing without errors
3. Message ID handling (with and without)
4. Error handling and display
5. Both auto and manual stream modes
"""

import asyncio
import aiohttp
import json
import os
from typing import Dict, Any


class FastAPITester:
    """Test FastAPI endpoints with real requests"""
    
    def __init__(self, base_url: str = "http://localhost:3001"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_health_check(self) -> bool:
        """Test backend health"""
        try:
            async with self.session.get(f"{self.base_url}/api/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Health check passed: {data}")
                    return True
                else:
                    print(f"❌ Health check failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    async def test_chat_endpoint(self, mode: str, test_query: str, message_id: str = None) -> Dict[str, Any]:
        """Test chat endpoint"""
        endpoint = f"{self.base_url}/api/chat/{mode}"
        
        payload = {
            "messages": [
                {"role": "user", "content": test_query}
            ]
        }
        
        if message_id:
            payload["message_id"] = message_id
        
        payload["stream_mode"] = mode
        
        print(f"\n🔄 Testing {mode} mode...")
        print(f"📤 Sending to: {endpoint}")
        print(f"📋 Payload: {json.dumps(payload, indent=2)}")
        
        try:
            async with self.session.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                print(f"📊 Response status: {response.status}")
                print(f"📋 Response headers: {dict(response.headers)}")
                
                if response.status != 200:
                    error_text = await response.text()
                    print(f"❌ Error response: {error_text}")
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}",
                        "chunks": []
                    }
                
                chunks = []
                error_detected = False
                
                print("\n📡 Stream chunks:")
                print("-" * 50)
                
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    if line_str:
                        chunks.append(line_str)
                        print(f"📦 {len(chunks):3d}: {line_str[:100]}{'...' if len(line_str) > 100 else ''}")
                        
                        # Check for errors
                        if line_str.startswith("data: "):
                            try:
                                data = json.loads(line_str[6:])
                                if data.get("type") == "error":
                                    error_detected = True
                                    print(f"🚨 ERROR DETECTED: {data.get('errorText')}")
                            except json.JSONDecodeError:
                                pass
                
                print("-" * 50)
                print(f"📊 Total chunks: {len(chunks)}")
                
                return {
                    "success": not error_detected,
                    "error": "Stream error detected" if error_detected else None,
                    "chunks": chunks,
                    "chunk_count": len(chunks)
                }
                
        except Exception as e:
            print(f"❌ Request error: {e}")
            return {
                "success": False,
                "error": str(e),
                "chunks": []
            }
    
    async def test_message_id_scenarios(self) -> Dict[str, Any]:
        """Test different message ID scenarios"""
        results = {}
        
        # Test 1: Without message_id
        print("\n" + "="*60)
        print("TEST 1: WITHOUT MESSAGE_ID")
        print("="*60)
        
        result1 = await self.test_chat_endpoint(
            mode="auto",
            test_query="What is 2 + 2?",
            message_id=None
        )
        results["without_message_id"] = result1
        
        # Test 2: With message_id
        print("\n" + "="*60)
        print("TEST 2: WITH MESSAGE_ID")
        print("="*60)
        
        result2 = await self.test_chat_endpoint(
            mode="auto",
            test_query="What is 3 + 3?",
            message_id="test_msg_123"
        )
        results["with_message_id"] = result2
        
        return results
    
    async def test_both_modes(self) -> Dict[str, Any]:
        """Test both auto and manual modes"""
        results = {}
        
        # Test auto mode
        print("\n" + "="*60)
        print("TEST: AUTO MODE")
        print("="*60)
        
        result_auto = await self.test_chat_endpoint(
            mode="auto",
            test_query="Please calculate 15 * 8 and tell me the result",
            message_id="auto_test_456"
        )
        results["auto_mode"] = result_auto
        
        # Test manual mode
        print("\n" + "="*60)
        print("TEST: MANUAL MODE")
        print("="*60)
        
        result_manual = await self.test_chat_endpoint(
            mode="manual",
            test_query="Please calculate 20 * 4 and tell me the result",
            message_id="manual_test_789"
        )
        results["manual_mode"] = result_manual
        
        return results
    
    async def test_error_scenarios(self) -> Dict[str, Any]:
        """Test error handling scenarios"""
        results = {}
        
        # Test with invalid endpoint
        print("\n" + "="*60)
        print("TEST: INVALID ENDPOINT")
        print("="*60)
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/chat/invalid",
                json={"messages": [{"role": "user", "content": "test"}]}
            ) as response:
                error_text = await response.text()
                results["invalid_endpoint"] = {
                    "status": response.status,
                    "error": error_text,
                    "success": False
                }
                print(f"📊 Status: {response.status}")
                print(f"❌ Error: {error_text}")
        except Exception as e:
            results["invalid_endpoint"] = {
                "error": str(e),
                "success": False
            }
            print(f"❌ Exception: {e}")
        
        return results


async def main():
    """Main test function"""
    print("FastAPI DeepSeek Integration Test")
    print("=" * 50)
    
    # Check environment
    if not os.getenv('DEEPSEEK_API_KEY'):
        print("❌ DEEPSEEK_API_KEY environment variable not set")
        return
    
    print("✅ DEEPSEEK_API_KEY found")
    
    async with FastAPITester() as tester:
        # Test 1: Health check
        print("\n" + "="*60)
        print("HEALTH CHECK")
        print("="*60)
        
        health_ok = await tester.test_health_check()
        if not health_ok:
            print("❌ Backend is not healthy, stopping tests")
            return
        
        # Test 2: Message ID scenarios
        message_id_results = await tester.test_message_id_scenarios()
        
        # Test 3: Both modes
        mode_results = await tester.test_both_modes()
        
        # Test 4: Error scenarios
        error_results = await tester.test_error_scenarios()
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        all_results = {
            **message_id_results,
            **mode_results,
            **error_results
        }
        
        success_count = sum(1 for result in all_results.values() if result.get("success", False))
        total_count = len(all_results)
        
        print(f"✅ Successful tests: {success_count}/{total_count}")
        
        for test_name, result in all_results.items():
            status = "✅" if result.get("success", False) else "❌"
            chunk_info = f" ({result.get('chunk_count', 0)} chunks)" if 'chunk_count' in result else ""
            error_info = f" - {result.get('error', '')}" if result.get('error') else ""
            print(f"{status} {test_name}{chunk_info}{error_info}")
        
        print("\n" + "="*60)
        print("INTEGRATION TEST COMPLETED")
        print("="*60)
        
        if success_count == total_count:
            print("🎉 All tests passed! FastAPI + DeepSeek integration is working correctly.")
        else:
            print(f"⚠️  {total_count - success_count} test(s) failed. Please check the logs above.")


if __name__ == "__main__":
    asyncio.run(main())