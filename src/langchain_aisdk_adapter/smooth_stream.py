"""Smooth Stream Module - AI SDK Compatible smoothStream functionality.

This module provides smooth streaming capabilities similar to AI SDK's smoothStream,
offering controlled chunking and timing for text streams as an optional feature.
"""

import asyncio
import re
from typing import AsyncIterable, AsyncGenerator, Union, Literal, Callable, List


class SmoothStreamTransformer:
    """Transformer class for smooth streaming functionality.
    
    This class implements the core logic for AI SDK's smoothStream,
    providing controlled chunking and timing for text streams.
    """
    
    def __init__(
        self,
        stream: AsyncIterable[str],
        delay_in_ms: int,
        chunking: Union[Literal['word', 'line'], re.Pattern, Callable[[str], List[str]]]
    ):
        self.stream = stream
        self.delay_in_ms = delay_in_ms
        self.chunking = chunking
        self._buffer = ""
        self._is_protocol_format = False
    
    def __aiter__(self):
        return self._transform()
    
    async def _transform(self) -> AsyncGenerator[str, None]:
        """Transform the input stream with smooth chunking and delays."""
        try:
            async for chunk in self.stream:
                if not chunk:
                    continue
                
                # Handle both string chunks and AI SDK protocol chunks
                if isinstance(chunk, dict):
                    # Handle AI SDK protocol format
                    self._is_protocol_format = True
                    if chunk.get('type') == 'text':
                        text_content = chunk.get('content', '')
                        if text_content:
                            self._buffer += text_content
                            
                            # Process buffer based on chunking strategy
                            chunks_to_yield = self._process_buffer()
                            
                            for processed_chunk in chunks_to_yield:
                                if processed_chunk:
                                    # Yield in AI SDK protocol format
                                    yield {'type': 'text', 'content': processed_chunk}
                                    if self.delay_in_ms > 0:
                                        await asyncio.sleep(self.delay_in_ms / 1000.0)
                    else:
                        # Pass through non-text chunks immediately
                        yield chunk
                elif isinstance(chunk, str):
                    # Handle plain string chunks
                    self._buffer += chunk
                    
                    # Process buffer based on chunking strategy
                    chunks_to_yield = self._process_buffer()
                    
                    for processed_chunk in chunks_to_yield:
                        if processed_chunk:
                            yield processed_chunk
                            if self.delay_in_ms > 0:
                                await asyncio.sleep(self.delay_in_ms / 1000.0)
                else:
                    # Pass through other chunk types
                    yield chunk
            
            # Yield any remaining buffer content
            if self._buffer:
                if self._is_protocol_format:
                    # If we were processing AI SDK protocol, yield in that format
                    yield {'type': 'text', 'content': self._buffer}
                else:
                    # Otherwise yield as string
                    yield self._buffer
                self._buffer = ""
                
        except Exception as e:
            # Ensure any remaining buffer is yielded on error
            if self._buffer:
                yield self._buffer
            raise e
    
    def _process_buffer(self) -> List[str]:
        """Process the buffer based on the chunking strategy."""
        if not self._buffer:
            return []
        
        if self.chunking == 'word':
            return self._chunk_by_word()
        elif self.chunking == 'line':
            return self._chunk_by_line()
        elif isinstance(self.chunking, re.Pattern):
            return self._chunk_by_regex(self.chunking)
        elif callable(self.chunking):
            return self._chunk_by_function(self.chunking)
        else:
            # Default: return the entire buffer
            result = [self._buffer]
            self._buffer = ""
            return result
    
    def _chunk_by_word(self) -> List[str]:
        """Chunk by word boundaries."""
        words = self._buffer.split(' ')
        if len(words) <= 1:
            return []
        
        # Keep the last word in buffer (might be incomplete)
        complete_words = words[:-1]
        self._buffer = words[-1]
        
        # Return words with spaces
        result = []
        for i, word in enumerate(complete_words):
            if i == 0:
                result.append(word + ' ')
            else:
                result.append(word + ' ')
        
        return result
    
    def _chunk_by_line(self) -> List[str]:
        """Chunk by line boundaries."""
        lines = self._buffer.split('\n')
        if len(lines) <= 1:
            return []
        
        # Keep the last line in buffer (might be incomplete)
        complete_lines = lines[:-1]
        self._buffer = lines[-1]
        
        # Return lines with newlines
        return [line + '\n' for line in complete_lines]
    
    def _chunk_by_regex(self, pattern: re.Pattern) -> List[str]:
        """Chunk by regex pattern."""
        matches = list(pattern.finditer(self._buffer))
        if not matches:
            return []
        
        result = []
        last_end = 0
        
        for match in matches:
            # Add text before the match
            if match.start() > last_end:
                result.append(self._buffer[last_end:match.start()])
            
            # Add the match itself
            result.append(match.group())
            last_end = match.end()
        
        # Update buffer with remaining text
        self._buffer = self._buffer[last_end:]
        
        return [chunk for chunk in result if chunk]
    
    def _chunk_by_function(self, func: Callable[[str], List[str]]) -> List[str]:
        """Chunk using a custom function."""
        try:
            chunks = func(self._buffer)
            if not chunks:
                return []
            
            # Assume the function processes the entire buffer
            # Keep the last chunk in buffer if it might be incomplete
            if len(chunks) > 1:
                result = chunks[:-1]
                self._buffer = chunks[-1]
                return result
            else:
                # Single chunk - keep in buffer for now
                return []
        except Exception:
            # On error, return the buffer as-is
            result = [self._buffer]
            self._buffer = ""
            return result


def smooth_stream(
    delay_in_ms: int = 10,
    chunking: Union[Literal['word', 'line'], re.Pattern, Callable[[str], List[str]]] = 'word'
) -> Callable[[AsyncIterable[str]], AsyncIterable[str]]:
    """Create a smooth stream transform function for experimental_transform.
    
    This function provides smooth streaming capabilities similar to AI SDK's smoothStream,
    allowing you to control how text is chunked and the delay between chunks.
    It returns a transform function that can be used with experimental_transform.
    
    Args:
        delay_in_ms: Delay between chunks in milliseconds (default: 10)
        chunking: Chunking strategy - 'word', 'line', regex pattern, or custom function
    
    Returns:
        Callable[[AsyncIterable[str]], AsyncIterable[str]]: Transform function for experimental_transform
    
    Examples:
        ```python
        from langchain_aisdk_adapter.smooth_stream import smooth_stream
        from langchain_aisdk_adapter.stream_text import stream_text
        import re
        
        # Word-based chunking (default)
        result = stream_text(
            model=model,
            messages=messages,
            experimental_transform=smooth_stream(delay_in_ms=50, chunking='word')
        )
        
        # Line-based chunking
        result = stream_text(
            model=model,
            messages=messages,
            experimental_transform=smooth_stream(delay_in_ms=100, chunking='line')
        )
        
        # Regex-based chunking (split on punctuation)
        pattern = re.compile(r'[.!?]+\s*')
        result = stream_text(
            model=model,
            messages=messages,
            experimental_transform=smooth_stream(delay_in_ms=200, chunking=pattern)
        )
        
        # Custom function chunking
        def custom_chunker(text: str) -> List[str]:
            return text.split(',')
        
        result = stream_text(
            model=model,
            messages=messages,
            experimental_transform=smooth_stream(delay_in_ms=75, chunking=custom_chunker)
        )
        
        # Consume the smooth stream
        async for chunk in result:
            print(chunk, end='', flush=True)
        ```
    """
    def transform_function(stream: AsyncIterable[str]) -> AsyncIterable[str]:
        return SmoothStreamTransformer(stream, delay_in_ms, chunking)
    
    return transform_function


def apply_smooth_stream(
    stream: AsyncIterable[str],
    delay_in_ms: int = 10,
    chunking: Union[Literal['word', 'line'], re.Pattern, Callable[[str], List[str]]] = 'word'
) -> AsyncIterable[str]:
    """Apply smooth stream transformation directly to a stream.
    
    This function directly applies smooth streaming to an existing stream,
    useful for manual stream processing outside of experimental_transform.
    
    Args:
        stream: The input async iterable of strings
        delay_in_ms: Delay between chunks in milliseconds (default: 10)
        chunking: Chunking strategy - 'word', 'line', regex pattern, or custom function
    
    Returns:
        AsyncIterable[str]: Smoothly chunked stream
    
    Example:
        ```python
        from langchain_aisdk_adapter.smooth_stream import apply_smooth_stream
        
        # Apply to existing stream
        smooth = apply_smooth_stream(text_stream, delay_in_ms=50, chunking='word')
        
        async for chunk in smooth:
            print(chunk, end='', flush=True)
        ```
    """
    return SmoothStreamTransformer(stream, delay_in_ms, chunking)


def create_smooth_text_stream(
    text: str,
    delay_in_ms: int = 10,
    chunking: Union[Literal['word', 'line'], re.Pattern, Callable[[str], List[str]]] = 'word'
) -> AsyncIterable[str]:
    """Create a smooth stream from a static text string.
    
    This is a convenience function for creating smooth streams from static text,
    useful for testing or when you have pre-generated content.
    
    Args:
        text: The text to stream
        delay_in_ms: Delay between chunks in milliseconds
        chunking: Chunking strategy
    
    Returns:
        AsyncIterable[str]: Smoothly chunked stream of the text
    
    Example:
        ```python
        from langchain_aisdk_adapter.smooth_stream import create_smooth_text_stream
        
        text = "Hello world! This is a test message."
        smooth = create_smooth_text_stream(text, delay_in_ms=100, chunking='word')
        
        async for chunk in smooth:
            print(chunk, end='', flush=True)
        ```
    """
    async def text_generator():
        yield text
    
    return apply_smooth_stream(text_generator(), delay_in_ms, chunking)