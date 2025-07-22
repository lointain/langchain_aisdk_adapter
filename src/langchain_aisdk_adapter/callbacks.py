"""Stream callbacks for handling lifecycle events."""

from typing import Awaitable, Callable, Optional, Union


class StreamCallbacks:
    """Configuration options and helper callback methods for stream lifecycle events."""

    def __init__(
        self,
        on_start: Optional[Callable[[], Union[None, Awaitable[None]]]] = None,
        on_final: Optional[Callable[[str], Union[None, Awaitable[None]]]] = None,
        on_token: Optional[Callable[[str], Union[None, Awaitable[None]]]] = None,
        on_text: Optional[Callable[[str], Union[None, Awaitable[None]]]] = None,
    ):
        """Initialize stream callbacks.
        
        Args:
            on_start: Called once when the stream is initialized
            on_final: Called once when the stream is closed with the final completion message
            on_token: Called for each tokenized message
            on_text: Called for each text chunk
        """
        self.on_start = on_start
        self.on_final = on_final
        self.on_token = on_token
        self.on_text = on_text


class CallbacksTransformer:
    """Transform stream that handles callbacks during stream processing.
    
    This class processes a stream of text chunks and invokes optional callback functions
    at different stages of the stream's lifecycle:
    - on_start: Called once when the stream is initialized
    - on_token: Called for each tokenized message
    - on_text: Called for each text chunk
    - on_final: Called once when the stream is closed with the final completion message
    """

    def __init__(self, callbacks: Optional[StreamCallbacks] = None):
        """Initialize the callbacks transformer.
        
        Args:
            callbacks: Optional callbacks configuration
        """
        self.callbacks = callbacks or StreamCallbacks()
        self.aggregated_response = ""
        self._started = False

    async def _call_callback(
        self, 
        callback: Optional[Callable], 
        *args
    ) -> None:
        """Safely call a callback function, handling both sync and async callbacks."""
        if callback is None:
            return
        
        try:
            result = callback(*args)
            if hasattr(result, '__await__'):
                await result
        except Exception:
            # Silently ignore callback errors to prevent stream interruption
            pass

    async def start(self) -> None:
        """Initialize the transformer and call on_start callback."""
        if not self._started:
            self._started = True
            await self._call_callback(self.callbacks.on_start)

    async def transform(self, chunk: str) -> str:
        """Transform a text chunk and call relevant callbacks.
        
        Args:
            chunk: Text chunk to process
            
        Returns:
            The same text chunk (pass-through)
        """
        if not self._started:
            await self.start()

        self.aggregated_response += chunk

        await self._call_callback(self.callbacks.on_token, chunk)
        await self._call_callback(self.callbacks.on_text, chunk)

        return chunk

    async def finish(self) -> None:
        """Finalize the transformer and call on_final callback."""
        await self._call_callback(self.callbacks.on_final, self.aggregated_response)