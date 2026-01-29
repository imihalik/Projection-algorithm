async def _merge_event_streams(
        self,
        adapter_stream: AsyncIterator[str],
        sub_agent_queue: asyncio.Queue[str | None],
        conversation_id: UUID,
        user_id: str,
    ) -> AsyncIterator[str]:
        """Merge events from the main adapter stream and sub-agent event queue.

        This allows sub-agent tools to stream AG-UI events in real-time while
        the main agent continues processing. Events from both sources are
        interleaved as they become available. Also emits keepalive events if the main agent
        is taking too long to respond.

        Args:
            adapter_stream: The main adapter's event stream
            sub_agent_queue: Queue for events emitted by sub-agent tools
            conversation_id: Conversation ID for logging
            user_id: User ID for logging

        Yields:
            AG-UI events as SSE strings from both sources
        """
        try:
            timed_out = False
            task = asyncio.create_task(anext(adapter_stream))

            while True:
                try:
                    if timed_out:
                        # An iteration of `adapter_stream` has already been awaited and timed out.
                        # It might still be running, so wait for a bit.
                        await asyncio.sleep(WAIT_FOR_EVENT_TIMEOUT)
                        if not task.done():
                            # It's still running. Emit a heartbeat.
                            yield ":keepalive\n\n"
                            continue
                        else:
                            # It's done. Reset the timed_out flag.
                            timed_out = False

                    elif not task.done():
                        # First time running an iteration of `adapter_stream`. It may/may not time
                        # out.
                        await asyncio.wait_for(asyncio.shield(task), timeout=WAIT_FOR_EVENT_TIMEOUT)

                    # An iteration of `adapter_stream` has completed and its result is ready.
                    event = task.result()

                    # Yield any pending sub-agent events.
                    async for sub_event in self.drain_queue(sub_agent_queue):
                        yield sub_event

                    # Then yield the main event.
                    self._monitor_run_error_event(event, conversation_id, user_id)
                    yield event

                    # Get the next iteration of `adapter_stream`.
                    task = asyncio.create_task(anext(adapter_stream))

                except TimeoutError:
                    # An iteration of `adapter_stream` has timed out. Emit a heartbeat, and set the
                    # timed_out flag.
                    yield ":keepalive\n\n"
                    timed_out = True

        except StopAsyncIteration:
            pass

        finally:
            await adapter_stream.aclose()

        # After the main stream ends, drain any remaining sub-agent events.
        async for sub_event in self.drain_queue(sub_agent_queue):
            yield sub_event