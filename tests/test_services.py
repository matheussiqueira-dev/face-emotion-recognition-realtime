import asyncio
import threading

import pytest

from app.backend.services import ProcessingBusy, StreamService
from tests.conftest import make_config


class BlockingProcessor:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def run(self, callback):
        self.started.set()
        self.release.wait(timeout=5)

    def stop(self):
        pass  # Simulate a native inference call that cannot be interrupted.


def test_stop_timeout_retains_worker_and_rejects_overlapping_restart():
    async def scenario():
        processor = BlockingProcessor()
        calls = []

        def factory(config):
            calls.append(config)
            return processor

        service = StreamService(make_config(), factory, stop_timeout=0.01)
        try:
            await service._start_processor_async()
            assert await asyncio.to_thread(processor.started.wait, 1)
            original = service._thread
            assert await service.stop_processing() is False
            assert service._thread is original and original.is_alive()
            with pytest.raises(ProcessingBusy):
                await service._start_processor_async()
            with pytest.raises(ProcessingBusy):
                await service.update_config({"max_fps": 20})
            assert service.config.max_fps == 30
            assert len(calls) == 1
            assert calls[0] is not service.config
        finally:
            processor.release.set()
            if service._thread:
                await asyncio.to_thread(service._thread.join, 1)
            await service.shutdown()

    asyncio.run(scenario())


def test_slow_client_does_not_block_fast_client_delivery():
    class Client:
        def __init__(self, slow=False):
            self.slow = slow
            self.sent = asyncio.Event()

        async def send_json(self, payload):
            if self.slow:
                await asyncio.Event().wait()
            self.sent.set()

        async def close(self, code):
            pass

    async def scenario():
        service = StreamService(make_config())
        fast, slow = Client(), Client(slow=True)
        service._clients.update([slow, fast])
        task = asyncio.create_task(service._broadcast_loop())
        try:
            service._frame_queue.put_nowait({"frame": "test"})
            await asyncio.wait_for(fast.sent.wait(), timeout=0.5)
            assert not slow.sent.is_set()
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await service.shutdown()

    asyncio.run(scenario())


def test_metrics_count_every_label_shown_on_the_overlay():
    """The pipeline already applies min_emotion_score to raw model output; smoothed scores can sit below
    the threshold, and re-filtering them would drop labels the dashboard is displaying."""
    from app.core.models import EmotionResult, FaceBox
    from app.core.tracker import Track

    service = StreamService(make_config(min_emotion_score=60.0))
    service._loop = None  # No websocket clients: only the metrics side effect matters.
    track = Track(id=1, box=FaceBox(0, 0, 10, 10), start_time=0.0, last_seen=0.0)
    track.emotion = EmotionResult(dominant="happy", scores={"happy": 55.0, "sad": 45.0}, timestamp=0.0)

    with service._metrics_lock:
        service._register_metrics([track], fps=10.0)

    assert service.get_metrics_snapshot()["dominant_distribution"] == {"happy": 100.0}
