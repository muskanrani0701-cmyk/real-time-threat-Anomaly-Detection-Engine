import json
import time
import logging
from typing import Optional, Dict, Any, Union
import redis
import fakeredis
import config

logger = logging.getLogger("broker")
logging.basicConfig(level=logging.INFO)

# Global fallback shared server for local execution if Redis daemon is not running
_fallback_server = fakeredis.FakeServer()

class MessageBroker:
    """Decoupled message broker client supporting Redis and in-memory fallback."""

    def __init__(self, host: str = config.REDIS_HOST, port: int = config.REDIS_PORT, db: int = config.REDIS_DB):
        self.host = host
        self.port = port
        self.db = db
        self.client: Union[redis.Redis, fakeredis.FakeStrictRedis] = self._connect()

    def _connect(self) -> Union[redis.Redis, fakeredis.FakeStrictRedis]:
        try:
            r = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                socket_timeout=config.REDIS_SOCKET_TIMEOUT,
                socket_connect_timeout=config.REDIS_CONNECT_TIMEOUT,
                health_check_interval=config.REDIS_HEALTH_CHECK_INTERVAL,
                decode_responses=False
            )
            r.ping()
            logger.info(f"Connected to live Redis server at {self.host}:{self.port}")
            return r
        except Exception as e:
            logger.warning(f"Could not connect to Redis at {self.host}:{self.port} ({e}). Falling back to local in-memory broker.")
            return fakeredis.FakeStrictRedis(server=_fallback_server, decode_responses=False)

    def enqueue_video_frame(self, source_id: str, frame_bytes: bytes, metadata: Optional[Dict[str, Any]] = None) -> float:
        """
        Enqueues binary video frame to video_stream.
        Returns the enqueue latency in milliseconds (target <= 20ms per NFR-1).
        """
        start_time = time.perf_counter()
        meta = metadata or {}
        meta["source_id"] = source_id
        meta["timestamp"] = meta.get("timestamp", time.time())
        meta_json = json.dumps(meta).encode("utf-8")
        
        # Package meta length (4 bytes) + meta json + binary frame
        payload = len(meta_json).to_bytes(4, byteorder="big") + meta_json + frame_bytes
        self.client.rpush(config.VIDEO_QUEUE.encode("utf-8"), payload)
        
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return latency_ms

    def enqueue_network_log(self, log_data: Dict[str, Any]) -> float:
        """
        Enqueues JSON network telemetry to network_stream.
        Returns the enqueue latency in milliseconds (target <= 20ms per NFR-1).
        """
        start_time = time.perf_counter()
        if "timestamp" not in log_data:
            log_data["timestamp"] = time.time()
        payload = json.dumps(log_data).encode("utf-8")
        self.client.rpush(config.NETWORK_QUEUE.encode("utf-8"), payload)
        
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return latency_ms

    def dequeue(self, queue_name: str, timeout: int = 1) -> Optional[bytes]:
        """
        Pops item from queue using blocking pop (BLPOP).
        Prevents busy waiting without sleep calls.
        """
        q_bytes = queue_name.encode("utf-8") if isinstance(queue_name, str) else queue_name
        item = self.client.blpop([q_bytes], timeout=timeout)
        if item is not None:
            return item[1]
        return None

    def publish_alert(self, alert_data: Dict[str, Any]) -> int:
        """Publishes anomaly alert to threat_alerts Pub/Sub channel."""
        payload = json.dumps(alert_data).encode("utf-8")
        return self.client.publish(config.ALERT_CHANNEL.encode("utf-8"), payload)

    def set_temporal_state(self, key: str, value: Dict[str, Any], ttl_seconds: int = 10):
        """Caches intermediate model embeddings/distances for temporal correlation."""
        payload = json.dumps(value).encode("utf-8")
        self.client.set(key.encode("utf-8"), payload, ex=ttl_seconds)

    def get_temporal_state(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieves cached state for temporal correlation."""
        data = self.client.get(key.encode("utf-8"))
        if data:
            return json.loads(data.decode("utf-8"))
        return None

# Singleton instance
broker = MessageBroker()
