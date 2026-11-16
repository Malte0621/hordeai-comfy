"""Public worker response fixtures for real SDK metadata lookups."""

from uuid import UUID


def worker_path(worker_id):
    return f"/v2/workers/{worker_id}"


def worker_details(worker_id, *, kind="image", name="local worker", **stats):
    return {
        "id": str(worker_id), "name": name, "type": kind,
        "maintenance_mode": False, "bridge_agent": "AI Horde Worker:1.0:proof",
        "online": True, "performance": "5 generations/s", "threads": 2,
        "uptime": 3600, "requests_fulfilled": 25, "kudos_rewards": 500.0,
        **stats,
    }


def expect_image_worker(server):
    worker_id = str(UUID(int=2000))
    server.expect("GET", worker_path(worker_id), worker_details(worker_id))


def expect_text_worker(server):
    worker_id = str(UUID(int=3000))
    server.expect("GET", worker_path(worker_id), worker_details(worker_id, kind="text", name="local text worker", tokens_generated=1234))
