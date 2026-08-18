def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_needs_no_model(client):
    """Health must not depend on the model being usable."""
    for _ in range(3):
        assert client.get("/health").status_code == 200
