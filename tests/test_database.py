from database.db import check_database_health


def test_database_health():
    assert check_database_health() is True
