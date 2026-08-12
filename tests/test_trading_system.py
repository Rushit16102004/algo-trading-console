import os
import sys
import datetime
import pytest

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend_engine.auth import hash_password, verify_password, create_access_token
from backend_engine.kill_switch import get_kill_switch_state, set_kill_switch_state
from backend_engine.market_session import is_market_open, is_exit_only_zone
from backend_engine.risk_engine import risk_engine
from backend_engine.models import User, PaperTrade
from backend_engine.database import SessionLocal, Base, engine

@pytest.fixture(scope="module")
def setup_db():
    from backend_engine.users_db import init_db
    init_db()
    db = SessionLocal()
    # Create test user
    test_user = db.query(User).filter(User.email == "test@gmail.com").first()
    if not test_user:
        test_user = User(
            email="test@gmail.com",
            password_hash=hash_password("111111"),
            role="USER"
        )
        db.add(test_user)
        db.commit()
        db.refresh(test_user)
    yield test_user
    db.close()

def test_password_hashing(setup_db):
    pw = "111111"
    hashed = hash_password(pw)
    assert verify_password(pw, hashed) is True
    assert verify_password("wrong", hashed) is False

def test_kill_switch():
    set_kill_switch_state(True)
    assert get_kill_switch_state() is True
    set_kill_switch_state(False)
    assert get_kill_switch_state() is False

def test_market_session():
    # Verify we can run checks
    status = is_market_open()
    assert isinstance(status, bool)
    exit_status = is_exit_only_zone()
    assert isinstance(exit_status, bool)

def test_risk_engine_blocks_on_kill_switch(setup_db):
    user = setup_db
    # Turn on kill switch
    set_kill_switch_state(True)
    # Check trade execution
    res = risk_engine.can_execute_trade(
        user_id=user.id,
        strategy="243A",
        symbol="NIFTY",
        candle_timestamp="2026-08-12 15:00:00",
        signal_type="BUY"
    )
    assert res["allowed"] is False
    assert res["reason"] == "KILL_SWITCH_ACTIVE"
    # Reset
    set_kill_switch_state(False)
