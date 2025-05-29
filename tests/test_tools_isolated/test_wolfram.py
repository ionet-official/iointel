import pytest
from iointel.src.agent_methods.tools.wolfram import Wolfram


@pytest.mark.skip(reason="Waiting to get a working wolfram API key")
def test_wolfram_tool():
    result = Wolfram().query("2+2")
    assert "4" in result


@pytest.mark.skip(reason="Waiting to get a working wolfram API key")
def test_wolfram_tool_solve_equation():
    result = Wolfram().query("2x+3=5. Find x")
    assert "x = 1" in result


@pytest.mark.skip(reason="Waiting to get a working wolfram API key")
def test_wolfram_real_hard_equation():
    result = Wolfram().query(
        "13x^5-7x^4+3x^3+1=0, find approximations for all real solutions"
    )
    assert "-0.47" in result
