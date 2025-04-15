import pytest
from iointel.src.agent_methods.tools.wolfram import query_wolfram


@pytest.mark.skip
def test_wolfram_tool():
    result = query_wolfram("2+2")
    assert "4" in result


@pytest.mark.skip
def test_wolfram_tool_solve_equation():
    result = query_wolfram("2x+3=5. Find x")
    assert "x = 1" in result


@pytest.mark.skip
def test_wolfram_real_hard_equation():
    result = query_wolfram(
        "13x^5-7x^4+3x^3+1=0, find approximations for all real solutions"
    )
    assert "-0.47" in result
