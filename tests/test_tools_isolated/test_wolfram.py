import asyncio

from iointel.src.agent_methods.tools_before_rebase.wolfram import query_wolfram_async


def test_wolfram_tool():
    result = asyncio.run(query_wolfram_async('2+2'))
    assert '4' in result

def test_wolfram_tool_solve_equation():
    result = asyncio.run(query_wolfram_async('2x+3=5. Find x'))
    assert 'x = 1' in result

def test_wolfram_real_hard_equation():
    result = asyncio.run(query_wolfram_async('13x^5-7x^4+3x^3+1=0, find approximations for all real solutions'))
    assert '-0.47' in result