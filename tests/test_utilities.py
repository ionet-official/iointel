from iointel import pretty_output


def test_pretty_context():
    assert pretty_output.is_enabled, "Pretty output not enabled"
    with pretty_output.disabled() as ctx:
        assert not pretty_output.is_enabled, "Pretty output not disabled"
        assert ctx is pretty_output, "Wrong context manager instance"
    assert pretty_output.is_enabled, "Pretty output not enabled"


def test_pretty_global():
    assert pretty_output.is_enabled, "Pretty output not enabled"
    try:
        pretty_output.global_onoff(False)
        assert not pretty_output.is_enabled, "Pretty output not disabled"
    finally:
        pretty_output.global_onoff(True)
