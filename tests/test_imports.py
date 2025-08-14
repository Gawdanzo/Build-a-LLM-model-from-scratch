def test_import():
    import llm_lab
    assert hasattr(llm_lab, '__version__')
