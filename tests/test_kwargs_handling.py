"""Test that functions with **kwargs are handled correctly in function metadata."""
import pytest
from iointel.src.utilities.func_metadata import func_metadata
from typing import Dict, Any, Optional


def sample_function_with_kwargs(
    required_param: str,
    optional_param: str = "default",
    **kwargs
) -> Dict[str, Any]:
    """Test function with kwargs."""
    return {"required": required_param, "optional": optional_param, "kwargs": kwargs}


def sample_function_without_kwargs(
    required_param: str,
    optional_param: str = "default"
) -> Dict[str, Any]:
    """Test function without kwargs."""
    return {"required": required_param, "optional": optional_param}


@pytest.mark.asyncio
async def test_kwargs_handling():
    """Test that **kwargs parameters are properly handled in function metadata."""
    # Test function with **kwargs
    metadata = func_metadata(sample_function_with_kwargs)
    
    # The argument model should not include 'kwargs' as a field
    model_fields = metadata.arg_model.model_fields
    assert 'kwargs' not in model_fields, "kwargs should not be included as a model field"
    assert 'required_param' in model_fields, "required_param should be in model fields"
    assert 'optional_param' in model_fields, "optional_param should be in model fields"
    
    # Test validation - should work with just required param
    test_args = {"required_param": "test"}
    validated_model = metadata.arg_model.model_validate(test_args)
    validated_dict = validated_model.model_dump_one_level()
    
    assert validated_dict["required_param"] == "test"
    assert validated_dict["optional_param"] == "default"
    
    # Test calling the function (kwargs will be empty since we filter unrecognized params)
    result = await metadata.call_fn_with_arg_validation(
        sample_function_with_kwargs, 
        False,  # not async
        test_args,
        None
    )
    
    assert result["required"] == "test"
    assert result["optional"] == "default"
    assert result["kwargs"] == {}  # No extra args passed through


@pytest.mark.asyncio  
async def test_function_without_kwargs():
    """Test that functions without **kwargs still work correctly."""
    metadata = func_metadata(sample_function_without_kwargs)
    
    model_fields = metadata.arg_model.model_fields
    assert 'required_param' in model_fields
    assert 'optional_param' in model_fields
    assert len(model_fields) == 2  # Should only have the two expected fields
    
    # Test validation
    test_args = {"required_param": "test"}
    validated_model = metadata.arg_model.model_validate(test_args)
    validated_dict = validated_model.model_dump_one_level()
    
    assert validated_dict["required_param"] == "test"
    assert validated_dict["optional_param"] == "default"


@pytest.mark.asyncio
async def test_user_input_tool_simulation():
    """Test that user_input tool parameters work correctly."""
    from iointel.src.utilities.registries import TOOLS_REGISTRY
    
    if 'user_input' in TOOLS_REGISTRY:
        tool = TOOLS_REGISTRY['user_input']
        metadata = tool.fn_metadata
        
        # Should not include kwargs as a field
        model_fields = metadata.arg_model.model_fields
        assert 'kwargs' not in model_fields, "user_input tool should not have kwargs field"
        
        # Should include the actual parameters
        assert 'prompt' in model_fields
        assert 'input_type' in model_fields
        assert 'placeholder' in model_fields
        
        # Test validation with the parameters that were failing before
        test_config = {
            'prompt': 'What do you think of the joke?', 
            'input_type': 'text', 
            'placeholder': 'Enter your critique...'
        }
        
        # This should not raise a validation error
        validated_model = metadata.arg_model.model_validate(test_config)
        validated_dict = validated_model.model_dump_one_level()
        
        assert validated_dict['prompt'] == 'What do you think of the joke?'
        assert validated_dict['input_type'] == 'text'
        assert validated_dict['placeholder'] == 'Enter your critique...'


if __name__ == "__main__":
    import asyncio
    
    async def run_tests():
        await test_kwargs_handling()
        await test_function_without_kwargs()
        await test_user_input_tool_simulation()
        print("✅ All kwargs handling tests passed!")
    
    asyncio.run(run_tests())