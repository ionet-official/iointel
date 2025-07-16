#!/usr/bin/env python3
"""
Command-line interface for managing prompt templates.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any
from .template_engine import template_engine, TemplateContext
from .instructions import instruction_manager


def format_template_list(templates: list) -> str:
    """Format template list for display."""
    if not templates:
        return "No templates found."
    
    output = []
    for template_name in sorted(templates):
        template = template_engine.get_template(template_name)
        if template:
            # Get first line of template as preview
            first_line = template.template_str.split('\n')[0][:60]
            if len(template.template_str) > 60:
                first_line += "..."
            output.append(f"  {template_name:<20} {first_line}")
        else:
            output.append(f"  {template_name:<20} (not found)")
    
    return "\n".join(output)


def format_template_info(template_name: str) -> str:
    """Format detailed template information."""
    template = template_engine.get_template(template_name)
    if not template:
        return f"Template '{template_name}' not found."
    
    info = []
    info.append(f"Template: {template_name}")
    info.append(f"Name: {template.name}")
    info.append(f"Length: {len(template.template_str)} characters")
    info.append(f"Lines: {len(template.template_str.split())}")
    
    # Show includes
    if template._includes:
        info.append(f"Includes: {', '.join(template._includes)}")
    
    # Show conditionals
    if template._conditionals:
        conditions = [cond for cond, _ in template._conditionals]
        info.append(f"Conditionals: {', '.join(conditions)}")
    
    info.append("\nContent:")
    info.append("-" * 40)
    info.append(template.template_str)
    
    return "\n".join(info)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Manage prompt templates for iointel",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available templates")
    list_parser.add_argument("--type", choices=["all", "instruction", "template"], 
                           default="all", help="Type of templates to list")
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show template details")
    show_parser.add_argument("template_name", help="Template name")
    
    # Render command
    render_parser = subparsers.add_parser("render", help="Render a template")
    render_parser.add_argument("template_name", help="Template name")
    render_parser.add_argument("--context", help="JSON context for template")
    render_parser.add_argument("--var", action="append", nargs=2, metavar=("KEY", "VALUE"),
                              help="Template variable (can be used multiple times)")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new template")
    create_parser.add_argument("template_name", help="Template name")
    create_parser.add_argument("--file", help="Load template from file")
    create_parser.add_argument("--interactive", action="store_true", 
                              help="Create template interactively")
    
    # Edit command
    edit_parser = subparsers.add_parser("edit", help="Edit a template")
    edit_parser.add_argument("template_name", help="Template name")
    edit_parser.add_argument("--editor", default="nano", help="Editor to use")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a template")
    delete_parser.add_argument("template_name", help="Template name")
    delete_parser.add_argument("--confirm", action="store_true", 
                              help="Skip confirmation prompt")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a template")
    validate_parser.add_argument("template_name", help="Template name")
    validate_parser.add_argument("--context", help="JSON context for validation")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test template rendering")
    test_parser.add_argument("template_name", help="Template name")
    test_parser.add_argument("--test-cases", help="JSON file with test cases")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "list":
            if args.type == "instruction":
                templates = instruction_manager.list_instruction_types()
            else:
                templates = template_engine.list_templates()
            
            print(f"Available templates ({len(templates)}):")
            print(format_template_list(templates))
        
        elif args.command == "show":
            print(format_template_info(args.template_name))
        
        elif args.command == "render":
            context = TemplateContext()
            
            # Add context from JSON
            if args.context:
                json_context = json.loads(args.context)
                context.update(**json_context)
            
            # Add variables from command line
            if args.var:
                for key, value in args.var:
                    context.update(**{key: value})
            
            # Render template
            result = template_engine.render_template(args.template_name, context)
            print("Rendered template:")
            print("-" * 40)
            print(result)
        
        elif args.command == "create":
            if args.file:
                # Load from file
                with open(args.file, 'r') as f:
                    content = f.read()
            elif args.interactive:
                # Interactive creation
                print(f"Creating template '{args.template_name}'")
                print("Enter template content (press Ctrl+D when done):")
                lines = []
                try:
                    while True:
                        line = input()
                        lines.append(line)
                except EOFError:
                    content = '\n'.join(lines)
            else:
                print("Error: Must specify --file or --interactive")
                return
            
            # Save template
            template_engine.save_template(args.template_name, content)
            print(f"Template '{args.template_name}' created successfully")
        
        elif args.command == "edit":
            import tempfile
            import subprocess
            
            template = template_engine.get_template(args.template_name)
            if not template:
                print(f"Template '{args.template_name}' not found")
                return
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
                tmp.write(template.template_str)
                tmp_path = tmp.name
            
            try:
                # Open in editor
                subprocess.run([args.editor, tmp_path])
                
                # Read back content
                with open(tmp_path, 'r') as f:
                    new_content = f.read()
                
                # Save updated template
                template_engine.save_template(args.template_name, new_content)
                print(f"Template '{args.template_name}' updated successfully")
            
            finally:
                # Clean up
                Path(tmp_path).unlink(missing_ok=True)
        
        elif args.command == "delete":
            if not args.confirm:
                response = input(f"Delete template '{args.template_name}'? (y/N): ")
                if response.lower() != 'y':
                    print("Cancelled")
                    return
            
            # Remove template file
            template_file = template_engine.templates_dir / f"{args.template_name}.txt"
            if template_file.exists():
                template_file.unlink()
                print(f"Template '{args.template_name}' deleted successfully")
            else:
                print(f"Template file not found: {template_file}")
        
        elif args.command == "validate":
            template = template_engine.get_template(args.template_name)
            if not template:
                print(f"Template '{args.template_name}' not found")
                return
            
            context = TemplateContext()
            if args.context:
                json_context = json.loads(args.context)
                context.update(**json_context)
            
            try:
                result = template.render(context)
                print("Template validation successful")
                print(f"Rendered length: {len(result)} characters")
                
                # Check for unresolved variables
                unresolved = []
                for line in result.split('\n'):
                    if '${' in line:
                        unresolved.append(line.strip())
                
                if unresolved:
                    print("Warning: Unresolved variables found:")
                    for line in unresolved[:5]:  # Show first 5
                        print(f"  {line}")
                    if len(unresolved) > 5:
                        print(f"  ... and {len(unresolved) - 5} more")
            
            except Exception as e:
                print(f"Template validation failed: {e}")
        
        elif args.command == "test":
            if not args.test_cases:
                print("Error: --test-cases file required")
                return
            
            with open(args.test_cases, 'r') as f:
                test_cases = json.load(f)
            
            template = template_engine.get_template(args.template_name)
            if not template:
                print(f"Template '{args.template_name}' not found")
                return
            
            print(f"Running {len(test_cases)} test cases for '{args.template_name}'...")
            
            passed = 0
            failed = 0
            
            for i, test_case in enumerate(test_cases, 1):
                print(f"\nTest case {i}: {test_case.get('name', 'Unnamed')}")
                
                context = TemplateContext()
                context.update(**test_case.get('context', {}))
                
                try:
                    result = template.render(context)
                    expected = test_case.get('expected')
                    
                    if expected and expected not in result:
                        print(f"  FAILED: Expected '{expected}' not found in result")
                        failed += 1
                    else:
                        print(f"  PASSED")
                        passed += 1
                
                except Exception as e:
                    print(f"  ERROR: {e}")
                    failed += 1
            
            print(f"\nResults: {passed} passed, {failed} failed")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()