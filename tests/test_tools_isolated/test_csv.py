import os
import pytest
from iointel.src.agent_methods.tools.agno.csv import Csv
from iointel import Agent
from pydantic_ai.models.openai import OpenAIModel


__filename_without_suffix = "test"
__filename = f"{__filename_without_suffix}.csv"


@pytest.fixture
def temp_csv_dir(tmp_path):
    """Create a temporary directory with some test CSV files."""
    csv_dir = tmp_path / "csv_files"
    csv_dir.mkdir()

    # Create a test CSV file
    test_file = csv_dir / __filename
    test_file.write_text("name,age,city\nJohn,30,New York\nJane,25,London")

    return csv_dir


@pytest.fixture
def csv_tool(temp_csv_dir):
    """Create a CSV tool instance with the temporary directory."""
    test_file = temp_csv_dir / __filename
    return Csv(csvs=[test_file])


def test_init_with_csvs(temp_csv_dir):
    """Test initialization with a list of CSV files."""
    test_file = temp_csv_dir / __filename
    csv_tool = Csv(csvs=[test_file])
    assert csv_tool.csvs == [test_file]


def test_init_without_csvs():
    """Test initialization without CSV files."""
    csv_tool = Csv()
    assert csv_tool.csvs is None


def test_list_csv_files(csv_tool):
    """Test the list_csv_files method."""
    result = csv_tool.list_csv_files()
    assert result == [__filename_without_suffix]  # ,data.csv"


def test_read_csv_file_integration(csv_tool):
    """Test reading an actual CSV file from the temporary directory."""
    result = csv_tool.read_csv_file("test")
    assert """{"name": "John", "age": "30", "city": "New York"}""" in result
    assert """{"name": "Jane", "age": "25", "city": "London"}""" in result


def test_get_columns_integration(csv_tool):
    """Test getting columns from an actual CSV file."""
    result = csv_tool.get_columns("test")
    assert result == """["name", "age", "city"]"""


def test_query_csv_file_integration(csv_tool):
    """Test executing SQL queries against a CSV file."""
    # Test a simple SELECT query
    result = csv_tool.query_csv_file(
        "test", "SELECT name, age FROM test WHERE age > 25"
    )
    assert "John" in result

    # # Test a query with aggregation
    result = csv_tool.query_csv_file(
        "test", "SELECT city, COUNT(*) as count FROM test GROUP BY city"
    )
    assert "New York,1" in result
    assert "London,1" in result

    # Test a query with no results
    result = csv_tool.query_csv_file("test", "SELECT * FROM test WHERE age > 100")
    assert "John" not in result
    assert "New York" not in result
    assert "London" not in result


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
async def test_csv_with_agent(temp_csv_dir):
    test_file = temp_csv_dir / __filename
    my_instance = Csv(csvs=[test_file])

    my_agent1 = Agent(
        name="MyAgent",
        instructions="You are a helpful agent that can read CSV files.",
        model=OpenAIModel(
            model_name="gpt-4o-mini",
        ),
        tools=[
            my_instance.read_csv_file,
            my_instance.list_csv_files,
            my_instance.get_columns,
            my_instance.query_csv_file,
        ],
    )
    result1 = await my_agent1.run(
        f"Use the tool to read the content of the file {__filename_without_suffix} and reply with the number of lines."
    )
    assert "" in result1["result"]

    result2 = await my_agent1.run(
        "Use the tools to tell me how many CSV files are available for reading"
    )
    assert "2" in result2["result"]

    result3 = await my_agent1.run("What are the columns in csv file?")
    assert "name" in result3
    assert "age" in result3
    assert "city" in result3
