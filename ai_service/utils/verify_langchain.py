import sys

print(f"Python executable: {sys.executable}")
try:
    import langchain

    print(f"LangChain version: {langchain.__version__}")
    print("Successfully imported langchain")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
