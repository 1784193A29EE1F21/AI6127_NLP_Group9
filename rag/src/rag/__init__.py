"""AI6127 RAG system: hybrid retrieval + grounded generation over Wikipedia."""

__all__ = ["main"]


def main() -> None:
    """`uv run rag` -> launch the Gradio UI."""
    import runpy
    from pathlib import Path

    runpy.run_path(str(Path(__file__).resolve().parents[2] / "app.py"), run_name="__main__")
