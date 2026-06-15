from engine.retrieval.vector_db_utils import clean_metadata


def test_parent_article_text_not_in_pinecone_meta():
    meta = {
        "code_ru": "УК РК",
        "article_number": "136",
        "parent_article_text": "длинный текст...",
    }

    result = clean_metadata(meta)

    assert "parent_article_text" not in result
    assert result["code_ru"] == "УК РК"
