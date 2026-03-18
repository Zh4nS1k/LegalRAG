# fetch_adilet.py — загрузка актуальных кодексов с adilet.zan.kz
# Кодексы меняются; запускайте перед обновлением базы (перед build_vector_db.py).
# УК РК (в т.ч. ст. 120–135 — преступления против несовершеннолетних): K1400000226
# URL: https://adilet.zan.kz/rus/docs/K1400000226

import re
import time
from pathlib import Path

import requests
import urllib3
from bs4 import BeautifulSoup

from ai_service.core import config

# SSL: adilet.zan.kz иногда не проходит верификацию на Mac/Python 3.12
# (промежуточный сертификат в trust store). Для официального сайта verify=False безопасно.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

USER_AGENT = (
    "Mozilla/5.0 (compatible; LegalRAG/1.0; +https://github.com/legalrag)"
)
REQUEST_TIMEOUT = 60
DELAY_BETWEEN_REQUESTS = 1.0  # вежливость к серверу
MAX_RETRIES = 3
RETRY_DELAY = 2.0


def fetch_document(doc_id: str, lang: str = "rus") -> str | None:
    """Загружает HTML страницы документа с Adilet. Возвращает текст ответа или None."""
    url = f"https://adilet.zan.kz/{lang}/docs/{doc_id}"
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT,
                verify=False,  # SSL: обход проблем с сертификатом на Mac/Python 3.12
            )
            r.raise_for_status()
            r.encoding = r.apparent_encoding or "utf-8"
            return r.text
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(f"  Ошибка загрузки {url}: {e}")
                return None
    return None


def extract_text_from_html(html: str) -> str:
    """
    Извлекает основной текст документа из HTML Adilet.
    Пробует типичные контейнеры контента; иначе — тело страницы без скриптов/навигации.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Удаляем скрипты и стили
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    # Типичные селекторы контента на правовых порталах
    for selector in (
        "article",
        ".document-content",
        ".doc-content",
        "#documentContent",
        ".document-body",
        "[class*='document']",
        "main",
    ):
        node = soup.select_one(selector)
        if node and len(node.get_text(strip=True)) > 500:
            return _normalize_text(node.get_text(separator="\n"))

    # Fallback: весь body
    body = soup.find("body")
    if body:
        return _normalize_text(body.get_text(separator="\n"))

    return _normalize_text(soup.get_text(separator="\n"))


def _normalize_text(s: str) -> str:
    """Нормализация пробелов и пустых строк."""
    if not s:
        return ""
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n", "\n\n", s)
    return s.strip()


def main():
    config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Загрузка кодексов с adilet.zan.kz (русский и казахский)...")
    print(f"Документов: {len(config.ADILET_SOURCES) * 2}\n")

    ok = 0
    total = len(config.ADILET_SOURCES)
    for i, (filename, doc_id) in enumerate(config.ADILET_SOURCES, 1):
        # 1. Fetch Russian
        print(f"  [{i}/{total}] {filename} <- {doc_id} (RU) ... ", end="", flush=True)
        html_ru = fetch_document(doc_id, lang="rus")
        if html_ru is None:
            print("пропуск (ошибка загрузки)")
        else:
            text_ru = extract_text_from_html(html_ru)
            if len(text_ru) < 100:
                print("пропуск (мало текста)")
            else:
                out_path = config.DOCUMENTS_DIR / filename
                out_path.write_text(text_ru, encoding="utf-8")
                print(f"OK ({len(text_ru)} символов)")
                ok += 1
        time.sleep(DELAY_BETWEEN_REQUESTS)

        # 2. Fetch Kazakh
        filename_kz = filename.replace(".txt", "_kz.txt")
        print(f"  [{i}/{total}] {filename_kz} <- {doc_id} (KZ) ... ", end="", flush=True)
        html_kz = fetch_document(doc_id, lang="kaz")
        if html_kz is None:
            print("пропуск (ошибка загрузки)")
        else:
            text_kz = extract_text_from_html(html_kz)
            if len(text_kz) < 100:
                print("пропуск (мало текста)")
            else:
                out_path_kz = config.DOCUMENTS_DIR / filename_kz
                out_path_kz.write_text(text_kz, encoding="utf-8")
                print(f"OK ({len(text_kz)} символов)")
                ok += 1
        time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"\nГотово: {ok}/{len(config.ADILET_SOURCES) * 2} документов сохранено в {config.DOCUMENTS_DIR}")
    if ok > 0:
        print("Дальше: python build_vector_db.py  # пересобрать векторную базу")


if __name__ == "__main__":
    main()
