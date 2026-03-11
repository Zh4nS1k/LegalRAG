# benchmark.py — оценка Legal RAG по фиксированным вопросам (RU/KZ)

"""
Запуск: python benchmark.py

Вопросы задаются в BENCHMARK_QUESTIONS ниже.
Результаты сохраняются в benchmark_results/ (JSON + Excel).
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from ai_service.core import config

# Папка для результатов
config.BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
TEST_QUERIES_PATH = Path(os.environ.get("LEGAL_RAG_TEST_QUERIES_PATH", "test_queries.json"))
DEFAULT_FALLBACK = "Информация не найдена в доступных текстах законов."

# Вопросы для бенчмарка (id, вопрос, язык)
BENCHMARK_QUESTIONS = [
#     {
#         "id": "aslan_illegal_business_kz",
#         "query": """Мәселенің мәні:
#
# Аслан – шағын кәсіпкер, қалада құрылыс материалдарын сатумен айналысады. Ол салықтық жүктемені азайту үшін ресми тіркелмей, яғни лицензиясыз және салық органдарына есеп бермей, ірі көлемде құрылыс материалдарын сатып алып, қайта сатады.
# Бір жыл ішінде Асланның айналымы 500 миллион теңгеден асады, бірақ ол табысын жасырып, салық төлемейді. Мемлекеттік кірістер комитеті жүргізген тексеріс барысында заңсыз кәсіпкерлік қызметінің дәлелдері анықталады.
#
# І. Құқықтық талдау:
# Сұрақ:1. Кейс бойынша қылмыстық құқық бұзушылықтың құрамы бар жоқтығына  құқықтық талдау жасап оны тиісті НҚА сілтеме жасап негіздеңіз.
# 2. Кейс бойынша қылмыстық құқық бұзушылық әрекеті қандай қылмысқа жатады?
# 3. Кейс бойынша қылмыстық құқық бұзушылық әрекетінде қандай қылмыстың құрамы бар?Кейс бойынша қылмыстық құқық бұзушылық іс-әрекетінде ауырлататын және жеңілдететін жағдайлары қандай?
# 4. Қандай қылмыстық жауапкершілік қарастырылған?
# 5. Кейс бойынша қылмыстық құқық бұзушылық іс-әрекетін ескере отырып, қандай жаза тағайындалады?""",
#         "lang": "kz",
#         "description": "Лицензиясыз тіркелмей ірі көлемде сауда жасау, салық төлемеу (заңсыз кәсіпкерлік, УК РК)",
#     },
#     {
#         "id": "gabyt_pyramid_kz",
#         "query": """Мәселенің мәні:
#
# Ғабит жеке кәсіпкер ретінде “KazProfit” атты инвестициялық компанияны ашады. Ол азаматтарға “жоғары пайда әкелетін инвестиция” ұсынатынын айтып, ай сайын 30-50% табыс беретінін жарнамалайды.
# Клиенттер компанияға 5 000 000 теңгеден бастап инвестиция салады, бірақ компанияның нақты экономикалық қызметі жоқ. Ғабит жаңа клиенттерден түскен қаражаттың бір бөлігін ескі салымшыларға пайыз ретінде төлеп, қалған ақшаны жеке қажеттіліктеріне жұмсайды.
# Бір жылдан кейін жаңа салымшылардың ағымы азайып, компания қаржылық міндеттемелерін орындай алмайды. Нәтижесінде 500-ден астам азамат жалпы сомасы 1 миллиард теңгеден аса қаражатынан айырылады. Жәбірленушілер құқық қорғау органдарына арыз түсіреді, ал тергеу барысында “KazProfit” қаржылық пирамида ретінде жұмыс істегені анықталады.
#
# І. Құқықтық талдау:
# Сұрақ:1. Кейс бойынша қылмыстық құқық бұзушылықтың құрамы бар жоқтығына  құқықтық талдау жасап оны тиісті НҚА сілтеме жасап негіздеңіз.
# 2. Кейс бойынша қылмыстық құқық бұзушылық әрекеті қандай қылмысқа жатады?
# 3. Кейс бойынша қылмыстық құқық бұзушылық әрекетінде қандай қылмыстың құрамы бар?Кейс бойынша қылмыстық құқық бұзушылық іс-әрекетінде ауырлататын және жеңілдететін жағдайлары қандай?
# 4. Қандай қылмыстық жауапкершілік қарастырылған?
# 5. Кейс бойынша қылмыстық құқық бұзушылық іс-әрекетін ескере отырып, қандай жаза тағайындалады?""",
#         "lang": "kz",
#         "description": "Қаржылық пирамида (инвестиция уәде, төлемдер жаңа салымнан, ірі залал)",
#     },
    
#     {
#         "id": "smuggling_group_kz",
#         "query": """Мәселенің мәні:
#
# Азамат Дархан бірнеше жылдан бері көлеңкелі бизнесте айналысып келеді. Ол өз таныстары арқылы ұйымдасқан топ құрып, сол топ арқылы заңсыз тауарларды шекарадан өткізіп, контрабандалық жолмен сатады. Дархан топтың құрылымын нақты белгілеп, әр мүшенің рөлін айқындайды (біреуі құжаттарды жасайды, екіншісі кеден қызметкерлерімен келіседі, үшіншісі тауарды өткізеді). Ұйым бірнеше ай бойы әрекет етіп, айтарлықтай табыс табады. Жедел іздестіру шаралары барысында топтың барлық мүшелері ұсталып, Дарханның басшылық еткені дәлелденеді.
#
# І. Құқықтық талдау:
# Сұрақ:1. Кейс бойынша қылмыстық құқық бұзушылықтың құрамы бар жоқтығына  құқықтық талдау жасап оны тиісті НҚА сілтеме жасап негіздеңіз.
# 2. Кейс бойынша қылмыстық құқық бұзушылық әрекеті қандай қылмысқа жатады?
# 3. Кейс бойынша қылмыстық құқық бұзушылық әрекетінде қандай қылмыстың құрамы бар?Кейс бойынша қылмыстық құқық бұзушылық іс-әрекетінде ауырлататын және жеңілдететін жағдайлары қандай?
# 4. Қандай қылмыстық жауапкершілік қарастырылған?
# 5. Кейс бойынша қылмыстық құқық бұзушылық іс-әрекетін ескере отырып, қандай жаза тағайындалады?""",
#         "lang": "kz",
#         "description": "Ұйымдасқан топпен контрабанда (құжат қолдан жасау, кеденмен келісу, өткізу)",
#     },
#     {
#         "id": "fire_safety_negligence_kz",
#         "query": """Мәселенің мәні:
#
# Азамат Сәкен иелігіндегі сауда үйінің ішкі жөндеу жұмыстарын жүргізіп жатқанда, үнемдеу мақсатында кәсіби емес жұмысшыларды жалдайды және электр сымдарын орнату кезінде өрт қауіпсіздігі нормаларын сақтамайды. Ғимаратта өрт сөндіру жүйесі де жұмыс істемейтін болып шығады. Бірнеше аптадан кейін, электр сымдарының қысқа тұйықталуы салдарынан ғимаратта өрт шығып, бірнеше адам жарақат алады және дүкендегі мүлік жанып кетеді. Тергеу нәтижесінде Сәкеннің өрт қауіпсіздігі талаптарын елемей, немқұрайлы қарағаны дәлелденеді.
#
# І. Құқықтық талдау:
# Сұрақ:1. Кейс бойынша қылмыстық құқық бұзушылықтың құрамы бар жоқтығына  құқықтық талдау жасап оны тиісті НҚА сілтеме жасап негіздеңіз.
# 2. Кейс бойынша қылмыстық құқық бұзушылық әрекеті қандай қылмысқа жатады?
# 3. Кейс бойынша қылмыстық құқық бұзушылық әрекетінде қандай қылмыстың құрамы бар?Кейс бойынша қылмыстық құқық бұзушылық іс-әрекетінде ауырлататын және жеңілдететін жағдайлары қандай?
# 4. Қандай қылмыстық жауапкершілік қарастырылған?
# 5. Кейс бойынша қылмыстық құқық бұзушылық іс-әрекетін ескере отырып, қандай жаза тағайындалады?""",
#         "lang": "kz",
#         "description": "Өрт қауіпсіздігі талаптарын бұзу салдарынан өрт және зиян келтіру",
#     }
#     ,
#     {
#         "id": "ambulance_negligence_kz",
#         "query": """Мәселенің мәні:
#
# Азаматша Алия – жедел жәрдем дәрігері. Ол кезекшілік кезінде шақырту алып, бір тұрғын үйге барады. Үйде 65 жастағы ер адам жүрек тұсынан қатты ауырсынуға шағымданады. Алайда Алия науқастың жағдайын “аса ауыр емес” деп бағалап, толық тексеру жүргізбестен, ауруханаға жатқызудан бас тартады. Екі сағаттан кейін науқастың жағдайы күрт нашарлап, жедел инфаркттан қайтыс болады. Медициналық сараптама көрсеткендей, егер дер кезінде медициналық көмек көрсетілгенде, адамның өмірін сақтап қалуға мүмкіндік болған.
#
# І. Құқықтық талдау:
# Сұрақ:1. Кейс бойынша қылмыстық құқық бұзушылықтың құрамы бар жоқтығына  құқықтық талдау жасап оны тиісті НҚА сілтеме жасап негіздеңіз.
# 2. Кейс бойынша қылмыстық құқық бұзушылық әрекеті қандай қылмысқа жатады?
# 3. Кейс бойынша қылмыстық құқық бұзушылық әрекетінде қандай қылмыстың құрамы бар?Кейс бойынша қылмыстық құқық бұзушылық іс-әрекетінде ауырлататын және жеңілдететін жағдайлары қандай?
# 4. Қандай қылмыстық жауапкершілік қарастырылған?
# 5. Кейс бойынша қылмыстық құқық бұзушылық іс-әрекетін ескере отырып, қандай жаза тағайындалады?""",
#         "lang": "kz",
#         "description": "Дәрігердің немқұрайлығы салдарынан өлім (жедел жәрдем, көмек көрсетпеу)",
#     }
#     ,
#     {
#         "id": "industrial_pollution_kz",
#         "query": """Мәселенің мәні:
#
# Азамат Жасулан басшылық ететін шағын зауыт тұрмыстық химия өнімдерін шығарады. Өндіріс процесінде зауыттан шығатын қалдық сулар тиісті тазарту құрылғыларынан өткізілмей, жақын маңдағы өзенге төгіледі. Жасулан шығынды азайту үшін тазарту жүйесін өшіріп тастаған. Бірнеше аптадан кейін өзен суын пайдаланатын ауыл тұрғындары жаппай улану белгілерімен ауруханаға түседі. Экологиялық сараптама судан қауіпті заттар табылғанын растайды. Тексеру нәтижесінде зауыттың экологиялық талаптарды өрескел бұзғаны анықталады.
#
# І. Құқықтық талдау:
# Сұрақ:1. Кейс бойынша қылмыстық құқық бұзушылықтың құрамы бар жоқтығына  құқықтық талдау жасап оны тиісті НҚА сілтеме жасап негіздеңіз.
# 2. Кейс бойынша қылмыстық құқық бұзушылық әрекеті қандай қылмысқа жатады?
# 3. Кейс бойынша қылмыстық құқық бұзушылық әрекетінде қандай қылмыстың құрамы бар?Кейс бойынша қылмыстық құқық бұзушылық іс-әрекетінде ауырлататын және жеңілдететін жағдайлары қандай?
# 4. Қандай қылмыстық жауапкершілік қарастырылған?
# 5. Кейс бойынша қылмыстық құқық бұзушылық іс-әрекетін ескере отырып, қандай жаза тағайындалады?""",
#         "lang": "kz",
#         "description": "Өндірістік қалдық суларды төгу салдарынан жаппай улану (экологиялық талаптарды өрескел бұзу)",
#     }
#     ,
    {
        "id": "red_book_hunting_kz",
        "query": """Мәселенің мәні:

Азаматтар Ержан мен Қанат демалыс күндері таулы аймаққа аңшылыққа шығады. Оларда рұқсат құжаттары (лицензия) жоқ болса да, сирек кездесетін және Қызыл кітапқа енген арқарды атып өлтіреді. Олар өз әрекеттерін әлеуметтік желіде жариялап, мақтанышпен көрсетеді. Бұл видео экология инспекциясының назарына түсіп, тексеру жүргізіледі. Қылмыстық іс қозғалып, Ержан мен Қанаттың табиғатты қорғау заңдарын өрескел бұзғаны анықталады.

І. Құқықтық талдау:
Сұрақ:1. Кейс бойынша қылмыстық құқық бұзушылықтың құрамы бар жоқтығына  құқықтық талдау жасап оны тиісті НҚА сілтеме жасап негіздеңіз.
2. Кейс бойынша қылмыстық құқық бұзушылық әрекеті қандай қылмысқа жатады?
3. Кейс бойынша қылмыстық құқық бұзушылық әрекетінде қандай қылмыстың құрамы бар?Кейс бойынша қылмыстық құқық бұзушылық іс-әрекетінде ауырлататын және жеңілдететін жағдайлары қандай?
4. Қандай қылмыстық жауапкершілік қарастырылған?
5. Кейс бойынша қылмыстық құқық бұзушылық іс-әрекетін ескере отырып, қандай жаза тағайындалады?""",
        "lang": "kz",
        "description": "Лицензиясыз аңшылық, Қызыл кітаптағы арқарды ату",
    }
#     {
#         "id": "labor_principles_ru",
#         "query": "Какие основные принципы трудового законодательства Республики Казахстан?",
#         "lang": "ru",
#         "description": "Принципы трудового законодательства (ТК РК)",
#     },
#     {
#         "id": "criminal_minors_ru",
#         "query": "Какая ответственность по УК РК за преступления против несовершеннолетних (ст. 120–135 УК РК)?",
#         "lang": "ru",
#         "description": "Преступления против несовершеннолетних, УК РК",
#     },
#
#     {
#     "id": "midwife_case_kz",
#     "query": """Мәселенің мәні: Акушерка Амандықова босану бөлімінде жаңа туған екі нәрестенің білезіктерін қасақана ауыстырып, оларды әртүрлі отбасыларға береді. Бұл әрекеттің мақсаты — белгілі бір отбасына ұл бала беру арқылы материалдық сыйақы алу.
#
# І. Құқықтық талдау:
# Сұрақ:1. Кейс бойынша қылмыстық құқық бұзушылықтың құрамы бар жоқтығына құқықтық талдау жасап оны тиісті НҚА сілтеме жасап негіздеңіз.
# 2. Кейс бойынша қылмыстық құқық бұзушылық әрекеті қандай қылмысқа жатады?
# 3. Кейс бойынша қылмыстық құқық бұзушылық әрекетінде қандай қылмыстың құрамы бар? Кейс бойынша қылмыстық құқық бұзушылық іс-әрекетінде ауырлататын және жеңілдететін жағдайлары қандай?
# 4. Қандай қылмыстық жауапкершілік қарастырылған?
# 5. Кейс бойынша қылмыстық құқық бұзушылық іс-әрекетін ескере отырып, қандай жаза тағайындалады?""",
#     "lang": "kz",
#     "description": "Акушерка нәрестелердің білезіктерін ауыстыру (балаларды ауыстыру, УК РК)",
# },
# {
#         "id": "kasenov_child_neglect_kz",
#         "query": """Мәселенің мәні:
# Азамат Касенов өзінің 10 жасар ұлы Марлен-ды тәрбиелеу және күтіп-бағу міндеттерін орындамайды. Ол баласына қажетті тамақ, киім-кешек, медициналық көмек көрсетпейді, сондай-ақ оның білім алуына кедергі жасайды. Соның салдарынан бала денсаулығы мен дамуына зиян келеді.
#
# І. Құқықтық талдау:
# Сұрақ:1. Кейс бойынша қылмыстық құқық бұзушылықтың құрамы бар жоқтығына құқықтық талдау жасап оны тиісті НҚА сілтеме жасап негіздеңіз.
# 2. Кейс бойынша қылмыстық құқық бұзушылық әрекеті қандай қылмысқа жатады?
# 3. Кейс бойынша қылмыстық құқық бұзушылық әрекетінде қандай қылмыстың құрамы бар? Кейс бойынша қылмыстық құқық бұзушылық іс-әрекетінде ауырлататын және жеңілдететін жағдайлары қандай?
# 4. Қандай қылмыстық жауапкершілік қарастырылған?
# 5. Кейс бойынша қылмыстық құқық бұзушылық іс-әрекетін ескере отырып, қандай жаза тағайындалады?""",
#         "lang": "kz",
#         "description": "Ата-ана тәрбие міндеттерін орындамау (ст. 140 УК РК)",
#     },
#     {
#         "id": "e_child_selling_kz",
#         "query": """Мәселенің мәні:
# 25 жастағы азамат Е. қаржылық пайда табу мақсатында 14 жастағы жасөспірім Ж.-ны үшінші тұлғаларға заңсыз жұмысқа орналастыру үшін сатқан.
#
# І. Құқықтық талдау:
# Сұрақ:1. Кейс бойынша қылмыстық құқық бұзушылықтың құрамы бар жоқтығына құқықтық талдау жасап оны тиісті НҚА сілтеме жасап негіздеңіз.
# 2. Кейс бойынша қылмыстық құқық бұзушылық әрекеті қандай қылмысқа жатады?
# 3. Кейс бойынша қылмыстық құқық бұзушылық әрекетінде қандай қылмыстың құрамы бар? Кейс бойынша қылмыстық құқық бұзушылық іс-әрекетінде ауырлататын және жеңілдететін жағдайлары қандай?
# 4. Қандай қылмыстық жауапкершілік қарастырылған?
# 5. Кейс бойынша қылмыстық құқық бұзушылық іс-әрекетін ескере отырып, қандай жаза тағайындалады?""",
#         "lang": "kz",
#         "description": "14 жасар баланы жұмысқа сату (ст. 135 УК РК)",
#     },
#     {
#         "id": "nark_group_kz",
#         "query": """Мәселенің мәні:
# Мұрат, Асқар және Жандос есірткі заттарын заңсыз сатумен айналысатын ұйымдасқан топ құрды. Мұрат топтың жетекшісі болып, есірткі жеткізушілермен келісім жасаған, Асқар заттарды бөлшектеп сатып отырған, ал Жандос сатып алушылармен байланысқа шығып, тапсырыстарды қабылдаған. Полиция оларды ұзақ уақыт бақылап, қылмыс үстінде ұстады.
#
# І. Құқықтық талдау:
# Сұрақ:1. Кейс бойынша қылмыстық құқық бұзушылықтың құрамы бар жоқтығына құқықтық талдау жасап оны тиісті НҚА сілтеме жасап негіздеңіз.
# 2. Кейс бойынша қылмыстық құқық бұзушылық әрекеті қандай қылмысқа жатады?
# 3. Кейс бойынша қылмыстық құқық бұзушылық әрекетінде қандай қылмыстың құрамы бар? Кейс бойынша қылмыстық құқық бұзушылық іс-әрекетінде ауырлататын және жеңілдететін жағдайлары қандай?
# 4. Қандай қылмыстық жауапкершілік қарастырылған?
# 5. Кейс бойынша қылмыстық құқық бұзушылық іс-әрекетін ескере отырып, қандай жаза тағайындалады?""",
#         "lang": "kz",
#         "description": "Ұйымдасқан топпен есірткі сату (ст. 297 УК РК)",
#     }
]


def _normalize_article(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"\d{1,4}", text)
    return match.group(0) if match else text.lower()


def _doc_article(doc: Any) -> str:
    meta = getattr(doc, "metadata", {}) or {}
    for key in ("article_number", "article", "article_id"):
        value = _normalize_article(meta.get(key))
        if value:
            return value
    return ""


def _load_eval_questions() -> List[Dict[str, Any]]:
    if TEST_QUERIES_PATH.exists():
        with open(TEST_QUERIES_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError(f"{TEST_QUERIES_PATH} должен быть JSON-массивом")
        normalized: List[Dict[str, Any]] = []
        for i, item in enumerate(payload, 1):
            if not isinstance(item, dict):
                continue
            query = (item.get("query") or "").strip()
            if not query:
                continue
            relevant_articles = (
                item.get("relevant_articles")
                or item.get("relevant_article_numbers")
                or item.get("ground_truth_articles")
                or []
            )
            normalized.append(
                {
                    "id": item.get("id", f"q_{i:03d}"),
                    "query": query,
                    "lang": item.get("lang", "ru"),
                    "description": item.get("description", ""),
                    "relevant_articles": [_normalize_article(x) for x in relevant_articles if _normalize_article(x)],
                    "expected_answer_snippet": item.get("expected_answer_snippet", ""),
                }
            )
        if normalized:
            print(f"Загружено {len(normalized)} вопросов из {TEST_QUERIES_PATH}")
            return normalized
        print(f"{TEST_QUERIES_PATH} найден, но не содержит валидных вопросов. Используется fallback-список.")
    else:
        print(f"{TEST_QUERIES_PATH} не найден. Используется встроенный список вопросов.")

    fallback: List[Dict[str, Any]] = []
    for item in BENCHMARK_QUESTIONS:
        fallback.append(
            {
                "id": item["id"],
                "query": item["query"],
                "lang": item.get("lang", "ru"),
                "description": item.get("description", ""),
                "relevant_articles": [],
                "expected_answer_snippet": "",
            }
        )
    return fallback


def _compute_retrieval_metrics(
    retrieved_articles: List[str],
    relevant_articles: List[str],
) -> Dict[str, Optional[float]]:
    rel_set = {_normalize_article(x) for x in relevant_articles if _normalize_article(x)}
    ranked = [_normalize_article(x) for x in retrieved_articles if _normalize_article(x)]

    if not rel_set:
        return {
            "precision@5": None,
            "recall@10": None,
            "mrr": None,
            "hit_rate@5": None,
        }

    top5 = ranked[:5]
    top10 = ranked[:10]
    top5_set = set(top5)
    top10_set = set(top10)

    tp5 = len(rel_set & top5_set)
    tp10 = len(rel_set & top10_set)
    precision_at_5 = tp5 / len(top5_set) if top5_set else 0.0
    recall_at_10 = tp10 / len(rel_set) if rel_set else 0.0
    hit_rate_at_5 = 1.0 if tp5 > 0 else 0.0

    reciprocal_rank = 0.0
    for idx, article in enumerate(ranked, 1):
        if article in rel_set:
            reciprocal_rank = 1.0 / idx
            break

    return {
        "precision@5": precision_at_5,
        "recall@10": recall_at_10,
        "mrr": reciprocal_rank,
        "hit_rate@5": hit_rate_at_5,
    }


def evaluate_groundedness(answer: str, context: str, llm: Any) -> float:
    judge_prompt = f"""
Оцени, насколько ответ полностью основан на предоставленном контексте.
Ответь только числом от 0.0 до 1.0, где:
1.0 — все утверждения в ответе имеют прямую поддержку в контексте
0.0 — ответ полностью выдуман или содержит галлюцинации
0.5 — частично основан, частично выдуман

Контекст:
{context[:8000]}

Ответ:
{answer}

Оценка (только число):
""".strip()
    try:
        response = llm.invoke(judge_prompt)
        text = response.content if hasattr(response, "content") else str(response)
        score = float(text.strip().replace(",", "."))
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.5


def _avg(values: List[Optional[float]]) -> Optional[float]:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def run_benchmark(timeout_sec: float = None):
    timeout_sec = timeout_sec or getattr(config, "BENCHMARK_TIMEOUT_SEC", 300)

    print("Загрузка RAG-цепи...")
    try:
        from rag_chain import invoke_qa, retriever, llm
    except Exception as e:
        print(f"Ошибка загрузки rag_chain: {e}")
        sys.exit(1)

    questions = _load_eval_questions()
    use_judge = os.environ.get("LEGAL_RAG_ENABLE_GROUNDEDNESS_JUDGE", "1") == "1"

    results = []
    print(f"\nБенчмарк: {len(questions)} вопросов, таймаут {timeout_sec} с\n")

    for i, item in enumerate(questions, 1):
        qid = item["id"]
        query = item["query"]
        lang = item.get("lang", "ru")
        desc = item.get("description", "")
        relevant_articles = item.get("relevant_articles", [])

        print(f"[{i}/{len(questions)}] {qid} ({lang}) — {desc or query[:60]}...")

        # Retrieval latency and retrieval metrics
        retrieval_start = time.perf_counter()
        try:
            retrieved_docs = retriever.invoke(query)
        except Exception as e:
            print(f"   Ошибка retrieval: {e}")
            retrieved_docs = []
        retrieval_sec = time.perf_counter() - retrieval_start
        retrieved_articles = [_doc_article(d) for d in retrieved_docs if _doc_article(d)]
        retrieval_metrics = _compute_retrieval_metrics(retrieved_articles, relevant_articles)

        # QA end-to-end latency
        qa_start = time.perf_counter()
        try:
            result = invoke_qa(query)
            qa_sec = time.perf_counter() - qa_start
        except Exception as e:
            qa_sec = time.perf_counter() - qa_start
            result = {"result": f"[Ошибка: {e}]", "source_documents": []}
            print(f"   Ошибка: {e}")

        answer = result.get("result", "")
        sources = result.get("source_documents", [])
        judge_sec = 0.0
        groundedness = None
        if use_judge:
            judge_start = time.perf_counter()
            judge_context = "\n\n".join(d.page_content for d in (sources or retrieved_docs)[:10])
            groundedness = evaluate_groundedness(answer=answer, context=judge_context, llm=llm)
            judge_sec = time.perf_counter() - judge_start

        refusal_rate = 1.0 if (answer or "").strip() == DEFAULT_FALLBACK else 0.0

        row = {
            "id": qid,
            "query": query,
            "lang": lang,
            "description": desc,
            "relevant_articles": relevant_articles,
            "retrieved_articles": retrieved_articles[:20],
            "answer": answer,
            "answer_length": len(answer),
            "sources_count": len(sources),
            "precision@5": retrieval_metrics["precision@5"],
            "recall@10": retrieval_metrics["recall@10"],
            "mrr": retrieval_metrics["mrr"],
            "hit_rate@5": retrieval_metrics["hit_rate@5"],
            "groundedness": groundedness,
            "refusal_rate": refusal_rate,
            "latency_retrieval_sec": round(retrieval_sec, 3),
            "latency_generation_sec": round(qa_sec, 3),
            "latency_judge_sec": round(judge_sec, 3),
            "latency_total_sec": round(retrieval_sec + qa_sec + judge_sec, 3),
            "sources": [
                {
                    "source": d.metadata.get("source", ""),
                    "code_ru": d.metadata.get("code_ru", ""),
                    "article_number": d.metadata.get("article_number", ""),
                    "preview": d.page_content[:200].replace("\n", " "),
                }
                for d in sources
            ],
        }
        results.append(row)
        print(
            "   Ответ: "
            f"{len(answer)} символов, источников: {len(sources)}, "
            f"retrieval={row['latency_retrieval_sec']}с, "
            f"generation={row['latency_generation_sec']}с, "
            f"judge={row['latency_judge_sec']}с"
        )

    # Сохранение
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    name_base = f"legal_rag_benchmark_{timestamp}"

    json_path = config.BENCHMARK_DIR / f"{name_base}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        summary = {
            "avg_precision@5": _avg([r.get("precision@5") for r in results]),
            "avg_recall@10": _avg([r.get("recall@10") for r in results]),
            "avg_mrr": _avg([r.get("mrr") for r in results]),
            "avg_hit_rate@5": _avg([r.get("hit_rate@5") for r in results]),
            "avg_groundedness": _avg([r.get("groundedness") for r in results]),
            "avg_refusal_rate": _avg([r.get("refusal_rate") for r in results]),
            "avg_latency_retrieval_sec": _avg([r.get("latency_retrieval_sec") for r in results]),
            "avg_latency_generation_sec": _avg([r.get("latency_generation_sec") for r in results]),
            "avg_latency_judge_sec": _avg([r.get("latency_judge_sec") for r in results]),
            "avg_latency_total_sec": _avg([r.get("latency_total_sec") for r in results]),
        }
        json.dump(
            {
                "timestamp": timestamp,
                "questions_count": len(results),
                "judge_enabled": use_judge,
                "summary": summary,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nРезультаты сохранены: {json_path}")

    # Excel (если есть pandas)
    try:
        import pandas as pd

        df = pd.DataFrame(
            [
                {
                    "id": r["id"],
                    "lang": r["lang"],
                    "description": r["description"],
                    "query": r["query"][:500],
                    "answer": r["answer"],
                    "sources_count": r["sources_count"],
                    "precision@5": r["precision@5"],
                    "recall@10": r["recall@10"],
                    "mrr": r["mrr"],
                    "hit_rate@5": r["hit_rate@5"],
                    "groundedness": r["groundedness"],
                    "refusal_rate": r["refusal_rate"],
                    "latency_retrieval_sec": r["latency_retrieval_sec"],
                    "latency_generation_sec": r["latency_generation_sec"],
                    "latency_judge_sec": r["latency_judge_sec"],
                    "latency_total_sec": r["latency_total_sec"],
                }
                for r in results
            ]
        )
        excel_path = config.BENCHMARK_DIR / f"{name_base}.xlsx"
        df.to_excel(excel_path, index=False, engine="openpyxl")
        print(f"Excel сохранён: {excel_path}")
    except ImportError:
        print("pandas/openpyxl не установлены — Excel не создан")

    # Краткая сводка
    avg_total = _avg([r["latency_total_sec"] for r in results]) or 0.0
    avg_p5 = _avg([r.get("precision@5") for r in results])
    avg_r10 = _avg([r.get("recall@10") for r in results])
    avg_mrr = _avg([r.get("mrr") for r in results])
    avg_h5 = _avg([r.get("hit_rate@5") for r in results])
    avg_grounded = _avg([r.get("groundedness") for r in results])
    avg_refusal = _avg([r.get("refusal_rate") for r in results])

    print(f"\nИтого: {len(results)} вопросов")
    print(f"Средняя общая latency: {avg_total:.2f} с")
    if avg_p5 is not None:
        print(f"Average Precision@5: {avg_p5:.3f}")
        print(f"Average Recall@10: {avg_r10:.3f}")
        print(f"Average MRR: {avg_mrr:.3f}")
        print(f"Average HitRate@5: {avg_h5:.3f}")
    else:
        print("Retrieval-метрики пропущены: в вопросах нет поля relevant_articles.")
    if avg_grounded is not None:
        print(f"Average Groundedness: {avg_grounded:.3f}")
    if avg_refusal is not None:
        print(f"Average Refusal Rate: {avg_refusal:.3f}")
    return results


if __name__ == "__main__":
    run_benchmark()
