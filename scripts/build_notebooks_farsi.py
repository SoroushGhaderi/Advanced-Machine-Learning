#!/usr/bin/env python3
"""Build Farsi copies of the course notebooks.

The script keeps code cells intact, translates markdown cells, and applies a
small set of user-facing text replacements in code cells such as plot titles.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "notebooks"
OUT_DIR = ROOT / "notebooks_farsi"


FAIRY_SETUP_CELL = {
    "cell_type": "code",
    "metadata": {"tags": ["hide-input"]},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Persian text rendering for notebook markdown and plots\n",
        "import matplotlib as mpl\n",
        "\n",
        "mpl.rcParams.update({\n",
        '    \"font.family\": \"sans-serif\",\n',
        '    \"font.sans-serif\": [\n',
        '        \"Vazirmatn\",\n',
        '        \"Vazir\",\n',
        '        \"IRANSans\",\n',
        '        \"Noto Sans Arabic\",\n',
        '        \"Noto Naskh Arabic\",\n',
        '        \"DejaVu Sans\",\n',
        '    ],\n',
        '    \"axes.unicode_minus\": False,\n',
        "})\n",
        "\n",
        "try:\n",
        "    from IPython.display import HTML, display\n",
        "\n",
        "    display(HTML('<div dir=\"rtl\" style=\"text-align: right\"></div>'))\n",
        "except Exception:\n",
        "    pass\n",
    ],
}

RTL_STYLE_CELL = {
    "cell_type": "markdown",
    "metadata": {"tags": ["hide-input"]},
    "source": [
        "<style>\n",
        "/* Improve RTL readability for notebook markdown in JupyterLab/classic */\n",
        ".jp-RenderedMarkdown,\n",
        ".rendered_html,\n",
        "div.text_cell_render,\n",
        "div.output_area,\n",
        ".jp-MarkdownOutput {\n",
        "  direction: rtl;\n",
        "  text-align: right;\n",
        "}\n",
        "\n",
        ".jp-RenderedMarkdown table,\n",
        ".rendered_html table,\n",
        "div.text_cell_render table {\n",
        "  direction: rtl;\n",
        "  text-align: right;\n",
        "}\n",
        "\n",
        ".jp-RenderedMarkdown li,\n",
        ".rendered_html li,\n",
        "div.text_cell_render li {\n",
        "  margin-right: 1.2em;\n",
        "}\n",
        "</style>\n",
    ],
}


TRANSLATIONS: dict[str, dict[int, str]] = {
    "00_course_setup_and_dataset.ipynb": {
        0: """# 00 — راه‌اندازی دوره و مجموعه‌داده

**زمان تخمینی:** 60–90 دقیقه
**پیش‌نیازها:** آشنایی پایه با Python، pandas، scikit-learn، و آمار مقدماتی.
**وابسته به:** هیچ‌چیز.

## اهداف یادگیری

- مسئله پیش‌بینی را در لحظه درست تصمیم‌گیری صورت‌بندی کنید.
- مجموعه‌داده‌های عمومی را با معیارهای فنی و آموزشی مقایسه کنید.
- داده UCI Bank Marketing را به‌صورت بازتولیدپذیر از فایل‌های محلی بارگذاری و cache کنید.
- تفاوت data catalog و data dictionary را توضیح دهید.
- یک catalog ساده در سطح dataset و field بسازید.

سؤال اصلی این دوره این است: **پیش از برقراری تماس، کدام مشتری‌ها باید برای کمپین بازاریابی سپرده مدت‌دار در اولویت قرار بگیرند؟** این یک مسئله رتبه‌بندی و انتخاب است، نه مسئله استنتاج علّی.

**یادداشت آموزشی:** دانشجوها معمولاً مستقیم سراغ تنظیم مدل می‌روند. اینجا عمداً آهسته‌تر پیش می‌رویم و انتخاب مجموعه‌داده، لحظه پیش‌بینی، و طراحی split را به‌عنوان گام‌های اصلی مدل‌سازی در نظر می‌گیریم.""",
        1: """### توضیحات ستون‌ها

| ستون | توضیح |
|---|---|
| `age` | سن مشتری بر حسب سال. |
| `job` | دسته شغلی. |
| `marital` | وضعیت تأهل. |
| `education` | سطح تحصیلات، شامل `unknown` وقتی ثبت نشده باشد. |
| `default` | این‌که آیا مشتری نکول اعتباری دارد یا نه. |
| `balance` | میانگین مانده سالانه حساب بانکی، به یورو. |
| `housing` | این‌که آیا مشتری وام مسکن دارد یا نه. |
| `loan` | این‌که آیا مشتری وام شخصی دارد یا نه. |
| `contact` | نوع ارتباط استفاده‌شده برای این تماس. |
| `day` | روز آخرین تماس در ماه. |
| `month` | ماه آخرین تماس. |
| `duration` | مدت تماس، بر حسب ثانیه. |
| `campaign` | تعداد تماس‌ها در این کمپین برای این مشتری. |
| `pdays` | تعداد روز از تماس قبلی؛ `-1` یعنی قبلاً تماسی وجود نداشته است. |
| `previous` | تعداد تماس‌های قبلی قبل از این کمپین. |
| `poutcome` | نتیجه کمپین قبلی. |
| `y` | این‌که آیا مشتری اشتراک سپرده مدت‌دار را پذیرفته است یا نه. |""",
        2: """### چرا این مجموعه‌داده؟

ما به مجموعه‌داده‌ای نیاز داریم که برای آموزش عملی مناسب باشد:

- یک target دودویی و واقعی،
- ستون‌های عددی و categorical ترکیبی،
- ریسک نشتِ روشن،
- اندازه‌ای که الگوهای اعتبارسنجی را نشان دهد،
- و یک زمینه کسب‌وکاری که دانشجو بتواند آن را توضیح دهد.

| مجموعه‌داده | چرا مناسب است | ایراد اصلی |
|---|---|---|
| UCI Bank Marketing | داده واقعی کمپین، نامتوازن بودن کلاس‌ها، و ریسک نشت واضح | `duration` اطلاعات پس از تماس است |
| UCI Adult | برای تمرین preprocessing مفید است | کمتر با workflow تصمیم‌گیری کمپین هم‌راستا است |
| UCI Online Shoppers | مسئله تبدیل ساده و قابل‌فهم | برای تاریخچه کمپین و حکمرانی آموزشیِ کمتری دارد |

**توصیه:** از Bank Marketing استفاده کنید. این مجموعه‌داده مسیر تمیزی برای کاتالوگ‌سازی، بررسی نشت، preprocessing، مدل پایه، و قضاوت عملی فراهم می‌کند.""",
        2: """### چرا هزینه به این شکل وزن‌دهی شده است

مقدار `cost` در `classification_metrics` یک ابتکار مخصوص این دوره است، نه یک فرمول آماری استاندارد. آن را این‌جا می‌گذاریم چون ارزیابی باید لحظه تصمیم‌گیری، یعنی قبل از انجام تماس، را منعکس کند.

منفی‌های کاذب وزن بیشتری می‌گیرند چون از دست دادن یک مشتری بالقوه به معنای از دست دادن فرصت بازاریابی کمیاب و احتمال درآمد است. مثبت‌های کاذب هم هزینه دارند، اما معمولاً فقط به معنی یک تماس تلف‌شده هستند، نه از دست رفتن یک تبدیل.""",
        4: """## کاتالوگ داده: قابل‌جست‌وجو و قابل‌حاکمیت کردن مجموعه‌داده

یک **کاتالوگ داده** اطلاعات سطح دارایی را نگه می‌دارد: این مجموعه‌داده چیست، چه کسی مالک آن است، از کجا آمده، آخرین بار چه زمانی به‌روزرسانی شده، و با چه سیاستی باید استفاده شود. یک **دیکشنری داده** محدودتر است و فقط فیلدهای داخل آن دارایی را توضیح می‌دهد: نام ستون، تعریف، نوع داده، دامنهٔ مقادیر، و قواعد اعتبار.

در این درس، تفاوت را این‌گونه خلاصه می‌کنیم:

1. کاتالوگ دربارهٔ «کل دارایی» است.
2. دیکشنری داده دربارهٔ «ستون‌ها و فیلدها» است.
3. کاتالوگ برای کشف، اعتماد، مالکیت، و حاکمیت استفاده می‌شود.
4. دیکشنری داده برای تفسیر دقیق معنا و قواعد هر فیلد استفاده می‌شود.

![مقایسهٔ کاتالوگ داده و دیکشنری داده: کاتالوگ در سطح دارایی عمل می‌کند و دیکشنری در سطح فیلد، تا هم حاکمیت و هم تفسیر دقیق ممکن شود.](../assets/data_catalog_learning_flow.png)

**یادداشت آموزشی:** کاتالوگ فقط مستندات نیست. بخشی از قرارداد مدل‌سازی است.""",
        5: """## کاتالوگ داده: نمونه اجرایی

```python
asset_catalog = pd.Series({
    "asset_name": "uci_bank_marketing",
    "local_path": str(path.relative_to(ROOT)),
    "source_url": "https://archive.ics.uci.edu/dataset/222/bank+marketing",
    "sha256": file_sha256(path),
    "rows": len(raw),
    "columns": raw.shape[1],
    "grain": "one row per marketing contact",
    "target": TARGET,
    "prediction_moment": "immediately before a scheduled call",
    "refresh_policy": "static course snapshot",
    "owner": "course maintainer",
    "steward": "simulated marketing-data steward",
}, name="value").to_frame()
display(asset_catalog)
```
""",
        6: """## کاتالوگ فیلد و دیکشنری داده

UCI مقدار `unknown` را یک دسته واقعی در نظر می‌گیرد؛ آن را بی‌سر‌و‌صدا به مقدار گمشده تبدیل نمی‌کنیم. `pdays == -1` یک مقدار نشانه‌ای است که یعنی مشتری قبلاً تماس نداشته است. `y=1` یعنی مشتری اشتراک را پذیرفته است. چندین فیلد به کمپین جاری مربوط‌اند؛ بنابراین باید در لحظه پیش‌بینی بررسی شوند، نه صرفاً بر اساس نام ستون‌ها.

کاتالوگ فیلد در ادامه، دو ویژگی حکمرانی را به خلاصهٔ معمول dtype/cardinality اضافه می‌کند:

- نقش هر فیلد در فرایند،
- و این‌که آیا فیلد در زمان پیش‌بینی در دسترس است یا نه.

این کار جدول را به یک feature contract تبدیل می‌کند، نه فقط یک schema dump.""",
        9: """### حداقل قرارداد کیفیت

توصیف کاتالوگ‌ها قول‌اند، نه شواهد. یک کاتالوگ مفید هر قول مهم را به یک check وصل می‌کند. برای این snapshot، ما به یک target دودویی و غیر null، سن‌های معقول، فیلدهای تقویمی معتبر، و sentinel مستند `pdays` نیاز داریم.

**هشدار:** اگر یک قانون حیاتی در داده production شکست، پاسخ درست این است که ingestion را fail کنیم یا batch را quarantine کنیم. قانون را برای عبور دادن pipeline بی‌سر‌و‌صدا ضعیف نکنید.""",
        11: """## ممیزی نشت: چرا `duration` باید حذف شود

`duration` فقط در طول یا بعد از تماس معلوم می‌شود، پس یک سیستم پیش از تماس نمی‌تواند از آن استفاده کند. این یک دام نشت کلاسیک است: ویژگی بسیار پیش‌بینی‌کننده به نظر می‌رسد، اما فقط به این دلیل که اطلاعاتِ رویدادی را که می‌خواهیم پیش‌بینی کنیم در خود دارد.

سلول بعدی یک pipeline ساده logistic regression را با و بدون `duration` مقایسه می‌کند. اگر امتیاز خیلی بالا برود، این یک هشدار است نه یک پیروزی استقرار.""",
        13: """**تفسیر:** `duration` معمولاً یک بهبود ظاهری چشمگیر ایجاد می‌کند چون تماس‌های موفق معمولاً طولانی‌تر هستند.

ریسک‌های دیگر ظریف‌ترند:

- `campaign` شامل تماس فعلی است،
- `day` و `month` تاریخ تماس فعلی را مشخص می‌کنند،
- و `pdays`، `previous`، `poutcome` تاریخچه تماس‌های قبلی را توصیف می‌کنند.

ما این فیلدها را فقط با این فرض صریح نگه می‌داریم که scoring دقیقاً پیش از یک تماس برنامه‌ریزی‌شده انجام می‌شود. لحظه تصمیم را تغییر دهید، قرارداد ویژگی هم باید تغییر کند.""",
        14: """## قرارداد split

- **توسعه (60%)**: fit، cross-validation، انتخاب ویژگی/مدل/hyperparameter.
- **اعتبارسنجی (20%)**: انتخاب آستانه عملیاتی و مقایسه محدود نهایی‌ها.
- **آزمون (20%)**: تا notebook 09 قفل می‌ماند؛ فقط یک‌بار برای ارزیابی نهایی استفاده می‌شود.

یک split زمانی واقعی برای پیش‌بینی استقرار بهتر است، اما این فایل سال و timestamp کامل و پایدار ندارد. split گروهی برای مشتریان تکراری کمک می‌کرد، اما شناسه مشتری در داده موجود نیست.

**هشدار بهترین‌عملکرد:** از validation یا test برای انتخاب ویژگی‌ها، آستانه‌ها، preprocessing، یا خانواده مدل استفاده نکنید. این همان جایی است که دفترچه‌ها ناخواسته کمتر از آنچه به نظر می‌رسند قابل‌اعتماد می‌شوند.""",
        16: """## یک baseline عملی

یک دفترچه‌ی آموزشی خوب نباید در مشاهده‌ی داده متوقف شود. ما همچنین به یک baseline کوچک و قابل‌استقرار نیاز داریم که دانشجوها بتوانند آن را بفهمند، تغییر دهند و با مدل‌های بعدی مقایسه کنند.

baseline زیر از این موارد استفاده می‌کند:

- فقط ستون‌های امن از نظر نشت،
- یک preprocessing pipeline استاندارد،
- و logistic regression به‌عنوان یک مدل اولیه‌ی مرسوم.

این انتخاب عمداً ساده است. اگر یک روش پیچیده نتواند از این setup بهتر شود، باید در اضافه‌کردن پیچیدگی محتاط باشیم.""",
        18: """## ارزیابی و تفسیر

برای مسئله‌های نامتوازن، accuracy به‌تنهایی معمولاً گمراه‌کننده است. ما این معیارها را ترجیح می‌دهیم:

- log loss برای کیفیت احتمال،
- balanced accuracy برای یک check عملیاتی ساده،
- و شمارش‌های شبیه confusion matrix وقتی بخواهیم دید عملیاتی از خطاها داشته باشیم.

**یادداشت آموزشی:** اگر دانشجوها بپرسند چرا این‌جا از test استفاده نمی‌کنیم، این نشانه‌ی خوبی است. پاسخ این است که هنوز در حال انتخاب workflow هستیم، پس روی داده‌ی توسعه می‌مانیم.""",
        20: """## توصیه‌ی عملی

از Bank Marketing به‌عنوان مجموعه‌داده‌ی درس استفاده کنید، اما قرارداد پیش‌بینی را سخت‌گیرانه نگه دارید:

- `duration` را حذف کنید،
- مشخص کنید هر فیلد چه زمانی در دسترس است،
- از pipeline استفاده کنید تا preprocessing داخل cross-validation بماند،
- و test set را تا دفترچه‌ی ارزیابی نهایی مهروموم نگه دارید.

اگر بعداً مدل را بهتر کردیم، بهبود اصلی باید از طراحی بهتر ویژگی، انتخاب آستانه و discipline در اعتبارسنجی بیاید، نه از پنهان‌کردن نشت.""",
        21: """## تمرین‌ها

1. کاتالوگ دارایی را برای یک محیط production با یک owner مشخص و freshness SLA بازنویسی کنید.
2. یک قانون بین‌فیلدی شامل `pdays`، `previous` و `poutcome` اضافه کنید.
3. لحظه‌ی پیش‌بینی را به «یک هفته قبل از کمپین» تغییر دهید و مشخص کنید کدام فیلدها دیگر در دسترس نیستند.
4. logistic regression را با dummy classifier جایگزین کنید و توضیح دهید چرا هنوز یک baseline مفید است.
5. یک معیار ارزیابی برای ذی‌نفع بازاریابی و یک معیار برای مهندس ML پیشنهاد کنید.

## خلاصه

ما یک مجموعه‌داده انتخاب کردیم، قراردادش را مستند کردیم، قواعد کیفیت پایه را بررسی کردیم، نشت را ممیزی کردیم و یک baseline کوچک و بدون نشت ساختیم. این کار بقیه‌ی درس را با زبان مشترکی برای ویژگی‌ها، splitها و ارزیابی آماده می‌کند.

## منابع

- [مجموعه‌داده UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- [مقاله‌ی مجموعه‌داده (Decision Support Systems)](https://doi.org/10.1016/j.dss.2014.03.001)
- [Data Catalog Vocabulary (DCAT) v3](https://www.w3.org/TR/vocab-dcat-3/)
- [مدیریت مدل در scikit-learn](https://scikit-learn.org/stable/model_selection.html)""",
        14: """**تفسیر:** `duration` معمولاً به‌دلیل آن‌که تماس‌های موفق را خیلی آشکار می‌کند، یک بهبود ظاهری چشمگیر ایجاد می‌کند. اما این بهبود واقعی نیست، چون این متغیر در زمان تصمیم‌گیری در دسترس نیست.

ریسک‌های ظریف‌تر هم وجود دارند:

- `campaign` شامل تماس فعلی است،
- `day` و `month` تاریخ تماس فعلی را مشخص می‌کنند،
- و `pdays`، `previous`، `poutcome` تاریخچه تماس‌های قبلی را توصیف می‌کنند.

ما این فیلدها را فقط با این فرض نگه می‌داریم که scoring دقیقاً پیش از یک تماس برنامه‌ریزی‌شده انجام می‌شود. اگر لحظه تصمیم‌گیری تغییر کند، قرارداد ویژگی‌ها هم باید تغییر کند.""",
        15: """## قرارداد split

- **توسعه (60%)**: fit، cross-validation، انتخاب ویژگی/مدل/hyperparameter.
- **اعتبارسنجی (20%)**: انتخاب آستانه عملیاتی و مقایسه محدود نهایی‌ها.
- **آزمون (20%)**: تا دفترچه 09 قفل می‌ماند؛ فقط یک‌بار برای ارزیابی نهایی استفاده می‌شود.

یک split زمانی واقعی برای پیش‌بینی استقرار بهتر است، اما این فایل سال و timestamp کامل و پایدار ندارد. split گروهی برای مشتریان تکراری مفید بود، اما شناسه مشتری در داده موجود نیست.

**هشدار بهترین‌عملکرد:** از validation یا test برای انتخاب ویژگی‌ها، آستانه‌ها، preprocessing، یا خانواده مدل استفاده نکنید. این همان جایی است که دفترچه‌ها ناخواسته کمتر از آنچه به نظر می‌رسند قابل‌اعتماد می‌شوند.""",
        16: """## یک baseline عملی

یک دفترچه آموزشی خوب نباید در ممیزی داده متوقف شود. ما همچنین یک baseline کوچک و قابل‌استقرار می‌خواهیم که دانشجوها بتوانند آن را بفهمند، تغییر دهند، و با مدل‌های بعدی مقایسه کنند.

baseline زیر از این موارد استفاده می‌کند:

- فقط ستون‌های بدون نشت،
- یک pipeline استاندارد preprocessing،
- و logistic regression به‌عنوان یک مدل اولیه مرسوم.

این انتخاب عمداً ساده است. اگر یک روش پیچیده نتواند از این setup بهتر شود، باید در اضافه‌کردن پیچیدگی محتاط باشیم.""",
        18: """## ارزیابی و تفسیر

برای مسائل نامتوازن، accuracy به‌تنهایی معمولاً گمراه‌کننده است. ما این‌ها را ترجیح می‌دهیم:

- ROC AUC برای کیفیت رتبه‌بندی،
- PR AUC برای شناسایی رخدادهای مثبت کمیاب،
- log loss برای کیفیت احتمال،
- و یک معیار آستانه‌دار مثل balanced accuracy برای یک check عملیاتی ساده.

**یادداشت آموزشی:** اگر دانشجوها بپرسند چرا اینجا از test استفاده نمی‌کنیم، این نشانه خوبی است. پاسخ این است که هنوز در حال انتخاب workflow هستیم، پس روی داده توسعه می‌مانیم.""",
        20: """## توصیه عملی

از Bank Marketing به‌عنوان dataset دوره استفاده کنید، اما قرارداد پیش‌بینی را سخت‌گیرانه نگه دارید:

- `duration` را حذف کنید،
- مشخص کنید هر فیلد چه زمانی در دسترس است،
- از pipeline استفاده کنید تا preprocessing داخل cross-validation بماند،
- و test را تا notebook ارزیابی نهایی قفل نگه دارید.

اگر بعداً مدل را بهتر کردیم، بهبود اصلی باید از طراحی بهتر ویژگی، انتخاب آستانه، و discipline در اعتبارسنجی بیاید، نه از پنهان‌کردن نشت.""",
        21: """## تمرین‌ها

1. کاتالوگ دارایی را برای یک محیط production با یک owner مشخص و freshness SLA بازنویسی کنید.
2. یک قانون بین‌فیلدی شامل `pdays`، `previous`، و `poutcome` اضافه کنید.
3. لحظه پیش‌بینی را به «یک هفته قبل از کمپین» تغییر دهید و مشخص کنید کدام فیلدها دیگر در دسترس نیستند.
4. logistic regression را با dummy classifier جایگزین کنید و توضیح دهید چرا هنوز یک baseline مفید است.
5. یک معیار ارزیابی برای ذی‌نفع بازاریابی و یک معیار برای مهندس ML پیشنهاد کنید.

## خلاصه

ما یک مجموعه‌داده انتخاب کردیم، قراردادش را مستند کردیم، قواعد کیفیت پایه را بررسی کردیم، نشت را ممیزی کردیم، و یک baseline کوچک و بدون نشت ساختیم. این کار بقیه دوره را با زبان مشترکی برای ویژگی‌ها، splitها، و ارزیابی آماده می‌کند.

## منابع

- [مجموعه‌داده UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- [مقاله مجموعه‌داده (Decision Support Systems)](https://doi.org/10.1016/j.dss.2014.03.001)
- [Data Catalog Vocabulary (DCAT) v3](https://www.w3.org/TR/vocab-dcat-3/)
- [مدیریت مدل در scikit-learn](https://scikit-learn.org/stable/model_selection.html)""",
        22: """## توصیه‌ی عملی

از Bank Marketing به‌عنوان مجموعه‌داده‌ی درس استفاده کنید، اما قرارداد پیش‌بینی را سخت‌گیرانه نگه دارید:

- `duration` را حذف کنید،
- مشخص کنید هر فیلد چه زمانی در دسترس است،
- از pipeline استفاده کنید تا preprocessing داخل cross-validation بماند،
- و test set را تا دفترچه‌ی ارزیابی نهایی مهروموم نگه دارید.

اگر بعداً مدل را بهتر کردیم، بهبود اصلی باید از طراحی بهتر ویژگی، انتخاب آستانه و discipline در اعتبارسنجی بیاید، نه از پنهان‌کردن نشت.""",
        23: """## تمرین‌ها

1. کاتالوگ دارایی را برای یک محیط production با یک owner مشخص و freshness SLA بازنویسی کنید.
2. یک قانون بین‌فیلدی شامل `pdays`، `previous` و `poutcome` اضافه کنید.
3. لحظه‌ی پیش‌بینی را به «یک هفته قبل از کمپین» تغییر دهید و مشخص کنید کدام فیلدها دیگر در دسترس نیستند.
4. logistic regression را با dummy classifier جایگزین کنید و توضیح دهید چرا هنوز یک baseline مفید است.
5. یک معیار ارزیابی برای ذی‌نفع بازاریابی و یک معیار برای مهندس ML پیشنهاد کنید.

## خلاصه

ما یک مجموعه‌داده انتخاب کردیم، قراردادش را مستند کردیم، قواعد کیفیت پایه را بررسی کردیم، نشت را ممیزی کردیم و یک baseline کوچک و بدون نشت ساختیم. این کار بقیه‌ی درس را با زبان مشترکی برای ویژگی‌ها، splitها و ارزیابی آماده می‌کند.

## منابع

- [مجموعه‌داده UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- [مقاله‌ی مجموعه‌داده (Decision Support Systems)](https://doi.org/10.1016/j.dss.2014.03.001)
- [Data Catalog Vocabulary (DCAT) v3](https://www.w3.org/TR/vocab-dcat-3/)
- [مدیریت مدل در scikit-learn](https://scikit-learn.org/stable/model_selection.html)""",
    },
    "01_gradient_boosting_fundamentals.ipynb": {
        0: """# 01 — مبانی Gradient Boosting

**زمان تخمینی:** 100–130 دقیقه
**پیش‌نیازها:** دفترچه 00، درخت تصمیم، splitهای آموزش/اعتبارسنجی، و معیارهای پایه طبقه‌بندی.
**داده:** داده محلی Bank Marketing از `data/raw/bank-full.csv`.
**تم عملی:** یک baseline قوی بسازید و بعد تصمیم بگیرید آیا gradient boosting پیچیدگی اضافه‌اش را توجیه می‌کند یا نه.

## اهداف یادگیری

در پایان این دفترچه، دانشجو باید بتواند:

- gradient boosting را به‌صورت دنباله‌ای از اصلاح‌های پیاپی توضیح دهد.
- preprocessing را داخل pipeline نگه دارد.
- بین baselineهای عملی مقایسه منصفانه انجام دهد.
- overfitting را با استفاده از validation داخلی تشخیص دهد.
- نتیجه مدل را به زبان trade-offهای واقعی کسب‌وکار بخواند.""",
        1: """## مفهوم: boosting به‌عنوان اصلاح ترتیبی

یک درخت تصمیم کم‌عمق یک weak learner است: به‌تنهایی عمداً محدود نگه داشته می‌شود. Gradient boosting چند weak learner را **به‌صورت ترتیبی** ترکیب می‌کند. هر درخت جدید طوری آموزش می‌بیند که loss فعلی مدل را کاهش دهد.

برای regression با squared error، هدف اصلاح به‌خوبی دیده می‌شود: residual.

\\[
F_m(x) = F_{m-1}(x) + \\eta h_m(x)
\\]

`F_m` ensemble فعلی است، `h_m` درخت جدید است، و `η` learning rate است. درخت جدید کاملِ مدل نیست؛ فقط یک اصلاح کوچک و کنترل‌شده به خروجی قبلی اضافه می‌کند.""",
        2: """### نمای کلی workflow گرادیان بوستینگ

![نمای فرآیند گرادیان بوستینگ: با یک پیش‌بینی ثابت شروع می‌شود، خطاهای احتمالی محاسبه می‌شوند، یک درخت اصلاحی کم‌عمق fit می‌شود، خروجی آن کوچک‌نمایی می‌شود، ensemble به‌روزرسانی می‌شود و این چرخه تکرار می‌شود.](../assets/gradient_boosting_process.png)

فرآیند را از چپ به راست بخوانید و سپس حلقه را پس از مرحله به‌روزرسانی دنبال کنید. درخت‌های random forest مستقل آموزش می‌بینند؛ اما درخت‌های boosting یک زنجیره مرتب می‌سازند، چون هر درخت به ensemble فعلی وابسته است.""",
        4: """## بارگذاری داده

ما از نسخه محلی داده Bank Marketing استفاده می‌کنیم. هدف این است که ببینیم آیا مشتری یک سپرده مدت‌دار را پذیرفته است یا نه.

مجموعه‌داده اصلی شامل `duration`، یعنی مدت تماس، است. این متغیر پیش‌بینی‌کننده بسیار قوی‌ای است اما قبل از تماس در دسترس نیست، بنابراین استفاده از آن یک مدل غیرواقعی برای برنامه‌ریزی کمپین می‌سازد. helper بالا آن را به‌صورت پیش‌فرض حذف می‌کند.""",
        6: """از آن‌جا که کلاس مثبت نسبتاً کمیاب است، یک مدل می‌تواند از نظر accuracy خوب به نظر برسد ولی همچنان بیشتر مشتریان بالقوه را از دست بدهد. ما accuracy، balanced accuracy، F1 و log loss را گزارش می‌کنیم.

**یادداشت آموزشی:** از دانشجوها بپرسید اگر هر مشتری از دست‌رفته بسیار گران‌تر از هر تماس فروش اضافی باشد، کدام معیار را انتخاب می‌کنند.""",
        7: """## پیش‌پردازش

گام preprocessing مقادیر گمشده عددی را impute می‌کند، ستون‌های عددی را scale می‌کند، و ستون‌های categorical را one-hot encode می‌کند. scaling برای مدل‌های درختی ضروری نیست، اما اجازه می‌دهد همان pipeline برای logistic regression هم کار کند، و این مقایسه baseline را منصفانه و ساده نگه می‌دارد.

نکته مهم این است که preprocessing داخل pipeline مدل قرار دارد. در cross-validation، هر fold فقط imputer، scaler و encoder خودش را روی ردیف‌های آموزشی همان fold fit می‌کند.""",
        9: """## ساخت boosting با دست روی یک مسئله toy regression

در این بخش چند درخت کم‌عمق را به‌صورت دستی روی یک مثال ساده regression می‌سازیم تا ببینیم اصلاح‌های پیاپی چگونه کار می‌کنند.""",
        11: """شکل پله‌ای انتظار می‌رود، چون درخت‌های کم‌عمق اصلاح‌هایی piecewise-constant تولید می‌کنند. تعداد دورهای boosting بیشتر، معمولاً برازش آموزش را بهتر می‌کند، اما اگر زیاد پیش برویم ریسک overfitting بالا می‌رود.""",
        13: """## از residual تا گرادیان‌های طبقه‌بندی

در طبقه‌بندی، به‌جای residual مستقیم با گرادیان‌های loss کار می‌کنیم. ایده اصلی همان است: هر مرحله باید بخشی از خطای فعلی را کاهش دهد.""",
        15: """## مدل‌سازی: اول baselineهای عملی را مقایسه کنید

قبل از اینکه یک مدل پیچیده را جشن بگیریم، چند baseline ساده و قابل‌فهم را مقایسه می‌کنیم تا بفهمیم پیچیدگی اضافه واقعاً ارزش دارد یا نه.""",
        17: """## ارزیابی: معیارها را به‌عنوان trade-off بخوانید

هر معیار یک بُعد از رفتار مدل را نشان می‌دهد. هدف این نیست که همه را همزمان بیشینه کنیم، بلکه باید ببینیم کدام trade-off با مسئله کسب‌وکار سازگارتر است.""",
        18: """## ظرفیت مدل را با یک holdout داخلی تنظیم کنید

یک شکاف رو‌به‌افزایش بین loss آموزش و holdout نشانه هشدار است. درخت‌های بیشتر می‌توانند loss آموزش را بهتر کنند، اما اگر عملکرد holdout افت کند یعنی model capacity از داده فراتر رفته است.""",
        20: """## یک‌بار refit کنید، بعد یک‌بار روی validation ارزیابی کنید

پس از انتخاب تنظیمات، مدل را یک بار روی داده مناسب refit می‌کنیم و فقط یک بار روی validation گزارش می‌دهیم. این کار از بیش‌برازش به validation جلوگیری می‌کند.""",
        23: """## تفسیر: مدل چه چیزی یاد گرفته است

پس از آموزش، می‌خواهیم بفهمیم کدام ویژگی‌ها و الگوها بیشترین نقش را در تصمیم‌های مدل داشته‌اند. تفسیر، جایگزین ارزیابی نیست؛ مکمل آن است.""",
        25: """## توصیه عملی

Gradient boosting زمانی می‌ارزد که baselineهای ساده را شکست دهد، نشت نداشته باشد، و trade-offهایش با هدف کسب‌وکار هم‌راستا باشند. اگر تفاوت عملکرد کوچک باشد، سادگی و پایداری مدل‌های ساده‌تر معمولاً انتخاب بهتری است.""",
        26: """## تمرین‌ها

- یک معیار اضافی تعریف کنید که برای این کمپین بازاریابی مهم باشد.
- learning rate و تعداد درخت‌ها را طوری تغییر دهید که overfitting را ببینید.
- توضیح دهید چرا preprocessing باید داخل pipeline بماند.""",
    },
    "02_advanced_feature_engineering.ipynb": {
        0: "# 02 — مهندسی ویژگی پیشرفته",
        1: """## مفهوم: ویژگی‌های categorical بدون نشت برچسب

ویژگی‌های categorical می‌توانند اطلاعات مفیدی بسازند، اما اگر آمار target را به‌صورت نادرست استفاده کنیم، به‌سادگی leakage ایجاد می‌شود. راه امن این است که encodingها و target statistics فقط روی داده آموزشی و با ترتیب مناسب محاسبه شوند.""",
        4: """## چک مفهومی: target statistics ترتیبی، سطر به سطر

در encodingهای مبتنی بر target، هر ردیف باید فقط از اطلاعات ردیف‌های قبلی یا fold آموزشی خودش استفاده کند. اگر میانگین target را روی کل داده حساب کنیم، اطلاعات آینده وارد آموزش می‌شود.""",
        6: "## ممیزی داده",
        8: """## فرضیه‌های ویژگی پیش از کدنویسی

پیش از ساخت ویژگی جدید، مشخص می‌کنیم کدام الگوها از نظر دامنه مسئله منطقی‌اند: سابقه تماس، فشار کمپین، وضعیت مالی، و الگوهای سنی. این کار از feature engineering تصادفی جلوگیری می‌کند.""",
        11: """## تشخیص این‌که آیا ویژگی‌های مهندسی‌شده target را جدا می‌کنند یا نه

بعد از ساخت ویژگی‌ها، باید بررسی کنیم آیا آن‌ها واقعاً جداسازی target را بهتر کرده‌اند یا فقط پیچیدگی اضافه کرده‌اند.""",
        13: """## معیارها و قضاوت baseline

هر feature set جدید را نسبت به یک baseline روشن می‌سنجیم. اگر بهبود معنی‌دار نباشد، ویژگی‌های اضافی ممکن است فقط هزینه نگه‌داری و خطر leakage را بالا ببرند.""",
        15: "## پیش‌پردازش",
        17: """## مدل‌سازی: اول baselineهای عملیِ قوی

ما با مدل‌های عملی و معقول شروع می‌کنیم و فقط بعد از آن سراغ روش‌های پیچیده‌تر می‌رویم.""",
        19: """## ارزیابی: cross-validation و ablation

cross-validation نشان می‌دهد نتایج چقدر پایدارند. ablation کمک می‌کند بفهمیم کدام گروه ویژگی واقعاً ارزش افزوده ایجاد کرده است.""",
        21: """## اعتبارسنجی ویژگی

هر ویژگی جدید باید از سه زاویه بررسی شود: منطقی بودن از دید دامنه، عدم نشت اطلاعات، و اثر واقعی روی عملکرد.""",
        23: """## تفسیر: ببینیم pipeline چه آموخته است

پس از آموزش، باید بفهمیم کدام ویژگی‌ها و تبدیل‌ها بیشترین اثر را داشته‌اند و آیا این الگوها با انتظار دامنه هماهنگ‌اند یا نه.""",
        25: """## benchmark پیشرفته: مدیریت native categorical با CatBoost

CatBoost می‌تواند برای برخی مسائل categorical را به‌صورت native مدیریت کند. این بخش برای مقایسه است تا ببینیم آیا این رویکرد نسبت به pipeline دستی مزیت عملی دارد یا نه.""",
        27: """## توصیه عملی

ویژگی‌های مهندسی‌شده زمانی ارزش دارند که بهبود پایدار، قابل‌توضیح، و leakage-safe ایجاد کنند. اگر شک دارید، یک آزمایش کوچک و تمیز بهتر از یک feature set بزرگ و مبهم است.""",
    },
    "03_imbalanced_learning.ipynb": {
        0: "# 03 - یادگیری روی داده نامتوازن",
        1: "## مفهوم: نامتوازن بودن چه چیزی را تغییر می‌دهد",
        2: "## راه‌اندازی",
        4: "## بارگذاری داده و ممیزی کلاس‌ها",
        6: "نرخ prevalence در validation باید به development نزدیک باشد چون split به‌صورت stratified انجام شده است. این به معنی sampling تصادفی کامل نیست، اما توزیع کلاس‌ها را پایدار نگه می‌دارد.",
        8: "## استراتژی مدل‌سازی: از ساده شروع کنید",
        10: "## مقایسه cross-validation",
        12: "## سنتز آگاه از categorical با SMOTENC",
        14: "مقایسه بصری یک trade-off رایج در داده نامتوازن را روشن‌تر می‌کند: روش‌هایی که مثبت‌های بیشتری را بازیابی می‌کنند اغلب false positive بیشتری هم ایجاد می‌کنند.",
        16: "## تنظیم آستانه روی پیش‌بینی‌های out-of-fold",
        18: "ماتریس‌های confusion تغییرات معیارها را به count تبدیل می‌کنند. این اغلب مفیدترین نما برای ذی‌نفعان غیر فنی است.",
        20: "این منحنی آستانه باید به‌عنوان یک ابزار تصمیم‌گیری خوانده شود، نه جست‌وجوی یک ثابت جهانی. اگر هزینه کمپین، ارزش مشتری، یا ظرفیت تماس تغییر کند، آستانه نیز باید دوباره تنظیم شود.",
        22: "## کالیبراسیون احتمال",
        24: "## تفسیر و توصیه عملی",
        25: "## اشتباهات رایج و هشدارهای نشت",
    },
    "04_optuna_hyperparameter_optimization.ipynb": {
        0: "# 04 — بهینه‌سازی hyperparameter با Optuna",
        3: "## Optuna مسئول چه چیزی است",
        5: "## طراحی study در Optuna",
        10: "یک بهبود خیلی کوچک در validation ممکن است با نوسان نمونه‌گیری از بین برود و هزینه جست‌وجوی اضافه را توجیه نکند.",
    },
    "05_ensemble_learning.ipynb": {
        0: "# 05 - یادگیری ensemble",
        1: "## نقشه مفهومی",
        3: "## 1. داده را بارگذاری کنید",
        6: "### تفسیر",
        7: "## 2. یک split توسعه و اعتبارسنجی بسازید",
        9: "## 3. یک pipeline پیش‌پردازش روشن بسازید",
        12: "## 4. مدل‌های baseline عملی را آموزش دهید",
        15: "### راهنمای خواندن baseline",
        16: "## 5. رأی‌گیری نرم",
        18: "### بحث رأی‌گیری",
        19: "## 6. Stacking",
        21: "### بحث stacking",
        22: "## 7. همه نامزدها را مقایسه کنید",
        26: "### نقطه تصمیم توصیه",
        27: "## 8. آستانه تصمیم را تنظیم کنید",
        29: "### بحث آستانه",
        30: "## 9. کالیبراسیون احتمال را بررسی کنید",
        32: "## توصیه عملی نهایی",
        33: "## اشتباهات رایج و هشدارهای نشت",
    },
    "06_anomaly_detection_extension.ipynb": {
        0: "# 07 — افزونه عملی تشخیص ناهنجاری",
        2: "## مفهوم: چه چیزی ناهنجاری محسوب می‌شود؟",
        3: "## بارگذاری داده",
        5: "نرخ اشتراک فقط برای جهت‌گیری نشان داده شده است. یک کلاس supervised کمیاب، به‌طور خودکار کلاس ناهنجاری نیست. اگر آن را چنین فرض کنیم، مسئله را اشتباه صورت‌بندی می‌کنیم.",
        6: "## پیش‌پردازش",
        8: "برای detectorهای واقعی پایین‌تر، preprocessing داخل هر pipeline قرار می‌گیرد. پیش‌نمایش بالا فقط برای آموزش است؛ هر detector باید ویژگی‌ها را فقط با داده همان مسیر خودش ببیند.",
        9: "## برچسب‌های اعتبارسنجی کنترل‌شده",
        11: "## baseline عملی: قوانین ساده کسب‌وکار",
        13: "امتیاز قانون 0 یعنی هیچ‌یک از فیلدهای تحت نظر از آستانه خود عبور نکرده‌اند. امتیازهای بالاتر یعنی نقض قانون‌های بیشتر.",
        14: "## detectorهای مبتنی بر مدل",
        16: "## ارزیابی با بودجه بررسی",
        19: "## تفسیر",
        21: "ردیف‌های با بالاترین امتیاز، اولین ردیف‌هایی هستند که یک بازبین انسانی می‌بیند. در یک سیستم واقعی، این جدول باید با توضیح دلیل هشدار همراه شود تا بررسی سریع‌تر و قابل‌اعتمادتر شود.",
        22: "## توصیه عملی",
    },
    "07_end_to_end_production_ml_project.ipynb": {
        0: "# 07 — پروژه ML تولیدی end-to-end",
        1: "## اهداف یادگیری",
        2: "## 1. صورت‌بندی مسئله",
        4: "## 2. بارگذاری داده محلی",
        6: "یک ممیزی سریع کمک می‌کند قبل از مدل‌سازی، توازن target، نوع ویژگی‌ها و ریسک‌های آشکار leakage را بفهمیم.",
        8: "## 3. آماده‌سازی ویژگی‌ها و split داده",
        10: "## 4. پیش‌پردازش بدون نشت",
        12: "## 5. ساخت baselineهای عملی",
        14: "## 6. مقایسه مدل‌ها روی داده validation",
        16: "منحنی‌های بصری، trade-off را برای آموزش ساده‌تر می‌کنند. یک مدل بهتر معمولاً منحنی‌ای دارد که به گوشه بالا-راست نزدیک‌تر است.",
        18: "## 7. انتخاب مدل و آستانه",
        21: "آستانه را فقط با داده validation انتخاب می‌کنیم. از این نقطه به بعد، test باید فقط یک‌بار برای برآورد نهایی استفاده شود.",
        23: "## 8. ارزیابی نهایی روی test",
        25: "## 9. تفسیر",
        27: "## 10. توصیه عملی",
        29: "## 11. ذخیره و بارگذاری دوباره artifact استنتاج",
        31: "تأیید بارگذاری دوباره، اشتباهات رایج استقرار را زود آشکار می‌کند: ستون‌های مفقود، ناهماهنگی preprocessing، یا آستانه‌ای که درست ذخیره نشده است.",
        33: "## 12. بررسی‌های سبک",
        35: "## تمرین‌ها",
        36: "**زمان تخمینی:** 90-120 دقیقه",
    },
}


CODE_REPLACEMENTS = {
    "# Known interoperability/UI warnings do not affect predictions or notebook execution.": "# هشدارهای سازگاری/رابط کاربری روی پیش‌بینی‌ها یا اجرای دفترچه اثر ندارند.",
    "# These UI/interoperability warnings do not change the lesson results.": "# این هشدارهای رابط کاربری/سازگاری نتیجه درس را تغییر نمی‌دهند.",
    "# Return the course root when present, otherwise the notebook's directory.": "# اگر ریشه دوره در دسترس بود همان را برگردان، وگرنه پوشه دفترچه را.",
    "# Set FAST_MODE=0 for full-size experiments; laptop mode is the default.": "# برای آزمایش‌های کامل، FAST_MODE=0 را تنظیم کنید؛ حالت لپ‌تاپ به‌صورت پیش‌فرض فعال است.",
    "# The course ships with a local dataset; notebooks never access the network.": "# این دوره با یک مجموعه‌داده محلی همراه است؛ دفترچه‌ها هرگز به شبکه دسترسی ندارند.",
    "# Load the data, encode y, and exclude post-call duration by default.": "# داده را بارگذاری کن، y را encode کن، و مدت تماس پس از تماس را به‌صورت پیش‌فرض حذف کن.",
    "# Deterministic stratified 60/20/20 split; test stays sealed until notebook 09.": "# split قطعی و stratified به نسبت 60/20/20؛ مجموعه آزمون تا دفترچه 09 قفل می‌ماند.",
    "# Preprocessing is fitted only inside the enclosing training/CV pipeline.": "# preprocessing فقط داخل pipeline آموزش/CV مربوطه fit می‌شود.",
    "# A single untuned baseline gives Optuna something honest to beat.": "# یک baseline بدون تنظیم به Optuna یک هدف منصفانه برای شکست‌دادن می‌دهد.",
    "# Return development-only cross-validated average precision for one Optuna trial.": "# میانگین precision اعتبارسنجیِ cross-validated فقط روی داده توسعه را برای یک trial برگردان.",
    "Precision-recall curve": "منحنی دقت-بازخوانی",
    "ROC curve": "منحنی ROC",
    "Resampling and weighting can distort probabilities": "نمونه‌برداری مجدد و وزن‌دهی می‌توانند احتمال‌ها را مخدوش کنند",
    "Probability calibration": "کالیبراسیون احتمال",
    "Precision-recall": "دقت-بازخوانی",
    "Recall": "بازخوانی",
    "Target counts:": "شمارش‌های هدف:",
    "Subscription rate:": "نرخ اشتراک:",
    "Rows:": "ردیف‌ها:",
    "Features after removing leakage:": "ویژگی‌ها پس از حذف نشت:",
    "Numeric features:": "ویژگی‌های عددی:",
    "Categorical features:": "ویژگی‌های categorical:",
    "Development-only cross-validated average precision": "میانگین precision اعتبارسنجیِ cross-validated روی داده توسعه",
    "Optuna optimization history": "تاریخچه بهینه‌سازی Optuna",
    "Trial": "trial",
    "Lower is better": "کمتر بهتر است",
    "Validation log loss": "log loss اعتبارسنجی",
    "Prediction speed": "سرعت پیش‌بینی",
    "Milliseconds per 1,000 rows": "میلی‌ثانیه برای هر 1,000 ردیف",
    "Validation calibration": "کالیبراسیون اعتبارسنجی",
    "Precision-recall curve": "منحنی دقت-بازخوانی",
    "ROC curve": "منحنی ROC",
    "Chosen model based on validation average precision:": "مدل انتخاب‌شده بر اساس average precision اعتبارسنجی:",
    "Selected threshold:": "آستانه انتخاب‌شده:",
    "Decision threshold": "آستانه تصمیم",
    "Validation trade-offs by threshold": "trade-offهای اعتبارسنجی بر حسب آستانه",
    "Test ROC AUC:": "ROC AUC آزمون:",
    "Test average precision:": "average precision آزمون:",
    "Classification report at selected threshold:": "گزارش طبقه‌بندی در آستانه انتخاب‌شده:",
    "Saved model artifact to:": "artifact مدل ذخیره شد در:",
    "Saved metadata to:": "فراداده ذخیره شد در:",
    "Expected validation error:": "خطای مورد انتظار اعتبارسنجی:",
    "Fitted ": "fit شد: ",
    "positive rate": "نرخ مثبت",
    "validation shape:": "شکل validation:",
    "development shape:": "شکل development:",
    "subscription rate in validation:": "نرخ اشتراک در validation:",
    "first transformed feature names:": "اولین نام‌های ویژگی تبدیل‌شده:",
    "number of transformed features:": "تعداد ویژگی‌های تبدیل‌شده:",
    "injected anomalies:": "ناهنجاری‌های تزریق‌شده:",
    "injected prevalence:": "prevalence تزریق‌شده:",
}


def translate_markdown_cell(text: str) -> str:
    """Translate notebook markdown content when a custom mapping exists."""
    return text


def rtl_markdown_text(text: str) -> str:
    """Wrap markdown text in RTL HTML markup for notebook viewers."""
    # Force explicit RTL layout with inline HTML so notebook viewers do not
    # depend on CSS support for text direction.
    inner_lines = []
    for line in text.splitlines():
        if line.strip():
            inner_lines.append("\u200f" + line)
        else:
            inner_lines.append("")
    inner = "\n".join(inner_lines).strip("\n")
    return f'<div dir="rtl" style="text-align: right">\n\n{inner}\n\n</div>'


def localize_code_cell(text: str) -> str:
    """Apply lightweight text replacements to code cell source text."""
    for line in text.splitlines():
        pass
    for src, dst in CODE_REPLACEMENTS.items():
        text = text.replace(src, dst)
    # Translate brief inline comments line by line when they are simple English notes.
    translated_lines = []
    for line in text.splitlines():
        if "#" in line:
            head, sep, tail = line.partition("#")
            stripped = tail.strip()
            if stripped:
                comment = stripped
                comment = comment.replace("clean and readable", "تمیز و خوانا")
                comment = comment.replace("duration excluded by default", "duration به‌صورت پیش‌فرض حذف می‌شود")
                comment = comment.replace("positive class", "کلاس مثبت")
                comment = comment.replace("business-cost proxy", "تقریب هزینه کسب‌وکار")
                comment = comment.replace("false negatives are weighted 5x false positives because missing a likely", "منفی‌های کاذب 5 برابر مثبت‌های کاذب وزن می‌گیرند چون از دست‌دادن یک")
                comment = comment.replace("subscriber wastes a scarce call opportunity and can lose revenue, while a", "مشترک بالقوه فرصت تماس کمیاب را هدر می‌دهد و می‌تواند به از دست‌رفتن درآمد منجر شود، در حالی که یک")
                comment = comment.replace("false positive mainly spends a call on someone unlikely to convert.", "مثبت کاذب عمدتاً فقط یک تماس را صرف فردی می‌کند که احتمال تبدیل شدنش پایین است.")
                comment = comment.replace("Ranking quality on the positive class, which is more useful for rare events.", "کیفیت رتبه‌بندی برای کلاس مثبت، که برای رخدادهای کمیاب مفیدتر است.")
                comment = comment.replace("Among predicted positives, how many are actually positive.", "از میان مثبت‌های پیش‌بینی‌شده، چند مورد واقعاً مثبت‌اند.")
                comment = comment.replace("Among actual positives, how many we successfully found.", "از میان مثبت‌های واقعی، چند مورد را درست پیدا کرده‌ایم.")
                translated_lines.append(head + "# " + comment)
            else:
                translated_lines.append(line)
        else:
            translated_lines.append(line)
    text = "\n".join(translated_lines)
    return text


def localize_output_text(text: str) -> str:
    """Localize selected output strings for the Farsi notebook copy."""
    for src, dst in CODE_REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text


def localize_outputs(cell: dict) -> None:
    """Rewrite text outputs in-place for the localized notebook copy."""
    outputs = cell.get("outputs") or []
    for output in outputs:
        if "text" in output:
            if isinstance(output["text"], list):
                output["text"] = [localize_output_text(chunk) for chunk in output["text"]]
            else:
                output["text"] = localize_output_text(output["text"])
        if "data" in output and isinstance(output["data"], dict):
            for key in ("text/plain", "text/html", "text/markdown"):
                if key in output["data"]:
                    value = output["data"][key]
                    if isinstance(value, list):
                        output["data"][key] = [localize_output_text(chunk) for chunk in value]
                    else:
                        output["data"][key] = localize_output_text(value)


def build_notebooks() -> None:
    """Build the localized Farsi notebook set from the English source files."""
    OUT_DIR.mkdir(exist_ok=True)
    for src_path in sorted(SRC_DIR.glob("*.ipynb")):
        nb = json.loads(src_path.read_text(encoding="utf-8"))
        mapping = TRANSLATIONS.get(src_path.name, {})
        if nb.get("cells"):
            nb["cells"].insert(1 if nb["cells"][0].get("cell_type") == "markdown" else 0, deepcopy(FAIRY_SETUP_CELL))
            nb["cells"].insert(2 if nb["cells"][0].get("cell_type") == "markdown" else 1, deepcopy(RTL_STYLE_CELL))
        for idx, cell in enumerate(nb.get("cells", [])):
            source = "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else cell.get("source", "")
            if cell.get("cell_type") == "markdown":
                if idx in mapping:
                    source = rtl_markdown_text(mapping[idx])
                else:
                    source = rtl_markdown_text(source)
            elif cell.get("cell_type") == "code":
                source = localize_code_cell(source)
                localize_outputs(cell)
            cell["source"] = source.splitlines(keepends=True)
        out_path = OUT_DIR / src_path.name
        out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    build_notebooks()
