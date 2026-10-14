# Lấy mẫu CV từ Resume.csv (Kaggle snehaanbhawal/resume-dataset), tách mục từ HTML, ẩn danh.
import re
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup

# 5 ngành khớp 5 JD + FINANCE làm nhóm "gần nghĩa" với ACCOUNTANT (tạo nhãn trung gian 1-2)
CATS = ['INFORMATION-TECHNOLOGY', 'HR', 'ACCOUNTANT', 'FINANCE', 'SALES', 'TEACHER']
PII = [(r'[\w.+-]+@[\w-]+\.[\w.]+', '[EMAIL]'), (r'https?://\S+|www\.\S+', '[URL]'),
       (r'\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}', '[PHONE]')]


def html_to_text(html):
    """Dùng Resume_html để giữ đúng ranh giới mục: mỗi tiêu đề mục nằm trên một dòng riêng."""
    out = []
    for sec in BeautifulSoup(html, 'html.parser').find_all('div', class_='section'):
        title = sec.find(class_='sectiontitle')
        head = title.get_text(' ', strip=True).upper() if title else ''
        if title:
            title.decompose()
        # chỉ xuống dòng ở thẻ khối (p, div, li, br); thẻ inline (span) nối bằng dấu cách
        html_blk = re.sub(r'<\s*/?\s*(p|div|li|br|tr)\b[^>]*>', '\n', str(sec))
        body = [l for l in (' '.join(x.split()) for x in BeautifulSoup(html_blk, 'html.parser').get_text(' ').splitlines()) if l]
        out += ([head] if head else []) + body
    return '\n'.join(out)


def anonymize(text):
    for pat, rep in PII:
        text = re.sub(pat, rep, text)
    return text


def prepare(csv_path, out_dir, cats=CATS, per_cat=5, min_words=500, seed=42):
    """Chọn CV dài (>= min_words) để vấn đề cắt cụt token thể hiện rõ. File: <NGÀNH>__<ID>.txt"""
    df = pd.read_csv(csv_path, usecols=['ID', 'Resume_html', 'Category'])
    df = df[df.Category.isin(cats)].copy()
    df['text'] = df.Resume_html.map(html_to_text).map(anonymize)
    df = df[df.text.str.split().str.len() >= min_words]
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for cat in cats:
        for r in df[df.Category == cat].sample(per_cat, random_state=seed).itertuples():
            (out_dir / f'{cat}__{r.ID}.txt').write_text(r.text, encoding='utf-8')
