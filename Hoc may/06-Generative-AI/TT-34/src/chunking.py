# Chia CV theo mục, rồi chia tiếp mục dài thành các cửa sổ vừa giới hạn token của model.
SECTION_KEYS = {
    'kinh_nghiem': ['kinh nghiệm', 'work experience', 'professional experience', 'experience', 'work history', 'employment'],
    'ky_nang':     ['kỹ năng', 'technical skills', 'skills', 'skill highlights', 'core qualifications', 'qualifications', 'highlights'],
    'du_an':       ['dự án', 'projects', 'project', 'accomplishments', 'core accomplishments'],
    'hoc_van':     ['học vấn', 'education'],
}


def split_cv(text):
    """Trả về {tên_mục: nội dung}. Phần không thuộc mục nào (tên, tóm tắt...) vào 'khac'."""
    sections, current = {}, 'khac'
    for line in text.splitlines():
        s = line.strip().lower().strip(':#*-_ ')
        key = next((k for k, ws in SECTION_KEYS.items()
                    if len(s) < 40 and any(s.startswith(w) for w in ws)), None)
        if key:                       # dòng tiêu đề mục -> đổi mục hiện tại
            current = key
        else:
            sections.setdefault(current, []).append(line)
    sections = {k: '\n'.join(v).strip() for k, v in sections.items()}
    return {k: v for k, v in sections.items() if v}


def windows(text, size):
    """Cắt văn bản thành các đoạn <= size từ để không bị model cắt cụt âm thầm."""
    w = text.split()
    return [' '.join(w[i:i + size]) for i in range(0, max(len(w), 1), size)]
