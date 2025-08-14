from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class SMSExample:
    text: str
    label: int  # 0=ham, 1=spam

def load_sms_spam(root: str | Path) -> List[SMSExample]:
    p = Path(root) / "sms_spam_collection"
    data_file = p / "SMSSpamCollection"
    items: List[SMSExample] = []
    if not data_file.exists():
        return items
    for line in data_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        head, text = line.split("\t", 1)
        label = 1 if head.strip().lower() == "spam" else 0
        items.append(SMSExample(text=text, label=label))
    return items
